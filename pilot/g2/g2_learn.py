"""G2 S7·S9 공통: 위치별 행동 분포(생성 경로·teacher forcing 경로), KL 탐침, 학습 상태 저장.

- gen_full: 원 predict_action_batch와 같은 계산(같은 generate 호출·같은 logits processor)에 더해,
  처리 전 output.logits의 256개 행동 토큰 slice log-softmax [B,7,256] (= 저장 log-prob가 쓰는 분포 q₀)를 돌려준다.
- tf_full: 원 evaluate_action과 같은 계산에 더해 [B,7,256] log-softmax (= 학습 경로 분포 p₀)를 돌려준다.
- probe_kl: 고정 rollout-prefix 조건부 KL(π_ref ‖ π_now)의 위치 7개 합을 탐침 상태 평균 (명세 3.3절 v0.12 (a)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

A0 = 32000 - 256


@torch.no_grad()
def gen_full(policy, x: dict, temperature: float):
    """반환: values [B,1], actions [B,7], logprob_joint [B,1], logp_dist [B,7,256] (float32), pad [B]."""
    from prismatic.extern.hf.modeling_prismatic import AllowedTokensLogitsProcessor
    from transformers import LogitsProcessorList
    vla = policy.vla
    feats = policy._preprocess_obs(x)
    out = vla.generate(feats["input_ids"], attention_mask=feats["attention_mask"], pixel_values=feats["pixel_values"],
                       max_new_tokens=7, return_dict_in_generate=True, output_hidden_states=True, output_logits=True,
                       logits_processor=LogitsProcessorList([AllowedTokensLogitsProcessor()]),
                       do_sample=temperature != 0.0, temperature=temperature)
    ids = out.sequences[:, -7:]
    lt = torch.stack(out.logits, dim=1)[:, :, A0:32000]
    lpt = F.log_softmax(lt, dim=-1)
    lp = torch.gather(lpt, 2, (ids - A0).unsqueeze(-1)).squeeze(-1).sum(1, keepdim=True)
    h = out.hidden_states[0][-1][:, -1]
    base = vla.base_model.model if hasattr(vla, "base_model") else vla
    v = base.value_head(h)
    pad = (feats["attention_mask"] == 0).sum(1)
    return v, ids, lp, lpt.float(), pad


def tf_full(policy, x: dict, actions: torch.Tensor):
    """teacher forcing (원 evaluate_action과 같은 forward). 반환: logprob_joint [B,1], value [B,1], logp_dist [B,7,256], pad [B].
    기울기가 필요하면 바깥에서 grad 모드로 부른다."""
    vla = policy.vla
    feats = policy._preprocess_obs(x, actions)
    base = vla.base_model.model if hasattr(vla, "base_model") else vla
    # PeftModel(...) → PrismaticForConditionalGeneration.forward (원 evaluate_action의 super().forward와 같은 계산)
    outputs = vla(input_ids=feats["input_ids"], attention_mask=feats["attention_mask"], pixel_values=feats["pixel_values"],
                  labels=feats["labels"], output_hidden_states=True, return_dict=True)
    h = outputs.hidden_states[-1][:, -9]
    v = base.value_head(h)
    lt = outputs.logits[:, -9:-2, A0:32000]
    lpt = F.log_softmax(lt, dim=-1)
    idx = (feats["labels"][:, -8:-1] - A0).unsqueeze(-1)
    lp = torch.gather(lpt, 2, idx).squeeze(-1).sum(1, keepdim=True)
    pad = (feats["attention_mask"] == 0).sum(1)
    return lp, v, lpt.float(), pad


def check_tf_matches_original(policy, x, actions, atol=0.0):
    """tf_full이 원 evaluate_actions와 같은 값을 내는지 (같은 forward이므로 비트 단위로 같아야 함)."""
    with torch.no_grad():
        lp0, _e, v0 = policy.evaluate_actions(x, actions)
        lp1, v1, _d, _p = tf_full(policy, x, actions)
    return dict(lp_max_abs=float((lp0.float() - lp1.float()).abs().max()), v_max_abs=float((v0.float() - v1.float()).abs().max()))


# ---------------------------------------------------------------- KL 탐침
class Probe:
    """라운드 배치에서 성공·실패 무관 균등하게 고른 상태 n개 (관측·지시문·수집 때 실제 행동 토큰)."""

    def __init__(self, obs, instr, actions, n=64, seed=0):
        T, N = actions.shape[:2]
        rng = np.random.default_rng(seed)
        flat = rng.choice(T * N, size=min(n, T * N), replace=False)
        self.t, self.i = flat // N, flat % N
        self.obs = obs[self.t, self.i]
        self.instr = [instr[k] for k in self.i]
        self.act = actions[self.t, self.i]
        self.ref = None

    @torch.no_grad()
    def dists(self, policy, chunk=16):
        dev = torch.device("cuda:0")
        out = []
        for j in range(0, len(self.act), chunk):
            x = dict(image=torch.from_numpy(self.obs[j:j + chunk]).to(dev), task_description=self.instr[j:j + chunk])
            _lp, _v, d, _p = tf_full(policy, x, torch.from_numpy(self.act[j:j + chunk]).to(dev))
            out.append(d)
        return torch.cat(out)  # [n,7,256] log-prob

    def set_ref(self, policy):
        self.ref = self.dists(policy)

    def kl(self, policy) -> float:
        now = self.dists(policy)
        p = self.ref.exp()
        kl = (p * (self.ref - now)).sum(-1)  # [n,7]
        return float(kl.sum(-1).mean())


# ---------------------------------------------------------------- 학습 상태 저장·불러오기
def save_trainable(path: Path, adapter_or_policy, extra: dict | None = None):
    pol = getattr(adapter_or_policy, "policy", adapter_or_policy)
    names = [n for n, p in pol.vla.named_parameters() if p.requires_grad]
    state = {n: p.detach().cpu() for n, p in pol.vla.named_parameters() if p.requires_grad}
    torch.save(dict(params=state, names=names, opt_vla=pol.vla_optimizer.state_dict(), opt_vh=pol.vh_optimizer.state_dict(),
                    extra=extra or {}), path)


def load_trainable(path: Path, policy, load_optim: bool = False):
    ck = torch.load(path, map_location="cpu")
    named = dict(policy.vla.named_parameters())
    with torch.no_grad():
        for n, t in ck["params"].items():
            named[n].copy_(t.to(named[n].device))
    if load_optim:
        policy.vla_optimizer.load_state_dict(ck["opt_vla"])
        policy.vh_optimizer.load_state_dict(ck["opt_vh"])
    return ck.get("extra", {})


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    vr = float(np.var(returns))
    return float("nan") if vr == 0 else 1.0 - float(np.var(returns - values)) / vr


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
