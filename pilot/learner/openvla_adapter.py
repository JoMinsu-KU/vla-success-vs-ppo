"""ppo.py 어댑터: 원 RL4VLA OpenVLAPolicy(PEFT LoRA + value head 'a0')를 연결한다.

evaluate(idx): 저장 배치의 전이 idx(t·N + i)에 대해 원 `OpenVLAPolicy.evaluate_actions`(teacher forcing)로
joint log-prob와 value를 계산한다. 수집은 `collect_round`로 하며 원 `get_action`이 돌려주는 생성 경로의
log-prob·value를 그대로 저장한다 (C10 비교용).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ppo  # noqa: E402


def build_policy(ckpt: str, seed: int, lr: float = 1e-4, vh_lr: float = 3e-3):
    """병합 W1 + 새 LoRA(원 PPO 레시피 LoraConfig, gaussian A·0 B) + value head. 초기화 시드 = 학습 시드."""
    import p2_eval  # noqa: F401  (win_compat 등록)
    from prismatic.extern.hf import modeling_prismatic as mp
    from simpler_env.policies.openvla import openvla_train as ot
    cls = mp.OpenVLAForActionPredictionWithValueHead
    if not getattr(cls, "_p2_attn_patched", False):
        orig = cls.from_pretrained.__func__

        @classmethod
        def fp(c, *a, **kw):
            if kw.get("attn_implementation") == "flash_attention_2":
                kw["attn_implementation"] = "sdpa"
            return orig(c, *a, **kw)
        cls.from_pretrained = fp
        cls._p2_attn_patched = True
    args = SimpleNamespace(vla_path=ckpt, vla_unnorm_key="bridge_orig", vla_load_path="", vla_lora_rank=32,
                           vla_lr=lr, vla_vhlr=vh_lr, vla_optim_beta1=0.9, vla_optim_beta2=0.999,
                           vla_temperature=1.0, vla_temperature_eval=0.6)
    torch.manual_seed(seed)
    return ot.OpenVLAPolicy(args, device_id=0)


class OpenVLAAdapter:
    def __init__(self, policy, obs: np.ndarray, instr: list, actions: np.ndarray):
        self.policy = policy
        self.obs = obs            # [T, N, H, W, 3] uint8
        self.instr = instr        # [N]
        self.actions = actions    # [T, N, 7]
        self.N = actions.shape[1]
        self.params_policy = policy.params_vla
        self.params_value = policy.params_vh
        self.opt_policy = policy.vla_optimizer
        self.opt_value = policy.vh_optimizer
        self.dev = torch.device("cuda:0")

    def evaluate(self, idx):
        idx = np.asarray(idx)
        t, i = idx // self.N, idx % self.N
        x = dict(image=torch.from_numpy(self.obs[t, i]).to(self.dev), task_description=[self.instr[k] for k in i])
        a = torch.from_numpy(self.actions[t, i]).to(self.dev)
        lp, _ent, v = self.policy.evaluate_actions(x, a)
        return lp.float(), v.float()

    def train_mode(self):
        self.policy.prep_training()

    def eval_mode(self):
        self.policy.prep_rollout()


def collect_round(policy, make_pool, specs, temperature=1.0, topk="K0", seed=0, log=print):
    """환경 풀로 에피소드를 모아 RoundBatch와 관측을 만든다 (원 get_action: value·action·log-prob, 생성 경로)."""
    import batched_eval as be
    be.set_top_k(policy, topk)
    pool = make_pool()
    n_env = len(pool.conns)
    dev = torch.device("cuda:0")
    T, N = 80, len(specs)
    obs = np.zeros((T, N, 480, 640, 3), np.uint8)
    actions = np.zeros((T, N, 7), np.int64)
    lp_gen = np.zeros((T, N), np.float32)
    v_gen = np.zeros((T, N), np.float32)
    succ = np.zeros((T, N), bool)
    grasp = np.zeros((T, N), bool)
    consec = np.zeros((T, N), bool)
    instr = [None] * N
    policy.prep_rollout()
    t0 = time.time()
    for w0 in range(0, N, n_env):
        wave = specs[w0:w0 + n_env]
        torch.manual_seed(seed * 1_000_003 + w0)
        resets = pool.reset(wave)
        o = [r[0] for r in resets]
        ins = [r[1] for r in resets]
        for k, s in enumerate(ins):
            instr[w0 + k] = s
        for t in range(T):
            x = dict(image=torch.from_numpy(np.stack(o)).to(dev), task_description=list(ins))
            with torch.no_grad():
                v, a, lp = policy.get_action(x, deterministic=(temperature != 1.0))
            for k in range(len(wave)):
                obs[t, w0 + k] = o[k]
            actions[t, w0:w0 + len(wave)] = a.cpu().numpy()
            lp_gen[t, w0:w0 + len(wave)] = lp[:, 0].float().cpu().numpy()
            v_gen[t, w0:w0 + len(wave)] = v[:, 0].float().cpu().numpy()
            outs = pool.step(list(a.cpu().numpy()))
            o = [x_[0] for x_ in outs]
            for k, (_o, si) in enumerate(outs):
                succ[t, w0 + k] = si["success"]
                grasp[t, w0 + k] = si["grasped_latch"]
                consec[t, w0 + k] = si["consecutive_latch"]
    pool.close()
    log(f"[collect] {N} episodes in {(time.time() - t0) / 60:.1f} min, Y = {(succ[-1] & grasp[-1]).sum()}/{N}")
    batch = ppo.RoundBatch(actions=actions, logprob_gen=lp_gen, value_gen=v_gen, success=succ,
                           grasp_latch=grasp, consec_latch=consec, extra=dict(obs=obs, instr=instr))
    return batch
