"""E2 동일 배치 분기 (연구 명세 v0.10, 3.6절).

- 네 칸: O± (outcome), O⁺ (outcome_masked), C± (gae = A3⁰의 원래 업데이트), C⁺ (gae_masked).
  모든 칸이 같은 체크포인트·배치·critic·value 손실 경로·클리핑·optimizer 상태(복사)·미니배치 순서·학습률을 쓴다.
- E2-a 누적 기울기 진단: 공통 전체 배치에서 가중치만 바꾼 누적 기울기 하나 (여러 optimizer 스텝이 아님).
  세 층위: (i) 정책 손실만의 기울기, (ii) 정책 + value 총 기울기, (iii) optimizer를 거친 실제 Δθ.
  행마다 cos(g±, g⁺)와 노름 비율 ‖g⁺‖/‖g±‖. 노름이 0에 가까우면 코사인은 "정의 불가"(None).
  E2-a는 복제본(스냅샷 복원)에서 하고 결과만 남긴다.
- E2-b 첫 업데이트: 분기 체크포인트로 복원한 뒤 공통 배치에 각 규칙으로 한 라운드 업데이트.
  C±의 첫 업데이트는 같은 순서 시드로 계보의 실제 다음 상태를 재현해야 한다 (P4 'E2 분기 코드').
"""
from __future__ import annotations

import numpy as np
import torch

import core
import ppo

RULES = {"O±": "outcome", "O⁺": "outcome_masked", "C±": "gae", "C⁺": "gae_masked"}
EPS_NORM = 1e-12


@torch.no_grad()
def _recompute_all(adapter, n, chunk):
    adapter.train_mode()
    lp_all, v_all = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for j in range(0, n, chunk):
        idx = np.arange(j, min(n, j + chunk))
        lp, v = adapter.evaluate(idx)
        lp_all[idx] = lp[:, 0].float().cpu().numpy()
        v_all[idx] = v[:, 0].float().cpu().numpy()
    return lp_all, v_all


def _grad_vector(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().reshape(-1).cpu()
                      for p in params])


def _param_vector(params):
    return torch.cat([p.detach().float().reshape(-1).cpu() for p in params])


def full_batch_gradient(adapter, batch, rule, old_lp, old_v, cfg: ppo.PPOConfig, include_value: bool):
    """전체 배치 평균 손실의 기울기 하나 (θ = θ_old 그대로, optimizer 스텝 없음)."""
    T, N = batch.T, batch.N
    n = T * N
    adv, ret, _ = ppo.advantages_and_returns(batch, old_v.reshape(T, N), cfg.reward_kind, rule)
    adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
    params = list(adapter.params_policy) + list(adapter.params_value)
    for p in params:
        p.grad = None
    adapter.train_mode()
    for j in range(0, n, cfg.chunk):
        idx = np.arange(j, min(n, j + cfg.chunk))
        lp, v = adapter.evaluate(idx)
        dev = lp.device
        o = torch.as_tensor(old_lp[idx], device=dev).view(-1, 1)
        a = torch.as_tensor(adv_f[idx], device=dev).view(-1, 1)
        loss = cfg.policy_coef * core.policy_loss_controlled(lp, o, a, cfg.clip)
        if include_value:
            vp = torch.as_tensor(old_v[idx], device=dev).view(-1, 1)
            r = torch.as_tensor(ret_f[idx], device=dev).view(-1, 1)
            loss = loss + cfg.value_coef * core.value_loss_original(v, vp, r, cfg.clip, cfg.huber_delta)
        (loss * len(idx) / n).backward()
    return params


def _cos_ratio(a, b):
    na, nb = float(a.norm()), float(b.norm())
    if na < EPS_NORM or nb < EPS_NORM:
        return dict(cos=None, norm_ratio=None, norm_pm=na, norm_plus=nb, note="undefined (near-zero norm)")
    return dict(cos=float(torch.dot(a, b) / (na * nb)), norm_ratio=nb / na, norm_pm=na, norm_plus=nb)


def e2a_diagnostics(adapter, batch, snap, cfg: ppo.PPOConfig, policy_only_params=None):
    """세 층위 진단. snap = 분기 지점 스냅샷 (ppo.snapshot). 끝나면 snap으로 되돌린다."""
    n = batch.T * batch.N
    ppo.restore(adapter, snap)
    old_lp, old_v = _recompute_all(adapter, n, cfg.chunk)
    pol_params = list(policy_only_params) if policy_only_params is not None else list(adapter.params_policy)
    all_params = list(adapter.params_policy) + list(adapter.params_value)
    g_pol, g_tot, dtheta = {}, {}, {}
    for name, rule in RULES.items():
        ppo.restore(adapter, snap)
        full_batch_gradient(adapter, batch, rule, old_lp, old_v, cfg, include_value=False)
        g_pol[name] = _grad_vector(pol_params)
        full_batch_gradient(adapter, batch, rule, old_lp, old_v, cfg, include_value=True)
        g_tot[name] = _grad_vector(all_params)
        before = _param_vector(all_params)
        torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        adapter.opt_value.step()
        adapter.opt_policy.step()
        dtheta[name] = _param_vector(all_params) - before
    ppo.restore(adapter, snap)
    for p in all_params:
        p.grad = None
    out = {}
    for row, (pm, pl) in {"O": ("O±", "O⁺"), "C": ("C±", "C⁺")}.items():
        out[row] = dict(policy_grad=_cos_ratio(g_pol[pm], g_pol[pl]), total_grad=_cos_ratio(g_tot[pm], g_tot[pl]),
                        delta_theta=_cos_ratio(dtheta[pm], dtheta[pl]))
    Y = batch.Y()
    out["batch"] = dict(N=batch.N, Y_mean=float(Y.mean()), degenerate=bool(Y.min() == Y.max()))
    return out


def branch_first_update(adapter, batch, snap, rule_name, cfg: ppo.PPOConfig, log=None):
    """E2-b 1라운드째: 분기 체크포인트로 복원하고 공통 배치에 규칙 하나로 한 라운드 업데이트."""
    ppo.restore(adapter, snap)
    c = ppo.PPOConfig(**{**cfg.__dict__, "adv_rule": RULES[rule_name]})
    return ppo.ppo_update(adapter, batch, c, log=log)
