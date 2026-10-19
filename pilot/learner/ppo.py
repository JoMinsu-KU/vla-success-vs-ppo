"""통제 연구 PPO 학습기 (A3⁰ / A3ˢ, E2 네 칸 공통). 연구 명세 v0.10, 3.2절 C5·C7·C9·C10, 3.6절.

모델에 의존하지 않는 업데이트 로직과, 정책을 연결하는 어댑터 인터페이스로 나눈다.

어댑터(PolicyAdapter)가 제공해야 하는 것:
    evaluate(idx: np.ndarray) -> (logprob[B,1] float32 tensor, value[B,1] float32 tensor)
        저장 배치의 전이 idx(평탄화 번호 t·N + i)에 대해 teacher forcing으로 joint log-prob(7토큰 합)와 value를 계산.
    params_policy, params_value: 기울기 클리핑 대상 파라미터 목록 (원 코드: LoRA + value head 함께 노름 10)
    opt_policy, opt_value: 원 코드처럼 두 optimizer (LoRA lr 1e-4, value head lr 3e-3)
    train_mode(), eval_mode()

원 코드(openvla_train.OpenVLAPPO.train_ppo_step)와의 관계:
    - 손실: 정책 손실 + value 손실(클리핑한 예측과의 Huber 최대값) − 엔트로피(계수 0). 한 번 역전파, 공동 클리핑 10.
    - 원 코드형(mode="original"): old log-prob 7칸 복제 → 정책 손실이 수식의 7배, 마지막 부분 묶음도 항상 누적 20으로 나눔,
      old log-prob·value는 생성 때 저장값.
    - 통제형(mode="controlled"): joint log-prob 1개로 정책 손실을 한 번 계산하고 계수 7을 곱함(C9, 총 손실 동일),
      부분 묶음은 실제 미니배치 수로 나눔(C7), 업데이트 전에 θ_old의 teacher forcing 경로로 old log-prob·value를
      다시 계산(C10). 재계산은 업데이트와 같은 미니배치 분할·같은 묶음으로 해서 첫 미니배치의 ratio가 정확히 1이 되게 한다.
advantage 규칙 (3.6절 E2): "gae"(= A3⁰·C±), "gae_masked"(C⁺), "outcome"(O±), "outcome_masked"(O⁺).
    모든 규칙에서 value 손실과 반환값(returns)은 GAE로 같게 둔다 (critic·공유 경로를 네 칸에 같게).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

import core

T_EP = 80


@dataclass
class RoundBatch:
    """한 라운드의 롤아웃. 모든 배열의 앞 두 차원은 [T, N] (시점, 에피소드)."""
    actions: np.ndarray            # [T, N, 7] 행동 토큰
    logprob_gen: np.ndarray        # [T, N] 생성 때 저장한 joint log-prob
    value_gen: np.ndarray          # [T, N] 생성 때 value head 출력
    success: np.ndarray            # [T, N] bool, 스텝 t 행동 뒤 success
    grasp_latch: np.ndarray        # [T, N] bool
    consec_latch: np.ndarray       # [T, N] bool
    extra: dict = field(default_factory=dict)   # 관측 등 어댑터가 쓰는 자료

    @property
    def T(self):
        return self.actions.shape[0]

    @property
    def N(self):
        return self.actions.shape[1]

    def Y(self) -> np.ndarray:
        """Y = 마지막 스텝 (성공 ∧ 집기 래치) (C5 v0.9)."""
        return (self.success[-1] & self.grasp_latch[-1]).astype(np.int64)


def rewards_of(batch: RoundBatch, kind: str) -> np.ndarray:
    """[T, N, 1]. kind: "terminal" (A3⁰: r_T = Y) 또는 "potential" (A3ˢ: 원 잠재함수 변화량)."""
    if kind == "terminal":
        r = core.terminal_rewards(batch.Y(), batch.T)
    elif kind == "potential":
        r = core.potential_rewards(batch.grasp_latch, batch.consec_latch, batch.success)
    else:
        raise ValueError(kind)
    return r[..., None].astype(np.float32)


def advantages_and_returns(batch: RoundBatch, values: np.ndarray, reward_kind: str, rule: str):
    """values: [T, N] (V(s_t), t = 0..T−1). V(s_T)는 절단 마스크 0이라 쓰이지 않으므로 0으로 둔다.
    반환: (adv [T,N,1] 규칙 적용, returns [T,N,1] GAE 기반, adv_gae_normalized [T,N,1])."""
    T, N = batch.T, batch.N
    v = np.zeros((T + 1, N, 1), np.float32)
    v[:T, :, 0] = values
    masks = np.ones((T + 1, N, 1), np.float32)
    masks[T] = 0.0  # 80스텝 절단은 종결로 처리 (원 코드와 같음)
    returns, adv_gae = core.original_gae(rewards_of(batch, reward_kind), v, masks)
    if rule == "gae":
        adv = adv_gae
    elif rule == "gae_masked":
        adv = core.mask_negative(adv_gae)
    elif rule == "outcome":
        adv = core.outcome_advantage(batch.Y(), T)
    elif rule == "outcome_masked":
        adv = core.mask_negative(core.outcome_advantage(batch.Y(), T))
    else:
        raise ValueError(rule)
    return adv.astype(np.float32), returns.astype(np.float32), adv_gae.astype(np.float32)


def minibatch_order(n: int, seed: int) -> np.ndarray:
    """원 코드의 torch.randperm(batch_size)와 같은 방식, 다만 생성기를 시드로 고정해 E2 분기에서 같은 순서를 재현한다."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randperm(n, generator=g).numpy()


def plan_steps(order: np.ndarray, minibatch: int, accum: int, controlled: bool):
    """[(스텝 번호, [미니배치 인덱스 배열...], 나누는 값)]. 원 코드: 미니배치 8, 누적 20, 1 에폭."""
    mbs = [order[i:i + minibatch] for i in range(0, len(order), minibatch)]
    steps = []
    for s in range(math.ceil(len(mbs) / accum)):
        chunk = mbs[s * accum:(s + 1) * accum]
        steps.append((s, chunk, len(chunk) if controlled else accum))
    return steps


@torch.no_grad()
def recompute_old(adapter, steps, chunk: int):
    """C10: θ_old에서 teacher forcing으로 old log-prob·value를 다시 계산한다.
    업데이트 때와 같은 미니배치·같은 묶음 크기로 계산해 첫 미니배치의 ratio가 정확히 1이 되게 한다."""
    adapter.train_mode()  # 업데이트 forward와 같은 모드 (드롭아웃이 있더라도 경로를 맞춤)
    out_lp, out_v = {}, {}
    for _s, mbs, _d in steps:
        for mb in mbs:
            for j in range(0, len(mb), chunk):
                idx = mb[j:j + chunk]
                lp, v = adapter.evaluate(idx)
                for k, i in enumerate(idx):
                    out_lp[int(i)] = lp[k, 0].item()
                    out_v[int(i)] = v[k, 0].item()
    return out_lp, out_v


def c10_summary(lp_tf, lp_gen, v_tf, v_gen, adv, Y=None):
    """C10 잔여 불일치 감사: joint log w = log p₀ − log q₀ (teacher forcing − 생성 경로)의 분포.
    전체, advantage 부호별, (주어지면) 결과 Y별. ratio = 1은 q₀ = p₀를 뜻하지 않는다 (6차 검토 1번)."""
    lw = (lp_tf.astype(np.float64) - lp_gen.astype(np.float64))

    def s(x):
        if len(x) == 0:
            return None
        a = np.abs(x)
        return dict(n=int(len(x)), mean=float(x.mean()), mean_abs=float(a.mean()), p95_abs=float(np.quantile(a, 0.95)),
                    max_abs=float(a.max()), frac_outside_clip=float(np.mean(np.abs(np.exp(-x) - 1) > 0.2)))
    out = dict(all=s(lw), adv_pos=s(lw[adv > 0]), adv_nonpos=s(lw[adv <= 0]),
               value_abs_diff_mean=float(np.mean(np.abs(v_tf.astype(np.float64) - v_gen))))
    if Y is not None:
        out["Y1"], out["Y0"] = s(lw[Y > 0.5]), s(lw[Y <= 0.5])
    return out


@dataclass
class PPOConfig:
    mode: str = "controlled"          # "controlled" 또는 "original"
    reward_kind: str = "terminal"     # A3⁰ terminal, A3ˢ potential
    adv_rule: str = "gae"             # gae | gae_masked | outcome | outcome_masked
    minibatch: int = 8
    accum: int = 20
    chunk: int = 2                    # 메모리 때문에 미니배치를 나누는 크기 (수학적으로 같은 평균)
    clip: float = 0.2
    grad_clip: float = 10.0
    policy_coef: float = core.POLICY_COEF
    value_coef: float = core.VALUE_COEF
    entropy_coef: float = 0.0
    huber_delta: float = 10.0
    order_seed: int = 0
    recompute_old: bool = True        # C10 (controlled에서만 의미)
    max_steps: int | None = None      # 시험용: optimizer 스텝만 앞 몇 개 (old 값은 항상 전체 계산)


def ppo_update(adapter, batch: RoundBatch, cfg: PPOConfig, log=None):
    """한 라운드 업데이트 (1 에폭). 반환: 스텝별 기록과 첫 미니배치 ratio 통계."""
    T, N = batch.T, batch.N
    n = T * N
    controlled = cfg.mode == "controlled"
    order = minibatch_order(n, cfg.order_seed)
    steps_all = plan_steps(order, cfg.minibatch, cfg.accum, controlled)
    # max_steps는 optimizer 실행만 줄인다. old log-prob·value는 항상 라운드 전체에서 업데이트 전에 계산해 고정한다
    # (일부만 계산하면 GAE의 시간축 타깃이 달라진다. 6차 검토 2번).
    steps = steps_all if cfg.max_steps is None else steps_all[:cfg.max_steps]

    c10_audit = None
    if controlled and cfg.recompute_old:
        old_lp_d, old_v_d = recompute_old(adapter, steps_all, cfg.chunk)
        old_lp = np.array([old_lp_d[i] for i in range(n)], np.float32)
        old_v = np.array([old_v_d[i] for i in range(n)], np.float32)
        values_for_gae = old_v.reshape(T, N)
        c10_audit = dict(lp_gen=batch.logprob_gen.reshape(-1).astype(np.float32),
                         v_gen=batch.value_gen.reshape(-1).astype(np.float32))
    else:
        old_lp = batch.logprob_gen.reshape(-1).astype(np.float32)
        old_v = batch.value_gen.reshape(-1).astype(np.float32)
        values_for_gae = batch.value_gen
    adv, ret, _ = advantages_and_returns(batch, values_for_gae, cfg.reward_kind, cfg.adv_rule)
    adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
    if c10_audit is not None:  # 생성 경로 값은 학습에 쓰지 않고 감사용으로만 요약한다 (q₀ ≠ p₀ 측정)
        c10_audit = c10_summary(old_lp, c10_audit["lp_gen"], old_v, c10_audit["v_gen"], adv_f, np.tile(batch.Y(), T))

    params = list(adapter.params_policy) + list(adapter.params_value)
    adapter.train_mode()
    recs, first_ratio = [], None
    for s, mbs, divisor in steps:
        adapter.opt_policy.zero_grad(set_to_none=True)
        adapter.opt_value.zero_grad(set_to_none=True)
        tot = dict(loss=0.0, policy=0.0, value=0.0, clipfrac=0.0, n=0)
        for mb in mbs:
            m = len(mb)
            for j in range(0, m, cfg.chunk):
                idx = mb[j:j + cfg.chunk]
                c = len(idx)
                lp, v = adapter.evaluate(idx)
                dev = lp.device
                o = torch.as_tensor(old_lp[idx], device=dev).view(-1, 1)
                a = torch.as_tensor(adv_f[idx], device=dev).view(-1, 1)
                vp = torch.as_tensor(old_v[idx], device=dev).view(-1, 1)
                r = torch.as_tensor(ret_f[idx], device=dev).view(-1, 1)
                if controlled:
                    pl = cfg.policy_coef * core.policy_loss_controlled(lp, o, a, cfg.clip)
                else:
                    pl = core.policy_loss_original(lp, o.repeat(1, 7), a, cfg.clip)
                vl = core.value_loss_original(v, vp, r, cfg.clip, cfg.huber_delta)
                loss = pl + cfg.value_coef * vl if controlled else pl + vl
                # 미니배치 평균 = 묶음 평균의 가중합 (c/m), 스텝 누적은 divisor로 나눔 (원 코드: 항상 20)
                (loss * (c / m) / divisor).backward()
                ratio = torch.exp(lp.detach() - o)
                if first_ratio is None:
                    first_ratio = dict(max_abs_log_ratio=float((lp.detach() - o).abs().max()),
                                       ratio_equal_one=bool(torch.all(ratio == 1.0)))
                tot["loss"] += float(loss) * c
                tot["policy"] += float(pl) * c
                tot["value"] += float(vl) * c
                tot["clipfrac"] += float(((ratio - 1).abs() > cfg.clip).float().sum())
                tot["n"] += c
        gn = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
        finite = math.isfinite(gn)
        if finite:
            adapter.opt_value.step()
            adapter.opt_policy.step()
        rec = dict(step=s, n=tot["n"], divisor=divisor, loss=tot["loss"] / tot["n"], policy_loss=tot["policy"] / tot["n"],
                   value_loss=tot["value"] / tot["n"], clipfrac=tot["clipfrac"] / tot["n"], grad_norm=gn, finite=finite)
        recs.append(rec)
        if log:
            log(f"    ppo step {s + 1}/{len(steps)} n={rec['n']} loss {rec['loss']:.4f} pol {rec['policy_loss']:.4f} "
                f"val {rec['value_loss']:.4f} clipfrac {rec['clipfrac']:.3f} gnorm {gn:.3f}")
        if not finite:
            break
    return dict(steps=recs, n_steps=len(steps), n_steps_planned=len(steps_all), first_minibatch=first_ratio,
                c10_audit=c10_audit,
                adv_mean=float(adv_f.mean()), adv_pos_frac=float((adv_f > 0).mean()), Y_mean=float(batch.Y().mean()))


# ---------------------------------------------------------------- 상태 저장·복원 (E2 분기용, 메모리 안)

def snapshot(adapter) -> dict:
    """학습 가능한 파라미터 값과 두 optimizer 상태를 복사한다 (분기 지점의 체크포인트)."""
    import copy
    return dict(policy=[p.detach().clone() for p in adapter.params_policy],
                value=[p.detach().clone() for p in adapter.params_value],
                opt_policy=copy.deepcopy(adapter.opt_policy.state_dict()),
                opt_value=copy.deepcopy(adapter.opt_value.state_dict()))


def restore(adapter, snap: dict):
    import copy
    with torch.no_grad():
        for p, q in zip(adapter.params_policy, snap["policy"]):
            p.copy_(q)
        for p, q in zip(adapter.params_value, snap["value"]):
            p.copy_(q)
    adapter.opt_policy.load_state_dict(copy.deepcopy(snap["opt_policy"]))
    adapter.opt_value.load_state_dict(copy.deepcopy(snap["opt_value"]))


def param_diff(adapter, snap: dict) -> dict:
    """현재 파라미터와 스냅샷의 차이 (최대 절대값, 다른 원소 비율)."""
    mx, nd, tot = 0.0, 0, 0
    for p, q in zip(list(adapter.params_policy) + list(adapter.params_value), snap["policy"] + snap["value"]):
        d = (p.detach().float() - q.float()).abs()
        mx = max(mx, float(d.max()))
        nd += int((d > 0).sum())
        tot += d.numel()
    return dict(max_abs=mx, frac_different=nd / tot, bitwise_equal=(nd == 0))
