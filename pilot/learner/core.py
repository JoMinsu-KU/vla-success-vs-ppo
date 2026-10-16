"""통제 연구 공통 학습기의 핵심 계산 (연구 명세 v0.9, C5·C7·C8·C9, 3.3·3.6절).

모델과 무관한 순수 계산만 둔다 (넘파이·토치). 원 코드(RL4VLA)의 계산을 그대로 옮긴 함수(original_*)와
통제 연구용 함수를 나란히 두고, P4 단위 시험이 둘의 관계를 확인한다.
"""
from __future__ import annotations

import math

import numpy as np
import torch

# ---------------------------------------------------------------- 보상·라벨 (C5)


def terminal_rewards(Y: np.ndarray, T: int = 80) -> np.ndarray:
    """A3⁰ terminal 보상: r_t = Y·1[t = T]. 반환 [T, B] float32."""
    r = np.zeros((T, len(Y)), dtype=np.float32)
    r[-1] = Y.astype(np.float32)
    return r


def potential_rewards(grasp_latch: np.ndarray, consec_latch: np.ndarray, success: np.ndarray) -> np.ndarray:
    """원 보상(A3ˢ): Φ_t = 0.1·grasp_latch + 0.1·consecutive_latch + 1.0·(success ∧ grasp_latch), r_t = Φ_t − Φ_{t−1}.
    입력 [T, B] bool (스텝 t의 행동 뒤 info). 원 코드 simpler_wrapper.get_reward와 같다 (Φ_0 = 0)."""
    phi = 0.1 * grasp_latch + 0.1 * consec_latch + 1.0 * (success & grasp_latch)
    phi = phi.astype(np.float32)
    prev = np.vstack([np.zeros((1, phi.shape[1]), np.float32), phi[:-1]])
    return phi - prev


def discounted_return(r: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """Σ_t γ^{t−1} r_t (t = 1..T), 열마다. 입력 [T, B]."""
    T = r.shape[0]
    w = gamma ** np.arange(T, dtype=np.float64)
    return (w[:, None] * r).sum(0)


# ---------------------------------------------------------------- GAE와 정규화 (원 replay_buffer.py 그대로)


def original_gae(rewards: np.ndarray, values: np.ndarray, masks: np.ndarray, gamma=0.99, lam=0.95):
    """원 compute_returns_ppo. rewards [T,B,1], values [T+1,B,1], masks [T+1,B,1] (masks[T] = 0: 80스텝 절단 종결).
    반환 (returns [T,B,1], 정규화된 advantages [T,B,1])."""
    T = rewards.shape[0]
    returns = np.zeros_like(rewards)
    gae = 0
    for step in reversed(range(T)):
        delta = rewards[step] + gamma * values[step + 1] * masks[step + 1] - values[step]
        gae = delta + gamma * lam * masks[step + 1] * gae
        returns[step] = gae + values[step]
    adv = returns - values[:-1]
    return returns, normalize_adv(adv)


def normalize_adv(adv: np.ndarray) -> np.ndarray:
    """원 코드와 같은 버퍼 전체 정규화: (a − mean) / (std(모집단) + 1e-5)."""
    return (adv - adv.mean()) / (adv.std() + 1e-5)


# ---------------------------------------------------------------- E2 advantage (3.6절)


def outcome_advantage(Y: np.ndarray, T: int = 80, eps: float = 1e-8) -> np.ndarray:
    """Ã^O_{it} = (Y_i − Ȳ) / (√(Ȳ(1−Ȳ)) + ε), 궤적의 모든 시점에 같은 값. 반환 [T, B, 1]."""
    Yf = Y.astype(np.float64)
    m = Yf.mean()
    a = (Yf - m) / (math.sqrt(m * (1 - m)) + eps)
    return np.broadcast_to(a[None, :, None], (T, len(Y), 1)).astype(np.float32).copy()


def mask_negative(adv_normalized: np.ndarray) -> np.ndarray:
    """공통 정규화 뒤 음수를 0으로 (재정규화하지 않음)."""
    return np.maximum(adv_normalized, 0.0)


# ---------------------------------------------------------------- 우도 갈래 가중치 (3.3절)


def a2pm_weights(Y: np.ndarray) -> np.ndarray:
    """A2± 궤적 가중치 (Y_i − b_{−i}) / Ȳ, b_{−i} = 자신을 뺀 평균. Ȳ ∈ {0, 1}이면 모두 0 (업데이트 생략)."""
    Yf = Y.astype(np.float64)
    N = len(Yf)
    m = Yf.mean()
    if m in (0.0, 1.0):
        return np.zeros(N)
    b_loo = (Yf.sum() - Yf) / (N - 1)
    return (Yf - b_loo) / m


def a2p_weights(Y: np.ndarray) -> np.ndarray:
    """A2⁺를 0-1 REINFORCE로 본 가중치 Y_i / Ȳ (1.4절 (4)). Ȳ = 0이면 모두 0."""
    Yf = Y.astype(np.float64)
    m = Yf.mean()
    return np.zeros(len(Yf)) if m == 0 else Yf / m


# ---------------------------------------------------------------- PPO 손실 (원 openvla_train.py, C9)


def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def value_loss_original(values, value_preds, returns, clip=0.2, huber_delta=10.0):
    """원 코드의 value 손실: 클리핑한 예측과 원 예측의 Huber 손실 중 큰 것의 평균."""
    vclip = value_preds + (values - value_preds).clamp(-clip, clip)
    return torch.max(huber_loss(returns - values, huber_delta), huber_loss(returns - vclip, huber_delta)).mean()


def policy_loss_original(logprob, old_logprob_dup7, adv, clip=0.2):
    """원 코드: old log-prob가 [B,7]로 복제돼 있고 ratio가 [B,7]로 퍼진 뒤 마지막 차원으로 합산 → 수식의 7배."""
    ratio = torch.exp(logprob - old_logprob_dup7)  # [B,1] − [B,7] → [B,7]
    s1 = ratio * adv
    s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    return -torch.min(s1, s2).sum(dim=-1, keepdim=True).mean()


def policy_loss_controlled(logprob, old_logprob, adv, clip=0.2):
    """통제 학습기: joint log-prob 하나 [B,1]로 수식대로 한 번 계산 (계수는 밖에서 곱한다)."""
    ratio = torch.exp(logprob - old_logprob)
    s1 = ratio * adv
    s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    return -torch.min(s1, s2).mean()


POLICY_COEF, VALUE_COEF = 7.0, 1.0  # C9: L = 7·L_π + 1·L_V  (원 코드 총 손실과 같게)


# ---------------------------------------------------------------- optimizer 스텝 구성 (C7, 3.3절)


def steps_per_round(n_transitions: int, minibatch: int = 8, accum: int = 20) -> int:
    """라운드당 optimizer 스텝 수 ⌈N / (minibatch·accum)⌉ (1 에폭)."""
    return math.ceil(n_transitions / (minibatch * accum))


def chunk_loss_divisors(n_transitions: int, minibatch: int = 8, accum: int = 20, controlled: bool = True):
    """optimizer 스텝마다 미니배치 손실을 나누는 값. 원 코드는 항상 accum(20)으로 나눠 마지막 부분 묶음이 과소 스케일,
    통제 학습기는 실제 미니배치 수로 나눈다."""
    n_mb = math.ceil(n_transitions / minibatch)
    out = []
    for s in range(steps_per_round(n_transitions, minibatch, accum)):
        k = min(accum, n_mb - s * accum)
        out.append(k if controlled else accum)
    return out


def u_fix_indices(n_success_transitions: int, steps: int = 32, minibatch: int = 8, accum: int = 20,
                  rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """U-fix: 라운드당 `steps` 스텝, 스텝마다 minibatch·accum개를 성공 전이에서 복원 재표집."""
    rng = rng or np.random.default_rng()
    if n_success_transitions == 0:
        return []
    return [rng.integers(0, n_success_transitions, minibatch * accum) for _ in range(steps)]


# ---------------------------------------------------------------- 우도 갈래 안정화 (3.3절)


def kl_estimate(logp_round_start: torch.Tensor, logp_now: torch.Tensor) -> float:
    """라운드 배치의 행동에서 KL(π_round ‖ π_now)의 표본 추정 (k3 추정량: E[(r − 1) − log r], r = π_now/π_round)."""
    logr = (logp_now - logp_round_start).float()
    return float(((logr.exp() - 1) - logr).mean())


def should_stop_round(kl: float, kl_max: float) -> bool:
    return kl > kl_max


def is_diverged(nan_or_inf: bool, zero_success_streak: int, entropy: float, entropy_warmup: float) -> bool:
    """발산 판정 (3.3절): NaN·Inf, 또는 수집 성공률 연속 10라운드 0이면서 행동 토큰 엔트로피가 웜업의 2배 초과."""
    return nan_or_inf or (zero_success_streak >= 10 and entropy > 2 * entropy_warmup)
