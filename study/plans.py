"""A2⁺ 라운드 안 optimizer 스텝 계획 (명세 3.3절). GPU 없이 시험할 수 있게 train.py에서 분리."""
from __future__ import annotations

import numpy as np

import core


def a2p_step_plan(idx_all: np.ndarray, update: str, r: int, minibatch: int, accum: int, u_fix_steps: int) -> list:
    """반환: 스텝마다 전이 인덱스 배열.
    U-prop: 성공 전이를 라운드 시드(20,000 + r)의 무작위 순서로 한 번 훑는다 (스텝당 최대 minibatch·accum개, 마지막은 남은 수).
    U-fix : u_fix_steps 스텝, 스텝마다 minibatch·accum개를 성공 전이에서 복원 재표집."""
    rng = np.random.default_rng(20_000 + r)
    if update == "U-prop":
        order = idx_all[rng.permutation(len(idx_all))]
        plan, pos = [], 0
        for k in core.chunk_loss_divisors(len(order), minibatch, accum, controlled=True):
            n_tr = min(k * minibatch, len(order) - pos)
            plan.append(order[pos:pos + n_tr])
            pos += n_tr
        return plan
    if update == "U-fix":
        return [idx_all[j] for j in core.u_fix_indices(len(idx_all), u_fix_steps, minibatch, accum, rng)]
    raise ValueError(update)


def a2pm_step_plan(n_transitions: int, iters: int, r: int, minibatch: int, accum: int) -> list:
    """A2± 스텝 계획 (명세 3.3·3.5절): 반복 k(0부터)마다 전체 전이를 시드 20,000 + r + 100,000·k의 무작위 순서로
    한 번 훑는다 (스텝당 최대 minibatch·accum개, A2⁺ U-prop·PPO와 같은 분할). 5,120전이면 반복당 32스텝."""
    plan = []
    for k in range(iters):
        rng = np.random.default_rng(20_000 + r + 100_000 * k)
        order = rng.permutation(n_transitions)
        pos = 0
        for d in core.chunk_loss_divisors(n_transitions, minibatch, accum, controlled=True):
            n_tr = min(d * minibatch, n_transitions - pos)
            plan.append(order[pos:pos + n_tr])
            pos += n_tr
    return plan
