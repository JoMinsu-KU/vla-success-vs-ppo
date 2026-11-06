"""KL 장치의 스텝 되돌림 (G-PRE 초안 4절): 스텝 전 LoRA 파라미터와 optimizer 상태를 떠 두고, 넘으면 되돌린다."""
from __future__ import annotations

import copy


def snapshot(params, optimizer):
    return [p.detach().clone() for p in params], copy.deepcopy(optimizer.state_dict())


def restore(params, optimizer, snap):
    import torch
    saved, opt = snap
    with torch.no_grad():
        for p, s in zip(params, saved):
            p.copy_(s)
    optimizer.load_state_dict(copy.deepcopy(opt))
