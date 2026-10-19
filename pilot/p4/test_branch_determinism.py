"""P4 E2 분기 재현의 GPU 비결정성 확인.

같은 배치에서 계보 업데이트(C±)와 분기 체크포인트에서의 C± 재실행을 비교한다.
  A 기본 설정 (sdpa 메모리 효율 커널 등): 차이의 절대값과 업데이트 크기 대비 상대값 (LoRA·value head 따로)
  B 결정적 설정: math attention 커널만, torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG
     → 비트 단위로 같아지는지, 속도 차이
사용: python test_branch_determinism.py <W1 ckpt> <out_json> [N=3]
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "learner"))
sys.path.insert(0, str(ROOT / "eval"))

import hashlib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import e2  # noqa: E402
import openvla_adapter as oa  # noqa: E402
import ppo  # noqa: E402


def vec(ps):
    return torch.cat([p.detach().double().reshape(-1).cpu() for p in ps])


def compare(ad, snap0, lineage, branch_params_pol, branch_params_val):
    out = {}
    for name, idx_list, cur in (("lora", "policy", branch_params_pol), ("value_head", "value", branch_params_val)):
        p0 = torch.cat([q.double().reshape(-1).cpu() for q in snap0[idx_list]])
        pl = torch.cat([q.double().reshape(-1).cpu() for q in lineage[idx_list]])
        pb = cur
        upd = float((pl - p0).norm())
        d = (pb - pl)
        out[name] = dict(max_abs_diff=float(d.abs().max()), rel_diff_to_update=float(d.norm()) / max(upd, 1e-30),
                         frac_elements_different=float((d != 0).double().mean()), update_norm=upd,
                         bitwise_equal=bool((d == 0).all()))
    return out


def run(ad, batch, snap0, label):
    cfg = ppo.PPOConfig(order_seed=7, chunk=2, recompute_old=True)
    ppo.restore(ad, snap0)
    t0 = time.time()
    ppo.ppo_update(ad, batch, cfg)
    t_lin = time.time() - t0
    lineage = ppo.snapshot(ad)
    t0 = time.time()
    e2.branch_first_update(ad, batch, snap0, "C±", cfg)
    t_br = time.time() - t0
    r = compare(ad, snap0, lineage, vec(ad.params_policy), vec(ad.params_value))
    r["sec_lineage"], r["sec_branch"] = round(t_lin, 1), round(t_br, 1)
    print(f"[{label}] {json.dumps(r)}", flush=True)
    return r


def main():
    ckpt, out_json = sys.argv[1], Path(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    policy = oa.build_policy(ckpt, seed=0)
    import batched_eval as be
    unnorm = policy.vla.get_action_stats("bridge_orig")
    specs = []
    for k in range(N):
        h = hashlib.sha256(f"p4-det-ep{k}".encode()).digest()
        specs.append(dict(env_id="PutOnPlateInScene25Main-v3", obj_set="train",
                          episode_id=int.from_bytes(h[:8], "little") & ((1 << 62) - 1)))
    batch = oa.collect_round(policy, lambda: be.EnvPool(min(3, N), unnorm, scale_seed=0), specs, topk="K0")
    torch.cuda.empty_cache()
    ad = oa.OpenVLAAdapter(policy, batch.extra["obs"], batch.extra["instr"], batch.actions)
    snap0 = ppo.snapshot(ad)
    res = {"N": N, "Y": batch.Y().tolist()}
    res["A_default"] = run(ad, batch, snap0, "A default")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    res["B_deterministic"] = run(ad, batch, snap0, "B deterministic")
    res["peak_cuda_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("SAVED", flush=True)


if __name__ == "__main__":
    main()
