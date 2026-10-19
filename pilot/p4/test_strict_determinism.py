"""C11 엄격 결정성 시험 (6차 외부 검토 4·5번, G2 검증 모드의 4090 사전 확인).

1. 엄격 모드(use_deterministic_algorithms(True, warn_only=False)): 결정적 구현이 없는 연산이 있으면 오류로 멈춘다.
2. A/A 검사를 두 시작점에서: (i) 새 LoRA·optimizer (W1 스냅샷), (ii) 한 라운드 업데이트 뒤 Adam 모멘트가 쌓인 상태.
   각 시작점에서 같은 업데이트(C±)를 두 번 실행해 파라미터가 비트 단위로 같은지 본다.
3. 실제 attention 구현과 수치 설정을 기록한다.
사용: python test_strict_determinism.py <W1 ckpt> <out_json> [N=3]
"""
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "learner"))
sys.path.insert(0, str(ROOT / "eval"))

import numerics  # noqa: E402

numerics.enable_deterministic(strict=True)

import torch  # noqa: E402

import openvla_adapter as oa  # noqa: E402
import ppo  # noqa: E402


def vec(ps):
    return torch.cat([p.detach().double().reshape(-1).cpu() for p in ps])


def aa_check(ad, batch, start, cfg, label):
    out = []
    for rep in range(2):
        ppo.restore(ad, start)
        t0 = time.time()
        r = ppo.ppo_update(ad, batch, cfg)
        out.append((vec(ad.params_policy), vec(ad.params_value), time.time() - t0, r))
    p0 = torch.cat([q.double().reshape(-1).cpu() for q in start["policy"]])
    upd = float((out[0][0] - p0).norm())
    res = dict(lora_bitwise_equal=bool(torch.equal(out[0][0], out[1][0])),
               value_bitwise_equal=bool(torch.equal(out[0][1], out[1][1])),
               lora_max_abs_diff=float((out[0][0] - out[1][0]).abs().max()), lora_update_norm=upd,
               sec=[round(out[0][2], 1), round(out[1][2], 1)],
               first_minibatch=out[0][3]["first_minibatch"], c10_audit=out[0][3]["c10_audit"])
    print(f"[{label}] {json.dumps({k: v for k, v in res.items() if k != 'c10_audit'})}", flush=True)
    return res


def main():
    ckpt, out_json = sys.argv[1], Path(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    res = {"N": N}
    try:
        policy = oa.build_policy(ckpt, seed=0)
        res["numerics"] = numerics.numerics_report(policy.vla)
        print("numerics", json.dumps(res["numerics"]), flush=True)
        import batched_eval as be
        unnorm = policy.vla.get_action_stats("bridge_orig")
        specs = []
        for k in range(N):
            h = hashlib.sha256(f"p4-det-ep{k}".encode()).digest()
            specs.append(dict(env_id="PutOnPlateInScene25Main-v3", obj_set="train",
                              episode_id=int.from_bytes(h[:8], "little") & ((1 << 62) - 1)))
        batch = oa.collect_round(policy, lambda: be.EnvPool(min(3, N), unnorm, scale_seed=0), specs, topk="K0")
        torch.cuda.empty_cache()
        res["Y"] = batch.Y().tolist()
        ad = oa.OpenVLAAdapter(policy, batch.extra["obs"], batch.extra["instr"], batch.actions)
        cfg = ppo.PPOConfig(order_seed=7, chunk=2, recompute_old=True)
        snap0 = ppo.snapshot(ad)
        res["fresh"] = aa_check(ad, batch, snap0, cfg, "fresh W1 + new LoRA/Adam")
        # Adam 모멘트가 쌓인 상태: 한 라운드 업데이트 (다른 미니배치 순서) 뒤의 스냅샷
        ppo.restore(ad, snap0)
        ppo.ppo_update(ad, batch, ppo.PPOConfig(order_seed=11, chunk=2, recompute_old=True))
        snap1 = ppo.snapshot(ad)
        res["after_one_round"] = aa_check(ad, batch, snap1, cfg, "after one round (Adam moments)")
        res["strict_error"] = None
    except Exception as e:  # 엄격 모드에서 비결정 연산을 만나면 여기로 온다
        res["strict_error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-3000:]
        print("STRICT ERROR", res["strict_error"], flush=True)
    res["peak_cuda_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("SAVED", flush=True)


if __name__ == "__main__":
    main()
