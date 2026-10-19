"""P4 시험 — 실제 OpenVLA(W1 + 새 LoRA + value head)로 통제 PPO 학습기와 E2 분기 확인 (RTX 4090).

순서: 환경 풀로 W1 정책의 에피소드 N개를 먼저 모은 뒤(그다음 환경은 다시 만들지 않음) GPU 시험을 한다.
  G1 C10 크기: 생성 경로 log-prob·value 대 teacher forcing 재계산 (이 배치 전체)
  G2 C9 실제 모델: 같은 미니배치(8 전이)에서 원 코드형 손실(7칸 복제) 대 통제형(계수 7)의 손실·기울기
  G3 C10 반영: 통제형 업데이트에서 θ_old 재계산 뒤 첫 미니배치 ratio
  G4 E2 분기: 계보 업데이트(C±) 대 분기 체크포인트에서 같은 순서로 다시 한 C± (차이 크기, GPU 비결정성 포함)
  G5 E2-a 세 층위 진단 (O·C 행)
  G6 짧은 학습 정상 동작: 계보 업데이트 전후의 value 손실·정책 손실, NaN 없음
사용: python test_ppo_gpu.py <W1 ckpt> <out_json> [N 에피소드=6] [환경 수=3]
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "learner"))
sys.path.insert(0, str(ROOT / "eval"))
import hashlib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import core  # noqa: E402
import e2  # noqa: E402
import openvla_adapter as oa  # noqa: E402
import ppo  # noqa: E402


def log(m):
    print(m, flush=True)


ckpt, out_json = sys.argv[1], Path(sys.argv[2])
N = int(sys.argv[3]) if len(sys.argv) > 3 else 6
n_env = int(sys.argv[4]) if len(sys.argv) > 4 else 3
res = {"N_episodes": N}
t_start = time.time()

policy = oa.build_policy(ckpt, seed=0)
import batched_eval as be  # noqa: E402

unnorm = policy.vla.get_action_stats("bridge_orig")
specs = []
for k in range(N):
    h = hashlib.sha256(f"p4-ppo-ep{k}".encode()).digest()
    specs.append(dict(env_id="PutOnPlateInScene25Main-v3", obj_set="train",
                      episode_id=int.from_bytes(h[:8], "little") & ((1 << 62) - 1)))
batch = oa.collect_round(policy, lambda: be.EnvPool(n_env, unnorm, scale_seed=0), specs, temperature=1.0, topk="K0", log=log)
torch.cuda.empty_cache()
ad = oa.OpenVLAAdapter(policy, batch.extra["obs"], batch.extra["instr"], batch.actions)
n = batch.T * batch.N
res["Y"] = batch.Y().tolist()

# ---------------------------------------------------------------- G1 C10 크기
t0 = time.time()
lp_tf, v_tf = e2._recompute_all(ad, n, 2)
d_lp = np.abs(lp_tf - batch.logprob_gen.reshape(-1))
d_v = np.abs(v_tf - batch.value_gen.reshape(-1))
res["G1_C10"] = dict(n=n, logprob_abs_diff_max=float(d_lp.max()), logprob_abs_diff_mean=float(d_lp.mean()),
                     logprob_abs_diff_p95=float(np.percentile(d_lp, 95)), ratio_outside_clip_frac=float(
                         (np.abs(np.exp(lp_tf - batch.logprob_gen.reshape(-1)) - 1) > 0.2).mean()),
                     value_abs_diff_max=float(d_v.max()), value_abs_diff_mean=float(d_v.mean()),
                     sec=round(time.time() - t0, 1))
log(f"[G1 C10] {res['G1_C10']}")

# ---------------------------------------------------------------- G2 C9 실제 모델
cfg = ppo.PPOConfig(order_seed=0, chunk=2)
idx8 = ppo.minibatch_order(n, 0)[:8]
adv, ret, _ = ppo.advantages_and_returns(batch, batch.value_gen, "terminal", "gae")
adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
params = list(ad.params_policy) + list(ad.params_value)
snap0 = ppo.snapshot(ad)
grads, losses = {}, {}
for mode in ("original", "controlled"):
    for p in params:
        p.grad = None
    ad.train_mode()
    tot = 0.0
    for j in range(0, 8, 2):
        idx = idx8[j:j + 2]
        lp, v = ad.evaluate(idx)
        o = torch.as_tensor(batch.logprob_gen.reshape(-1)[idx], device=lp.device).view(-1, 1)
        a = torch.as_tensor(adv_f[idx], device=lp.device).view(-1, 1)
        vp = torch.as_tensor(batch.value_gen.reshape(-1)[idx], device=lp.device).view(-1, 1)
        r = torch.as_tensor(ret_f[idx], device=lp.device).view(-1, 1)
        if mode == "original":
            loss = core.policy_loss_original(lp, o.repeat(1, 7), a) + core.value_loss_original(v, vp, r)
        else:
            loss = 7 * core.policy_loss_controlled(lp, o, a) + core.value_loss_original(v, vp, r)
        (loss * 2 / 8).backward()
        tot += float(loss) * 2 / 8
    grads[mode] = torch.cat([p.grad.detach().float().reshape(-1).cpu() for p in params if p.grad is not None])
    losses[mode] = tot
ga, gb = grads["original"], grads["controlled"]
res["G2_C9"] = dict(loss_original=losses["original"], loss_controlled=losses["controlled"],
                    loss_rel_diff=abs(losses["original"] - losses["controlled"]) / max(1e-12, abs(losses["original"])),
                    grad_cos=float(torch.dot(ga, gb) / (ga.norm() * gb.norm())),
                    grad_norm_ratio=float(gb.norm() / ga.norm()), grad_max_abs_diff=float((ga - gb).abs().max()))
for p in params:
    p.grad = None
del grads, ga, gb
log(f"[G2 C9] {res['G2_C9']}")

# ---------------------------------------------------------------- G3·G4·G6 계보 업데이트, C10 ratio, E2 분기 재현
ppo.restore(ad, snap0)
cfg = ppo.PPOConfig(order_seed=7, chunk=2, recompute_old=True)
t0 = time.time()
lin = ppo.ppo_update(ad, batch, cfg, log=log)
res["G3_C10_first_minibatch"] = lin["first_minibatch"]
res["G6_lineage_update"] = dict(n_steps=lin["n_steps"], steps=lin["steps"], sec=round(time.time() - t0, 1),
                                all_finite=all(s["finite"] for s in lin["steps"]))
lineage_next = ppo.snapshot(ad)
# 업데이트 뒤 같은 배치에서 손실 (value 손실이 줄었는지)
lp_after, v_after = e2._recompute_all(ad, n, 2)
ret_t = torch.as_tensor(ppo.advantages_and_returns(batch, lp_tf.reshape(batch.T, batch.N) * 0 + v_tf.reshape(batch.T, batch.N),
                                                   "terminal", "gae")[1].reshape(-1))
vl_before = float(core.value_loss_original(torch.as_tensor(v_tf).view(-1, 1), torch.as_tensor(v_tf).view(-1, 1), ret_t.view(-1, 1)))
vl_after = float(core.value_loss_original(torch.as_tensor(v_after).view(-1, 1), torch.as_tensor(v_tf).view(-1, 1), ret_t.view(-1, 1)))
res["G6_lineage_update"].update(value_loss_before=vl_before, value_loss_after=vl_after,
                                mean_logprob_change_success=float((lp_after - lp_tf)[np.repeat(batch.Y()[None, :], batch.T, 0).reshape(-1) == 1].mean()) if batch.Y().any() else None,
                                mean_logprob_change_failure=float((lp_after - lp_tf)[np.repeat(batch.Y()[None, :], batch.T, 0).reshape(-1) == 0].mean()) if (1 - batch.Y()).any() else None)
log(f"[G3 C10 ratio] {res['G3_C10_first_minibatch']}")
log(f"[G6] value loss {vl_before:.4f} -> {vl_after:.4f}")

t0 = time.time()
e2.branch_first_update(ad, batch, snap0, "C±", cfg)
res["G4_branch_reproduction"] = dict(**ppo.param_diff(ad, lineage_next), sec=round(time.time() - t0, 1))
log(f"[G4 E2 branch C± vs lineage] {res['G4_branch_reproduction']}")

# ---------------------------------------------------------------- G5 E2-a 진단
t0 = time.time()
diag = e2.e2a_diagnostics(ad, batch, snap0, ppo.PPOConfig(chunk=2))
res["G5_E2a"] = dict(diag, sec=round(time.time() - t0, 1))
log(f"[G5 E2-a] {json.dumps(res['G5_E2a'], ensure_ascii=False)}")

res["total_min"] = round((time.time() - t_start) / 60, 1)
res["peak_cuda_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
log("SAVED")
