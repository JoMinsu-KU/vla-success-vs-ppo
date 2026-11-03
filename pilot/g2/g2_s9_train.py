"""G2 S9: A3⁰·A2⁺ 단기 학습 (각 최대 50라운드) — 학습 신호와 비용 (명세 3.3절, 4.2절 맹검, G2 명세 3절 사전 고정).

공통: W1(공개 웜업, 병합) + 새 LoRA(원 PPO 레시피, 초기화 시드 = 학습 시드 0), 라운드마다 환경 64개 × 80스텝 수집
      (온도 1.0, K0, 학습 과제 분포 Main-v3 train, 라운드·갈래별 고정 episode_id, 배치마다 장면 재구성).
      C11 결정적 수치 설정(--path로 attention 경로 선택), 실제 경로 기록.
A3⁰: 통제 PPO (C9 계수 7·1, C10 θ_old 재계산, C7 부분 묶음 실제 수), terminal 보상 r_T = Y, GAE(γ 0.99, λ 0.95),
     1 에폭, 미니배치 8 × 누적 20 (32스텝), LoRA lr 1e-4, value head lr 3e-3, 클리핑 10.
A2⁺: U-prop — 성공(Y = 1) 궤적의 전이를 무작위 순서로 한 번 훑는다. 스텝마다 전이 160개(마지막은 실제 수로 나눔),
     손실 = 토큰 평균 NLL (−joint log-prob / 7), AdamW(LoRA, 원 PPO optimizer 설정 lr 1e-4), 클리핑 10.
     0-성공 라운드는 업데이트하지 않는다. KL 장치는 기록만 (상한은 G-PRE).
KL 탐침: 라운드 배치에서 성공·실패 무관 균등 64개 상태, 위치 7개 정확한 조건부 KL(π_round ‖ π_now) 합의 상태 평균.

맹검 (명세 4.2절): 화면·cost 파일에는 비용 지표만 쓴다 (라운드 시간, GPU 시간, 피크 메모리, optimizer 스텝 수, A3⁰의 KL·C10 전체 요약).
  성공률·성공 수·손실·critic 설명 분산·A2⁺ KL은 blinded/<arm>_rounds.jsonl에만 쓰고 해시만 기록한다.
  A3⁰ 학습 신호는 사전 고정 스크립트 g2_signal.py가 "있음/없음"만 출력한다.
사용: python g2_s9_train.py <W1 ckpt> <out_dir> --arm A3 | A2p [--rounds 50] [--path flash_det|math]
"""
import argparse
import functools
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_gpu as G  # noqa: E402
import numerics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("out_dir")
ap.add_argument("--arm", required=True, choices=["A3", "A2p"])
ap.add_argument("--rounds", type=int, default=50)
ap.add_argument("--path", default="flash_det", choices=["flash_det", "math", "default"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--n-envs", type=int, default=64)
ap.add_argument("--save-rounds", default="10,50")
ap.add_argument("--resume", default="", help="ckpt_rNN.pt 경로 (그 라운드 뒤부터 이어서)")
ap.add_argument("--smoke-steps", type=int, default=0)
a = ap.parse_args()

if a.path != "default":
    numerics.enable_deterministic(strict=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import core  # noqa: E402
import g2_learn as L  # noqa: E402
import openvla_adapter as oa  # noqa: E402
import ppo  # noqa: E402

out = Path(a.out_dir)
(out / "blinded").mkdir(parents=True, exist_ok=True)
(out / "ckpt").mkdir(parents=True, exist_ok=True)
cost_f = out / f"cost_{a.arm}.jsonl"
blind_f = out / "blinded" / f"{a.arm}_rounds.jsonl"


def log(m):
    print(m, flush=True)
    with (out / f"log_{a.arm}.txt").open("a") as f:
        f.write(m + "\n")


def append(path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj, default=float) + "\n")


if a.path == "flash_det":
    from transformers.models.llama import modeling_llama as ml
    for name in ("flash_attn_func", "flash_attn_varlen_func"):
        setattr(ml, name, functools.partial(getattr(ml, name), deterministic=True))
attn = "sdpa" if a.path == "math" else "flash_attention_2"
policy = G.build_policy(a.ckpt, "bridge_orig", attn=attn, seed=a.seed)
G.set_top_k(policy, "K0")
start_round = 0
if a.resume:
    ex = L.load_trainable(Path(a.resume), policy, load_optim=True)
    start_round = int(ex["rounds_done"])
meta = dict(args=vars(a), numerics=numerics.numerics_report(policy.vla), start_round=start_round,
            lora_params=int(sum(p.numel() for p in policy.params_vla)), value_params=int(sum(p.numel() for p in policy.params_vh)))
L.write_json(out / f"meta_{a.arm}.json", meta)
log(f"[meta] {json.dumps(meta)}")
unnorm = policy.vla.get_action_stats("bridge_orig")
env = G.GpuEnv("PutOnPlateInScene25Main-v3", unnorm, a.n_envs, seed=a.seed)
save_rounds = {int(x) for x in a.save_rounds.split(",") if x}
gpu_s_total = 0.0


def a3_update(ad, batch, r):
    """통제 PPO 1 라운드. 반환 (cost dict, blinded dict)."""
    probe = L.Probe(batch.extra["obs"], batch.extra["instr"], batch.actions, 64, seed=10_000 + r)
    ad.train_mode()
    probe.set_ref(policy)
    cap, t_rc, kl1 = {}, {}, {}
    orig_aar, orig_rc = ppo.advantages_and_returns, ppo.recompute_old

    def aar(b, values, rk, rule):
        o = orig_aar(b, values, rk, rule)
        cap["v"], cap["ret"] = np.asarray(values, np.float64).reshape(-1), o[1].reshape(-1).astype(np.float64)
        return o

    def rc(*x, **k):
        t0 = time.time()
        o = orig_rc(*x, **k)
        torch.cuda.synchronize()
        t_rc["s"] = time.time() - t0
        return o

    def step_log(m):
        if "kl1" not in kl1 and m.strip().startswith("ppo step 1/"):
            kl1["kl1"] = probe.kl(policy)
            ad.train_mode()
    ppo.advantages_and_returns, ppo.recompute_old = aar, rc
    try:
        t0 = time.time()
        upd = ppo.ppo_update(ad, batch, ppo.PPOConfig(order_seed=1000 + r, chunk=a.chunk, recompute_old=True,
                                                      max_steps=a.smoke_steps or None), log=step_log)
        torch.cuda.synchronize()
        t_upd = time.time() - t0
    finally:
        ppo.advantages_and_returns, ppo.recompute_old = orig_aar, orig_rc
    kl_end = probe.kl(policy)
    c10 = upd["c10_audit"] or {}
    cost = dict(recompute_old_s=round(t_rc.get("s", 0), 1), update_s=round(t_upd - t_rc.get("s", 0), 1),
                opt_steps=upd["n_steps"], kl_probe_after_step1=kl1.get("kl1"), kl_probe_round_end=kl_end,
                first_minibatch=upd["first_minibatch"], all_finite=all(s["finite"] for s in upd["steps"]),
                c10_all=c10.get("all"), c10_value_abs_diff_mean=c10.get("value_abs_diff_mean"))
    blind = dict(explained_variance=L.explained_variance(cap["v"], cap["ret"]),
                 losses=[{k: s[k] for k in ("loss", "policy_loss", "value_loss", "clipfrac", "grad_norm")} for s in upd["steps"]],
                 adv_pos_frac=upd["adv_pos_frac"], c10_by_sign_and_Y={k: c10.get(k) for k in ("adv_pos", "adv_nonpos", "Y1", "Y0")})
    return cost, blind


def a2p_update(ad, batch, r):
    Y = batch.Y()
    succ = np.flatnonzero(Y)
    if len(succ) == 0:
        return dict(opt_steps=0, update_s=0.0, zero_success_round=True), dict(steps=[])
    probe = L.Probe(batch.extra["obs"], batch.extra["instr"], batch.actions, 64, seed=10_000 + r)
    ad.train_mode()
    probe.set_ref(policy)
    N, T = batch.N, batch.T
    idx_all = np.array([t * N + i for i in succ for t in range(T)])
    rng = np.random.default_rng(20_000 + r)
    order = idx_all[rng.permutation(len(idx_all))]
    divisors = core.chunk_loss_divisors(len(order), 8, 20, controlled=True)
    if a.smoke_steps:
        divisors = divisors[:a.smoke_steps]
    params = list(ad.params_policy)
    steps, pos = [], 0
    t0 = time.time()
    for s, n_mb in enumerate(divisors):
        n_tr = min(n_mb * 8, len(order) - pos)
        idx_step = order[pos:pos + n_tr]
        pos += n_tr
        ad.train_mode()
        ad.opt_policy.zero_grad(set_to_none=True)
        ad.opt_value.zero_grad(set_to_none=True)
        tot = 0.0
        for j in range(0, n_tr, a.chunk):
            idx = idx_step[j:j + a.chunk]
            lp, _v = ad.evaluate(idx)
            loss = (-lp / 7.0).sum() / n_tr
            loss.backward()
            tot += float(loss)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 10.0))
        finite = math.isfinite(gn) and math.isfinite(tot)
        if finite:
            ad.opt_policy.step()
        steps.append(dict(step=s, n=n_tr, loss=tot, grad_norm=gn, finite=finite, kl_probe=probe.kl(policy)))
        if not finite:
            break
    torch.cuda.synchronize()
    return dict(opt_steps=len(steps), update_s=round(time.time() - t0, 1), zero_success_round=False,
                all_finite=all(x["finite"] for x in steps)), dict(steps=steps)


for r in range(start_round, a.rounds):
    torch.cuda.reset_peak_memory_stats()
    t_round = time.time()
    eids = [G.episode_id_for(f"g2s9-{a.arm}-seed{a.seed}-round{r}", k) for k in range(a.n_envs)]
    tm = {}
    rec = G.rollout(policy, env, "train", eids, deterministic=False, rng_seed=100_000 * a.seed + r, record_obs=True, timing=tm)
    batch = G.to_roundbatch(rec)
    ad = oa.OpenVLAAdapter(policy, rec["obs"], rec["instr"], rec["actions"])
    if a.arm == "A3":
        cost, blind = a3_update(ad, batch, r)
    else:
        cost, blind = a2p_update(ad, batch, r)
    dt = time.time() - t_round
    gpu_s_total += dt
    cost.update(round=r + 1, round_s=round(dt, 1), collect_s=round(tm["total_s"], 1), collect_policy_s=round(tm["policy_s"], 1),
                collect_sim_render_s=round(tm["sim_s"], 1), peak=G.gpu_mem(), gpu_h_cumulative=round(gpu_s_total / 3600, 3))
    if r == 0:  # 라운드 0 = W1 수집 (학습 전). 명세 3.7절 전환 규칙 재적용용으로 공개 기록
        cost["round0_W1_collect"] = dict(n=int(len(rec["Y"])), Y=int(rec["Y"].sum()),
                                         objects=rec["selected"].get("carrot_ids", np.array([])).tolist(),
                                         overlays=rec["selected"].get("overlay_ids", np.array([])).tolist(),
                                         Y_per_episode=rec["Y"].astype(int).tolist())
    append(cost_f, cost)
    blind.update(round=r + 1, n=int(len(rec["Y"])), Y=int(rec["Y"].sum()), success_rate=float(rec["Y"].mean()),
                 success_last_orig=int(rec["success"][-1].sum()), Y_per_episode=rec["Y"].astype(int).tolist())
    append(blind_f, blind)
    msg = {k: cost[k] for k in ("round", "round_s", "collect_s", "opt_steps", "update_s", "gpu_h_cumulative") if k in cost}
    msg["peak_gib"] = cost["peak"]["peak_gib"]
    if a.arm == "A3":
        msg.update(recompute_s=cost["recompute_old_s"], kl1=cost["kl_probe_after_step1"], kl_end=cost["kl_probe_round_end"])
    log(f"[{a.arm} round {r + 1}] {json.dumps(msg, default=float)}")
    if (r + 1) in save_rounds:
        L.save_trainable(out / "ckpt" / f"{a.arm}_r{r + 1:02d}.pt", policy, extra=dict(rounds_done=r + 1))
    del ad, batch, rec
    torch.cuda.empty_cache()

env.close()
h = hashlib.sha256(blind_f.read_bytes()).hexdigest()
(out / f"blinded_sha256_{a.arm}.txt").write_text(h + "\n")
log(f"[done] {a.arm} rounds={a.rounds} gpu_h={gpu_s_total / 3600:.2f} blinded sha256={h}")
