"""본실험 학습 실행 하나 (튜닝 또는 E1). G2 S9 학습기(pilot/g2/g2_s9_train.py)를 본실험 규칙으로 넓힌 것.

갈래:
  A3  = A3⁰ 통제 PPO (C9 계수, C10 θ_old 재계산, terminal r_T = Y, GAE γ 0.99 λ 0.95, 1 에폭, 8 × 누적 20).
  A2p = A2⁺ 성공 궤적 토큰 NLL. U-prop(성공 전이를 한 번 훑음) 또는 U-fix(라운드당 32스텝, 성공 전이 복원 재표집).
        0-성공 라운드는 업데이트하지 않는다.
        KL 장치 (G-PRE 초안 4절): 스텝마다 고정 탐침 KL(π_round ‖ π_now)을 재고, kl_max를 넘으면 그 스텝을
        되돌리고(LoRA 파라미터 + Adam 상태) 라운드를 끝낸다. 첫 스텝이 되돌려진 라운드가 연속 10개면 발산으로 멈춘다.
공통: W1 + 새 LoRA(초기화 시드 = 학습 시드), 라운드마다 환경 64개 × 80스텝 수집 (온도 1.0, K0),
      라운드별 고정 episode_id (태그 = --episode-tag; E1은 같은 시드의 두 갈래가 같은 태그 → 같은 과제 표본).
수치 경로: --numerics math → C11 결정적 설정(sdpa math, 결정적 알고리즘, 시뮬레이터 호출 동안만 해제)
          --numerics default → 원 코드 경로(flash_attention_2, 비결정적). 튜닝·평가용.
맹검: 성공·손실·KL·스텝 수·라운드 시간은 <W>/blinded/<run_id>/에만. 공개는 진행 라운드 수와 상태, 끝난 뒤 GPU시간 합계.
재시작: <W>/runs/<run_id>/ckpt/latest.pt가 있으면 그 라운드 뒤부터 이어 간다 (optimizer 상태 포함).
      맹검 기록은 이어 가는 라운드 이후 줄을 지우고 다시 쓴다.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--arm", required=True, choices=["A3", "A2p", "A3s", "A2pm"])
ap.add_argument("--iters", type=int, default=1, help="A2±: 라운드당 전체 롤아웃 반복 수 (명세 3.5절 {1, 2})")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--rounds", type=int, required=True)
ap.add_argument("--numerics", required=True, choices=["math", "default"])
ap.add_argument("--episode-tag", required=True)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--vh-lr", type=float, default=3e-3)
ap.add_argument("--update", default="U-prop", choices=["U-prop", "U-fix"])
ap.add_argument("--kl-max", type=float, default=C.CFG["kl_max"])
ap.add_argument("--save-rounds", default="")
ap.add_argument("--latest-every", type=int, default=C.CFG["latest_every"])
ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--n-envs", type=int, default=C.CFG["n_envs"])
ap.add_argument("--smoke-steps", type=int, default=0, help="시험용: 라운드당 optimizer 스텝을 앞 몇 개로 제한")
ap.add_argument("--stop-after", type=int, default=0, help="시험용: 이 라운드 수까지만 하고 멈춤 (재시작 검사)")
ap.add_argument("--adv-rule", default="gae", choices=["gae", "gae_masked", "outcome", "outcome_masked"],
                help="A3 계열 advantage 규칙 (E2-b 변형 C±·C⁺·O±·O⁺, 조건부 A3⁰₊ = gae_masked). 기본 gae = A3⁰")
ap.add_argument("--init-ckpt", default="", help="E2-b: latest가 없을 때 이 체크포인트(LoRA + optimizer + 라운드 수)에서 출발")
ap.add_argument("--check-ckpt", default="", help="E2-b: 첫 라운드 뒤 파라미터·optimizer 상태를 이 체크포인트와 비트 단위 비교")
a = ap.parse_args()

W = C.work_dir()
C.ensure_dirs(W)
RUN = W / "runs" / a.run_id
BL = W / "blinded" / a.run_id
(RUN / "ckpt").mkdir(parents=True, exist_ok=True)
BL.mkdir(parents=True, exist_ok=True)
PUB = W / "public" / "runs" / f"{a.run_id}.json"

import numerics  # noqa: E402

if a.numerics == "math":
    numerics.enable_deterministic(strict=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import core  # noqa: E402
import g2_gpu as G  # noqa: E402
import g2_learn as L  # noqa: E402
import openvla_adapter as oa  # noqa: E402
import plans  # noqa: E402
import ppo  # noqa: E402
import rollback  # noqa: E402


def pub(status, rounds_done, **kw):
    C.write_json(PUB, dict(run_id=a.run_id, status=status, rounds_done=rounds_done, rounds=a.rounds, updated=C.now(), **kw))


def save_ckpt(path: Path, extra: dict):
    tmp = path.with_suffix(".tmp")
    L.save_trainable(tmp, policy, extra=extra)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 정책·재시작
attn = "sdpa" if a.numerics == "math" else "flash_attention_2"
policy = G.build_policy(C.w1_ckpt(), C.CFG["unnorm_key"], attn=attn, seed=a.seed, lr=a.lr, vh_lr=a.vh_lr)
G.set_top_k(policy, C.CFG["top_k"])
state = dict(rounds_done=0, gpu_s=0.0, kl_first_rollback_streak=0, zero_success_streak=0, entropy_warmup=None, init_round=0)
latest = RUN / "ckpt" / "latest.pt"
if latest.exists():
    ex = L.load_trainable(latest, policy, load_optim=True)
    state.update({k: ex[k] for k in state if k in ex})
elif a.init_ckpt:
    ex = L.load_trainable(Path(a.init_ckpt), policy, load_optim=True)
    state.update({k: ex[k] for k in state if k in ex})
    state["gpu_s"] = 0.0
    state["init_round"] = int(ex["rounds_done"])
start = int(state["rounds_done"])
for f in ("rounds.jsonl", "cost.jsonl"):  # 이어 가는 라운드 이후 기록 지우기
    rows = [r for r in C.read_jsonl(BL / f) if r["round"] <= start]
    (BL / f).write_text("".join(json.dumps(r, default=float) + "\n" for r in rows), encoding="utf-8")
C.write_json(BL / f"meta_start{start:03d}.json", dict(args=vars(a), start_round=start, numerics=numerics.numerics_report(policy.vla),
                                                       lora_params=int(sum(p.numel() for p in policy.params_vla))))
if start >= a.rounds:
    pub("done", start)
    sys.exit(0)
pub("running", start)
unnorm = policy.vla.get_action_stats(C.CFG["unnorm_key"])
env = G.GpuEnv(C.CFG["train_env"], unnorm, a.n_envs, seed=a.seed)
save_rounds = {int(x) for x in a.save_rounds.split(",") if x}


# ---------------------------------------------------------------- 업데이트
def entropy_of(logp: torch.Tensor) -> float:  # [n,7,256] log-prob → 위치·상태 평균 엔트로피
    return float(-(logp.exp() * logp).sum(-1).mean())


def a3_update(ad, batch, r, probe):
    kl1 = {}

    def step_log(m):
        if "kl1" not in kl1 and m.strip().startswith("ppo step 1/"):
            kl1["kl1"] = probe.kl(policy)
            ad.train_mode()
    cap, orig_aar = {}, ppo.advantages_and_returns

    def aar(b, values, rk, rule):  # critic 설명 분산 기록용 (E2 분기 지점 기록, 명세 3.6절)
        o = orig_aar(b, values, rk, rule)
        cap["v"], cap["ret"] = np.asarray(values, np.float64).reshape(-1), o[1].reshape(-1).astype(np.float64)
        return o
    ppo.advantages_and_returns = aar
    t0 = time.time()
    reward_kind = "potential" if a.arm == "A3s" else "terminal"  # A3ˢ = 원 잠재함수 보상 (명세 3.3절, C5)
    try:
        upd = ppo.ppo_update(ad, batch, ppo.PPOConfig(order_seed=1000 + r, chunk=a.chunk, recompute_old=True,
                                                      max_steps=a.smoke_steps or None, reward_kind=reward_kind,
                                                      adv_rule=a.adv_rule),
                             log=step_log)
    finally:
        ppo.advantages_and_returns = orig_aar
    torch.cuda.synchronize()
    c10 = upd["c10_audit"] or {}
    finite = all(s["finite"] for s in upd["steps"])
    return dict(opt_steps=upd["n_steps"], update_s=round(time.time() - t0, 1), all_finite=finite,
                kl_probe_after_step1=kl1.get("kl1"), kl_probe_round_end=probe.kl(policy),
                first_minibatch=upd["first_minibatch"], adv_pos_frac=upd["adv_pos_frac"],
                losses=[{k: s[k] for k in ("loss", "policy_loss", "value_loss", "clipfrac", "grad_norm")} for s in upd["steps"]],
                explained_variance=L.explained_variance(cap["v"], cap["ret"]) if cap else None,
                c10_all=c10.get("all"), c10_by_sign_and_Y={k: c10.get(k) for k in ("adv_pos", "adv_nonpos", "Y1", "Y0")},
                rolled_back_first_step=False), finite


def snapshot():
    return rollback.snapshot(policy.params_vla, policy.vla_optimizer)


def restore(snap):
    rollback.restore(policy.params_vla, policy.vla_optimizer, snap)


def plan_a2p_steps(idx_all: np.ndarray, r: int) -> list:
    return plans.a2p_step_plan(idx_all, a.update, r, C.CFG["minibatch"], C.CFG["accum"], C.CFG["u_fix_steps"])


def a2p_update(ad, batch, r, probe):
    Y = batch.Y()
    succ = np.flatnonzero(Y)
    if len(succ) == 0:
        return dict(opt_steps=0, update_s=0.0, zero_success_round=True, steps=[], rolled_back_first_step=False), True
    N, T = batch.N, batch.T
    idx_all = np.array([t * N + i for i in succ for t in range(T)])
    plan = plan_a2p_steps(idx_all, r)
    if a.smoke_steps:
        plan = plan[:a.smoke_steps]
    params = list(ad.params_policy)
    steps, finite_all, rolled_first = [], True, False
    t0 = time.time()
    for s, idx_step in enumerate(plan):
        n_tr = len(idx_step)
        snap = snapshot()
        ad.train_mode()
        ad.opt_policy.zero_grad(set_to_none=True)
        ad.opt_value.zero_grad(set_to_none=True)
        tot = 0.0
        for j in range(0, n_tr, a.chunk):
            lp, _v = ad.evaluate(idx_step[j:j + a.chunk])
            loss = (-lp / 7.0).sum() / n_tr
            loss.backward()
            tot += float(loss)
        gn = float(torch.nn.utils.clip_grad_norm_(params, C.CFG["grad_clip"]))
        finite = math.isfinite(gn) and math.isfinite(tot)
        if not finite:
            steps.append(dict(step=s, n=n_tr, loss=tot, grad_norm=gn, finite=False))
            finite_all = False
            break
        ad.opt_policy.step()
        kl = probe.kl(policy)
        rec = dict(step=s, n=n_tr, loss=tot, grad_norm=gn, finite=True, kl_probe=kl, rolled_back=False)
        if kl > a.kl_max:
            restore(snap)
            rec["rolled_back"] = True
            rolled_first = s == 0
            steps.append(rec)
            break
        steps.append(rec)
        del snap
    torch.cuda.synchronize()
    applied = sum(1 for x in steps if x.get("finite") and not x.get("rolled_back"))
    return dict(opt_steps=applied, update_s=round(time.time() - t0, 1), zero_success_round=False, steps=steps,
                rolled_back_first_step=rolled_first, kl_probe_round_end=probe.kl(policy)), finite_all


def a2pm_update(ad, batch, r, probe):
    """A2± (명세 3.3절): 라운드 롤아웃 전체, 궤적 가중치 w_i = (Y_i − b₋ᵢ)/Ȳ (core.a2pm_weights).
    손실 = Σ w·(−log π/7) / 스텝 전이 수 (A2⁺의 성공 전이 평균 NLL과 같은 스케일, 1.4절).
    스텝 계획: 반복(--iters)마다 전체 전이를 시드 20,000 + r + 100,000·반복 순서로 8 × 누적 20씩 훑는다 (반복 1이면 32스텝).
    KL 장치·되돌림·발산 규칙은 A2⁺와 같다. Ȳ ∈ {0, 1}이면 업데이트하지 않는다."""
    Y = batch.Y()
    w_traj = core.a2pm_weights(Y)
    if not np.any(w_traj != 0):
        return dict(opt_steps=0, update_s=0.0, zero_weight_round=True, steps=[], rolled_back_first_step=False), True
    N, T = batch.N, batch.T
    w_tr = np.tile(w_traj, T)  # 전이 인덱스 t·N + i의 가중치 = w_i
    plan = plans.a2pm_step_plan(T * N, a.iters, r, C.CFG["minibatch"], C.CFG["accum"])
    if a.smoke_steps:
        plan = plan[:a.smoke_steps]
    params = list(ad.params_policy)
    steps, finite_all, rolled_first = [], True, False
    t0 = time.time()
    for s, idx_step in enumerate(plan):
        n_tr = len(idx_step)
        snap = snapshot()
        ad.train_mode()
        ad.opt_policy.zero_grad(set_to_none=True)
        ad.opt_value.zero_grad(set_to_none=True)
        tot = 0.0
        for j in range(0, n_tr, a.chunk):
            idx = idx_step[j:j + a.chunk]
            lp, _v = ad.evaluate(idx)
            wt = torch.as_tensor(w_tr[idx], dtype=lp.dtype, device=lp.device).reshape(-1, *([1] * (lp.dim() - 1)))
            loss = (-wt * lp / 7.0).sum() / n_tr
            loss.backward()
            tot += float(loss)
        gn = float(torch.nn.utils.clip_grad_norm_(params, C.CFG["grad_clip"]))
        finite = math.isfinite(gn) and math.isfinite(tot)
        if not finite:
            steps.append(dict(step=s, n=n_tr, loss=tot, grad_norm=gn, finite=False))
            finite_all = False
            break
        ad.opt_policy.step()
        kl = probe.kl(policy)
        rec = dict(step=s, n=n_tr, loss=tot, grad_norm=gn, finite=True, kl_probe=kl, rolled_back=False)
        if kl > a.kl_max:
            restore(snap)
            rec["rolled_back"] = True
            rolled_first = s == 0
            steps.append(rec)
            break
        steps.append(rec)
        del snap
    torch.cuda.synchronize()
    applied = sum(1 for x in steps if x.get("finite") and not x.get("rolled_back"))
    return dict(opt_steps=applied, update_s=round(time.time() - t0, 1), zero_weight_round=False, steps=steps,
                rolled_back_first_step=rolled_first, kl_probe_round_end=probe.kl(policy),
                weights=dict(pos=float(w_traj[w_traj > 0][0]) if np.any(w_traj > 0) else None,
                             neg=float(w_traj[w_traj < 0][0]) if np.any(w_traj < 0) else None)), finite_all


# ---------------------------------------------------------------- 라운드
status = "running"
for r in range(start, a.rounds):
    torch.cuda.reset_peak_memory_stats()
    t_round = time.time()
    eids = [G.episode_id_for(f"{a.episode_tag}-round{r}", k) for k in range(a.n_envs)]
    tm = {}
    rec = G.rollout(policy, env, C.CFG["train_obj_set"], eids, deterministic=False, rng_seed=100_000 * a.seed + r,
                    record_obs=True, timing=tm)
    batch = G.to_roundbatch(rec)
    ad = oa.OpenVLAAdapter(policy, rec["obs"], rec["instr"], rec["actions"])
    probe = L.Probe(batch.extra["obs"], batch.extra["instr"], batch.actions, C.CFG["kl_probe_states"], seed=10_000 + r)
    ad.train_mode()
    probe.set_ref(policy)
    ent = entropy_of(probe.ref)
    if state["entropy_warmup"] is None:
        state["entropy_warmup"] = ent
    if a.arm in ("A3", "A3s"):
        upd, finite = a3_update(ad, batch, r, probe)
    elif a.arm == "A2pm":
        upd, finite = a2pm_update(ad, batch, r, probe)
    else:
        upd, finite = a2p_update(ad, batch, r, probe)
    dt = time.time() - t_round
    state["gpu_s"] += dt
    state["rounds_done"] = r + 1
    nY = int(rec["Y"].sum())
    state["zero_success_streak"] = state["zero_success_streak"] + 1 if nY == 0 else 0
    state["kl_first_rollback_streak"] = state["kl_first_rollback_streak"] + 1 if upd.get("rolled_back_first_step") else 0
    div_flag = core.is_diverged(not finite, state["zero_success_streak"], ent, state["entropy_warmup"])
    C.append_jsonl(BL / "rounds.jsonl", dict(round=r + 1, n=int(len(rec["Y"])), Y=nY, Y_per_episode=rec["Y"].astype(int).tolist(),
                                             entropy_probe=ent, entropy_warmup=state["entropy_warmup"],
                                             zero_success_streak=state["zero_success_streak"],
                                             kl_first_rollback_streak=state["kl_first_rollback_streak"],
                                             divergence_rule_met=bool(div_flag), carrot_ids=rec["selected"].get("carrot_ids", np.array([])).tolist(),
                                             **upd))
    C.append_jsonl(BL / "cost.jsonl", dict(round=r + 1, round_s=round(dt, 1), collect_s=round(tm["total_s"], 1),
                                           update_s=upd.get("update_s"), opt_steps=upd.get("opt_steps"), peak=G.gpu_mem()))
    if not finite:
        status = "diverged_nonfinite"
    elif state["kl_first_rollback_streak"] >= C.CFG["kl_divergence_rounds"]:
        status = "diverged_kl"
    if (r + 1) in save_rounds:
        save_ckpt(RUN / "ckpt" / f"r{r + 1:03d}.pt", dict(state))
    if (r + 1) % a.latest_every == 0 or (r + 1) == a.rounds or status != "running":
        save_ckpt(latest, dict(state))
    if a.check_ckpt and r == state["init_round"]:  # E2-b 분기 검사: C±의 첫 라운드 = 계보의 다음 체크포인트
        import torch as _t
        ref = _t.load(a.check_ckpt, map_location="cpu")
        named = {n: p_ for n, p_ in policy.vla.named_parameters() if p_.requires_grad}
        same_p = all(_t.equal(named[n].detach().cpu(), t_) for n, t_ in ref["params"].items())
        so, ro = policy.vla_optimizer.state_dict()["state"], ref["opt_vla"]["state"]
        same_o = len(so) == len(ro) and all(_t.equal(so[k][f].cpu(), ro[k][f]) for k in ro for f in ro[k] if _t.is_tensor(ro[k][f]))
        C.write_json(W / "public" / "branch_checks" / f"{a.run_id}.json",
                     dict(run_id=a.run_id, adv_rule=a.adv_rule, round=r + 1, params_bitwise_equal=bool(same_p),
                          adam_state_bitwise_equal=bool(same_o), checked=C.now()))
    pub(status, r + 1)
    del ad, batch, rec, probe
    torch.cuda.empty_cache()
    if status != "running" or (a.stop_after and r + 1 >= a.stop_after):
        break

env.close()
if status == "running" and state["rounds_done"] >= a.rounds:
    status = "done"
if status != "running":
    save_ckpt(RUN / "ckpt" / "final.pt", dict(state))
    for f in ("rounds.jsonl", "cost.jsonl"):
        (BL / f"sha256_{f}.txt").write_text(C.sha256_file(BL / f) + "\n")
    pub(status, state["rounds_done"], gpu_h_total=round(state["gpu_s"] / 3600, 2))
else:
    pub("paused", state["rounds_done"])
