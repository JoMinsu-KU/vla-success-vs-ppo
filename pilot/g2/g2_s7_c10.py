"""G2 S7: C10 잔여 생성–학습 불일치 측정 (명세 3.2절 C10 v0.12 ②, 판정 없이 기록).

새로 수집한 배치(환경 64 × 80스텝, 온도 1.0, K0)에서, 같은 관측·지시문·prefix에 대해
  q₀ = 생성 경로(원 predict_action_batch와 같은 generate 호출, 처리 전 logits의 256 slice) 위치별 분포
  p₀ = teacher forcing 경로(원 evaluate_action과 같은 forward) 위치별 분포
를 비교한다. teacher forcing은 두 구성으로 잰다:
  (L) 학습기 경로: train 모드, 업데이트 미니배치 순서(시드 0)로 8개씩 — 실제 C10 재계산과 같은 구성
  (B) 진단: 생성 때와 같은 묶음(같은 스텝의 환경 64개, 같은 padding) — 묶음 구성이 차이에 주는 몫
기록: 위치별 TV·KL(q₀‖p₀)·KL(p₀‖q₀), joint log w = log p₀(a) − log q₀(a) 분포(전체·advantage 부호별·Y별),
      보정 전후(w̄ = clip(p₀/q₀, 0.5, 2), joint 7토큰 단위, stopgrad) actor 기울기 코사인·노름 비,
      1회 업데이트 Δθ 차이와 탐침 정책 변화(KL 탐침), padding 수·배치 크기·LoRA 병합 여부·attention 경로.
사용: python g2_s7_c10.py <W1 ckpt> <out_dir> --label W1 [--lora <S9 ckpt .pt>] [--path flash_det|math] [--blind-Y]
  --blind-Y: Y별 요약을 blinded/ 파일로만 쓴다 (A3⁰ 학습 중 체크포인트에서 맹검 유지)
"""
import argparse
import functools
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_gpu as G  # noqa: E402
import numerics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("out_dir")
ap.add_argument("--label", required=True)
ap.add_argument("--lora", default="")
ap.add_argument("--path", default="flash_det", choices=["flash_det", "math", "default"])
ap.add_argument("--n-envs", type=int, default=64)
ap.add_argument("--blind-Y", action="store_true")
ap.add_argument("--tag", default="g2s7")
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
t_all = time.time()


def log(m):
    print(m, flush=True)


if a.path == "flash_det":
    from transformers.models.llama import modeling_llama as ml
    for name in ("flash_attn_func", "flash_attn_varlen_func"):
        setattr(ml, name, functools.partial(getattr(ml, name), deterministic=True))
policy = G.build_policy(a.ckpt, "bridge_orig", attn="sdpa" if a.path == "math" else "flash_attention_2", seed=0)
G.set_top_k(policy, "K0")
if a.lora:
    L.load_trainable(Path(a.lora), policy, load_optim=False)
res = dict(label=a.label, lora=a.lora, numerics=numerics.numerics_report(policy.vla), lora_merged=False)
unnorm = policy.vla.get_action_stats("bridge_orig")
env = G.GpuEnv("PutOnPlateInScene25Main-v3", unnorm, a.n_envs, seed=0)
N, T = a.n_envs, G.T_EP
dev = torch.device("cuda:0")

# ------------------------------------------------ 수집 (생성 경로 분포 q₀ 기록)
eids = [G.episode_id_for(f"{a.tag}-{a.label}", k) for k in range(N)]
rgb, instr = env.reset("train", eids, reconfigure=True, seed=7)
torch.manual_seed(7)
obs = np.zeros((T, N, 480, 640, 3), np.uint8)
acts = np.zeros((T, N, 7), np.int64)
lp_gen = np.zeros((T, N), np.float32)
v_gen = np.zeros((T, N), np.float32)
q0 = np.zeros((T, N, 7, 256), np.float32)
pad_gen = np.zeros((T, N), np.int64)
succ = np.zeros((T, N), bool)
grasp = np.zeros((T, N), bool)
consec = np.zeros((T, N), bool)
policy.prep_rollout()
for t in range(T):
    obs[t] = rgb.cpu().numpy()
    v, ids, lp, d, pad = L.gen_full(policy, dict(image=rgb, task_description=instr), temperature=1.0)
    acts[t], lp_gen[t], v_gen[t] = ids.cpu().numpy(), lp[:, 0].float().cpu().numpy(), v[:, 0].float().cpu().numpy()
    q0[t], pad_gen[t] = d.cpu().numpy(), pad.cpu().numpy()
    rgb, si = env.step(ids)
    succ[t], grasp[t], consec[t] = si["success"], si["grasp_latch"], si["consec_latch"]
env.close()
torch.cuda.empty_cache()
Y = succ[-1] & grasp[-1]
res["collect_min"] = round((time.time() - t_all) / 60, 1)
log(f"[collect] {res['collect_min']} min")

# gen_full의 joint log-prob가 저장 분포와 일치하는지 (자기 일관성)
chk = np.take_along_axis(q0, (acts - L.A0)[..., None], -1)[..., 0].sum(-1)
res["gen_selfcheck_max_abs"] = float(np.abs(chk - lp_gen).max())

# ------------------------------------------------ teacher forcing (L) 학습기 경로: train 모드, 미니배치 순서로 8개씩
n = T * N
order = ppo.minibatch_order(n, 0)
p0_L = np.zeros((n, 7, 256), np.float32)
lp_L = np.zeros(n, np.float32)
v_L = np.zeros(n, np.float32)
pad_L = np.zeros(n, np.int64)
flat_obs = obs.reshape(n, 480, 640, 3)
flat_act = acts.reshape(n, 7)
flat_ins = [instr[k % N] for k in range(n)]
policy.prep_training()
t0 = time.time()
with torch.no_grad():
    for j in range(0, n, 8):
        idx = order[j:j + 8]
        x = dict(image=torch.from_numpy(flat_obs[idx]).to(dev), task_description=[flat_ins[k] for k in idx])
        lp, v, d, pad = L.tf_full(policy, x, torch.from_numpy(flat_act[idx]).to(dev))
        lp_L[idx], v_L[idx], p0_L[idx], pad_L[idx] = lp[:, 0].float().cpu().numpy(), v[:, 0].float().cpu().numpy(), d.cpu().numpy(), pad.cpu().numpy()
res["tf_L_sec"] = round(time.time() - t0, 1)
# tf_full이 원 evaluate_actions와 같은 값인지
idx = order[:8]
x = dict(image=torch.from_numpy(flat_obs[idx]).to(dev), task_description=[flat_ins[k] for k in idx])
res["tf_full_vs_original_evaluate_actions"] = L.check_tf_matches_original(policy, x, torch.from_numpy(flat_act[idx]).to(dev))

# ------------------------------------------------ teacher forcing (B) 생성과 같은 묶음 (스텝별 64개), eval 모드
p0_B = np.zeros((T, N, 7, 256), np.float32)
lp_B = np.zeros((T, N), np.float32)
pad_B = np.zeros((T, N), np.int64)
policy.prep_rollout()
with torch.no_grad():
    for t in range(T):
        x = dict(image=torch.from_numpy(obs[t]).to(dev), task_description=list(instr))
        lp, _v, d, pad = L.tf_full(policy, x, torch.from_numpy(acts[t]).to(dev))
        lp_B[t], p0_B[t], pad_B[t] = lp[:, 0].float().cpu().numpy(), d.cpu().numpy(), pad.cpu().numpy()


def pos_stats(q, p):
    """q, p: [n,7,256] log-prob. 위치별 TV, KL(q‖p), KL(p‖q) 요약."""
    qe, pe = np.exp(q.astype(np.float64)), np.exp(p.astype(np.float64))
    tv = 0.5 * np.abs(qe - pe).sum(-1)
    klqp = (qe * (q - p)).sum(-1)
    klpq = (pe * (p - q)).sum(-1)
    s = lambda z: dict(mean=float(z.mean()), p95=float(np.quantile(z, 0.95)), max=float(z.max()))  # noqa: E731
    return dict(tv=[s(tv[:, j]) for j in range(7)], kl_q_p=[s(klqp[:, j]) for j in range(7)],
                kl_p_q=[s(klpq[:, j]) for j in range(7)], tv_all=s(tv), kl_q_p_sum7=s(klqp.sum(-1)))


q0f = q0.reshape(n, 7, 256)
res["L_trainer_path"] = pos_stats(q0f, p0_L)
res["B_same_batching"] = pos_stats(q0f, p0_B.reshape(n, 7, 256))
log(f"[pos L] tv_all {res['L_trainer_path']['tv_all']} | [pos B] tv_all {res['B_same_batching']['tv_all']}")

# advantage (terminal 보상, GAE, 재계산 value) 와 log w
batch = ppo.RoundBatch(actions=acts, logprob_gen=lp_gen, value_gen=v_gen, success=succ, grasp_latch=grasp, consec_latch=consec,
                       extra=dict(obs=obs, instr=instr))
adv, ret, _ = ppo.advantages_and_returns(batch, v_L.reshape(T, N), "terminal", "gae")
adv_f = adv.reshape(-1)
Yf = np.tile(Y, T).astype(float)
summ = ppo.c10_summary(lp_L, lp_gen.reshape(-1), v_L, v_gen.reshape(-1), adv_f, Yf)
yblock = {k: summ.pop(k) for k in ("Y1", "Y0")}
res["logw_L"] = summ
res["logw_B_all"] = ppo.c10_summary(lp_B.reshape(-1), lp_gen.reshape(-1), v_L, v_gen.reshape(-1), adv_f)["all"]
lw = lp_L - lp_gen.reshape(-1)
res["padding"] = dict(gen_pad_mean=float(pad_gen.mean()), gen_pad_max=int(pad_gen.max()), L_pad_mean=float(pad_L.mean()),
                      L_pad_max=int(pad_L.max()), B_pad_equals_gen=bool((pad_B == pad_gen).all()),
                      abs_logw_by_L_minus_gen_pad={str(k): float(np.abs(lw[(pad_L - pad_gen.reshape(-1)) == k]).mean())
                                                   for k in np.unique(pad_L - pad_gen.reshape(-1))},
                      batch_gen=N, batch_L=8)
if a.blind_Y:
    (out / "blinded" / f"s7_{a.label}_Y.json").write_text(json.dumps(yblock, default=float))
    res["logw_by_Y"] = "blinded (sha256 " + hashlib.sha256(json.dumps(yblock, default=float).encode()).hexdigest()[:16] + ")"
else:
    res["logw_by_Y"] = yblock
log(f"[logw L] {json.dumps(res['logw_L']['all'])}")

# ------------------------------------------------ 보정 전후 actor 기울기와 1회 업데이트
ad = oa.OpenVLAAdapter(policy, obs, instr, acts)
snap = ppo.snapshot(ad)
plan = ppo.plan_steps(order, 8, 20, controlled=True)
_s, mbs, div = plan[0]
wbar = np.clip(np.exp(lw), 0.5, 2.0).astype(np.float32)
res["wbar"] = dict(mean=float(wbar.mean()), frac_clipped_low=float((wbar == 0.5).mean()), frac_clipped_high=float((wbar == 2.0).mean()))
probe = L.Probe(obs, instr, acts, 64, seed=99)
policy.prep_training()
probe.set_ref(policy)
params_pol = list(ad.params_policy)
params_all = params_pol + list(ad.params_value)


def step_grad(weighted: bool, with_value: bool):
    ad.opt_policy.zero_grad(set_to_none=True)
    ad.opt_value.zero_grad(set_to_none=True)
    ad.train_mode()
    for mb in mbs:
        m = len(mb)
        for j in range(0, m, 8):
            idx = mb[j:j + 8]
            lp, v = ad.evaluate(idx)
            o = torch.as_tensor(lp_L[idx], device=dev).view(-1, 1)
            A = torch.as_tensor(adv_f[idx], device=dev).view(-1, 1)
            ratio = torch.exp(lp - o)
            surr = torch.min(ratio * A, torch.clamp(ratio, 0.8, 1.2) * A)
            w = torch.as_tensor(wbar[idx], device=dev).view(-1, 1) if weighted else torch.ones_like(surr)
            loss = 7.0 * (-(w * surr).mean())
            if with_value:
                vp = torch.as_tensor(v_L[idx], device=dev).view(-1, 1)
                r = torch.as_tensor(ret.reshape(-1)[idx], device=dev).view(-1, 1)
                loss = loss + core.value_loss_original(v, vp, r)
            (loss * (len(idx) / m) / div).backward()


def gv(ps):
    return torch.cat([(p.grad.double().reshape(-1) if p.grad is not None else torch.zeros(p.numel(), dtype=torch.float64, device=p.device)) for p in ps])


ppo.restore(ad, snap)
step_grad(False, False)
g0 = gv(params_pol)
step_grad(True, False)
g1 = gv(params_pol)
res["actor_grad_corrected_vs_not"] = dict(cos=float(torch.dot(g0, g1) / (g0.norm() * g1.norm())), norm_ratio=float(g1.norm() / g0.norm()))
del g0, g1
upd = {}
for key, w in (("uncorrected", False), ("corrected", True)):
    ppo.restore(ad, snap)
    step_grad(w, True)
    torch.nn.utils.clip_grad_norm_(params_all, 10.0)
    ad.opt_value.step()
    ad.opt_policy.step()
    upd[key] = (torch.cat([p.detach().double().reshape(-1) for p in params_pol]), probe.kl(policy))
p_s = torch.cat([q.double().reshape(-1).to(dev) for q in snap["policy"]])
d0, d1 = upd["uncorrected"][0] - p_s, upd["corrected"][0] - p_s
res["one_update_delta_theta"] = dict(cos=float(torch.dot(d0, d1) / (d0.norm() * d1.norm())),
                                     rel_diff=float((d1 - d0).norm() / d0.norm()),
                                     probe_kl_uncorrected=upd["uncorrected"][1], probe_kl_corrected=upd["corrected"][1])
log(f"[correction] grad {res['actor_grad_corrected_vs_not']} update {res['one_update_delta_theta']}")
res["total_min"] = round((time.time() - t_all) / 60, 1)
res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
L.write_json(out / f"g2_s7_{a.label}.json", res)
log("SAVED")
