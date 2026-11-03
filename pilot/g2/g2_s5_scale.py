"""G2 S5: 처리량과 전체 규모 학습 계산 (G1 조건 2, 명세 C7·C9·C10, 7.4절).

W1(공개 웜업) + 새 LoRA(시드 0)로 환경 64개 1라운드(5,120전이)를 수집(온도 1.0, K0)하고, 같은 배치에서:
  1. C9 전체 규모: 원 코드형(joint log-prob 7칸 복제, 손실 합) 대 통제형(joint 1개 × 계수 7)
     (a) 첫 optimizer 스텝(160전이)의 총 손실·기울기 상대 차이, 클리핑(노름 10) 뒤 기울기, 같은 시작점에서 1회 업데이트 차이
     (b) 라운드 전체(32스텝 분할, 원 코드형은 항상 20으로 나눔/통제형은 실제 수) 누적 기울기 — 부분 묶음 규칙 차이(C7) 기록
     두 형식 모두 old 값은 같은 값(생성 경로 저장값)을 써서 C9만 비교한다.
  2. 라운드 시간 분해: 수집(정책·시뮬레이션+렌더링·리셋), 상태 기록, old 값 재계산(C10), 업데이트 32스텝, 저장. 피크 GPU 메모리.
허용 오차(제안): 손실·기울기 상대 차이 1e-5. 넘으면 G1을 다시 연다 (D-006).
사용: python g2_s5_scale.py <W1 ckpt> <out_dir> [--attn flash_attention_2|sdpa] [--chunk 8]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_gpu as G  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import core  # noqa: E402
import openvla_adapter as oa  # noqa: E402
import ppo  # noqa: E402

TOL = 1e-5


def gvec(params):
    return torch.cat([(p.grad.detach().double().reshape(-1) if p.grad is not None
                       else torch.zeros(p.numel(), device=p.device, dtype=torch.float64)) for p in params])


def pvec(params):
    return torch.cat([p.detach().double().reshape(-1) for p in params])


def loss_and_backward(ad, idx_list, divisor, mode, old_lp, old_v, adv_f, ret_f, chunk):
    """한 optimizer 스텝 분량(미니배치 목록)의 손실을 역전파한다 (ppo.ppo_update와 같은 스케일)."""
    tot = 0.0
    for mb in idx_list:
        m = len(mb)
        for j in range(0, m, chunk):
            idx = mb[j:j + chunk]
            c = len(idx)
            lp, v = ad.evaluate(idx)
            dev = lp.device
            o = torch.as_tensor(old_lp[idx], device=dev).view(-1, 1)
            a = torch.as_tensor(adv_f[idx], device=dev).view(-1, 1)
            vp = torch.as_tensor(old_v[idx], device=dev).view(-1, 1)
            r = torch.as_tensor(ret_f[idx], device=dev).view(-1, 1)
            if mode == "original":
                loss = core.policy_loss_original(lp, o.repeat(1, 7), a) + core.value_loss_original(v, vp, r)
            else:
                loss = 7.0 * core.policy_loss_controlled(lp, o, a) + 1.0 * core.value_loss_original(v, vp, r)
            (loss * (c / m) / divisor).backward()
            tot += float(loss) * (c / m) / divisor
    return tot


def rel(a, b):
    return float((a - b).norm() / max(1e-30, float(b.norm())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=64, help="스모크 시험용. 본 측정은 64")
    ap.add_argument("--smoke-steps", type=int, default=0, help="스모크 시험: 라운드 누적 기울기를 앞 몇 스텝만")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(args=vars(a), tol=TOL)
    t_all = time.time()

    def log(m):
        print(m, flush=True)

    policy = G.build_policy(a.ckpt, "bridge_orig", attn=a.attn, seed=a.seed)
    G.set_top_k(policy, "K0")
    res["attn_impl"] = G.attn_impls(policy)
    unnorm = policy.vla.get_action_stats("bridge_orig")
    env = G.GpuEnv("PutOnPlateInScene25Main-v3", unnorm, a.n_envs, seed=a.seed)
    eids = [G.episode_id_for(f"g2s5-seed{a.seed}-round0", k) for k in range(a.n_envs)]

    # ------------------------------------------------ 수집 (시간 분해)
    torch.cuda.reset_peak_memory_stats()
    tm = {}
    r = G.rollout(policy, env, "train", eids, deterministic=False, rng_seed=a.seed, record_obs=True, timing=tm)
    # 상태 기록 비용 (전체 시뮬레이터 상태 64개, 스텝당 1회): 같은 환경에서 80회 get_state 시간
    t0 = time.time()
    for _ in range(80):
        _s = env.u.get_state()
    torch.cuda.synchronize()
    tm["state_record_80_s"] = time.time() - t0
    tm["state_dim"] = int(_s.shape[-1])
    res["collect"] = dict({k: round(v, 2) if isinstance(v, float) else v for k, v in tm.items()},
                          Y=int(r["Y"].sum()), success_last=int(r["success"][-1].sum()), peak=G.gpu_mem())
    log(f"[collect] {res['collect']}")
    env.close()
    torch.cuda.empty_cache()

    batch = G.to_roundbatch(r)
    ad = oa.OpenVLAAdapter(policy, r["obs"], r["instr"], r["actions"])
    n = batch.T * batch.N
    params = list(ad.params_policy) + list(ad.params_value)
    res["n_trainable"] = int(sum(p.numel() for p in params))
    snap0 = ppo.snapshot(ad)

    # ------------------------------------------------ C9 (a) 첫 optimizer 스텝
    old_lp = batch.logprob_gen.reshape(-1).astype(np.float32)
    old_v = batch.value_gen.reshape(-1).astype(np.float32)
    adv, ret, _ = ppo.advantages_and_returns(batch, batch.value_gen, "terminal", "gae")
    adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
    order = ppo.minibatch_order(n, 0)
    plan_c = ppo.plan_steps(order, 8, 20, controlled=True)
    plan_o = ppo.plan_steps(order, 8, 20, controlled=False)
    s0, mbs0, div0 = plan_o[0]
    G_, L_, U_ = {}, {}, {}
    for key in ("original", "original_repeat", "controlled"):
        mode = "controlled" if key == "controlled" else "original"
        ppo.restore(ad, snap0)
        ad.opt_policy.zero_grad(set_to_none=True)
        ad.opt_value.zero_grad(set_to_none=True)
        ad.train_mode()
        L_[key] = loss_and_backward(ad, mbs0, div0, mode, old_lp, old_v, adv_f, ret_f, a.chunk)
        g = gvec(params)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 10.0))
        gc = gvec(params)
        ad.opt_value.step()
        ad.opt_policy.step()
        G_[key] = (g.cpu(), gc.cpu(), gn)
        U_[key] = pvec(params).cpu()
    p0 = torch.cat([q.double().reshape(-1).cpu() for q in snap0["policy"] + snap0["value"]])
    dua = U_["original_repeat"] - U_["original"]
    du = U_["controlled"] - U_["original"]
    res["C9_first_step"] = dict(
        transitions=sum(len(m) for m in mbs0), loss_original=L_["original"], loss_controlled=L_["controlled"],
        loss_rel_diff=abs(L_["original"] - L_["controlled"]) / max(1e-30, abs(L_["original"])),
        grad_rel_diff=rel(G_["controlled"][0], G_["original"][0]),
        grad_cos=float(torch.dot(G_["controlled"][0], G_["original"][0]) / (G_["controlled"][0].norm() * G_["original"][0].norm())),
        grad_norm_original=G_["original"][2], grad_norm_controlled=G_["controlled"][2],
        clipped_grad_rel_diff=rel(G_["controlled"][1], G_["original"][1]),
        update_rel_diff=float(du.norm() / max(1e-30, float((U_["original"] - p0).norm()))),
        update_max_abs_diff=float(du.abs().max()), update_frac_elements_different=float((du != 0).float().mean()),
        param_dtype=str(params[0].dtype),
        aa_floor_original_vs_repeat=dict(
            loss_rel_diff=abs(L_["original"] - L_["original_repeat"]) / max(1e-30, abs(L_["original"])),
            grad_rel_diff=rel(G_["original_repeat"][0], G_["original"][0]),
            clipped_grad_rel_diff=rel(G_["original_repeat"][1], G_["original"][1]),
            update_rel_diff=float(dua.norm() / max(1e-30, float((U_["original"] - p0).norm()))),
            update_frac_elements_different=float((dua != 0).float().mean())))
    log(f"[C9 first step] {res['C9_first_step']}")
    del G_, U_, du, dua

    # ------------------------------------------------ C9 (b) 라운드 전체 누적 기울기 (업데이트 없이)
    full = {}
    if a.smoke_steps:
        plan_o, plan_c = plan_o[:a.smoke_steps], plan_c[:a.smoke_steps]
    for mode, plan in (("original", plan_o), ("controlled", plan_c)):
        ppo.restore(ad, snap0)
        ad.opt_policy.zero_grad(set_to_none=True)
        ad.opt_value.zero_grad(set_to_none=True)
        ad.train_mode()
        t0 = time.time()
        tot = 0.0
        for _s, mbs, div in plan:
            tot += loss_and_backward(ad, mbs, div, mode, old_lp, old_v, adv_f, ret_f, a.chunk)
        full[mode] = (gvec(params).cpu(), tot, time.time() - t0)
    last_div = (plan_o[-1][2], plan_c[-1][2])
    res["C9_round_accumulated"] = dict(
        n_steps=len(plan_c), last_step_divisor_original_controlled=last_div,
        loss_rel_diff=abs(full["original"][1] - full["controlled"][1]) / max(1e-30, abs(full["original"][1])),
        grad_rel_diff=rel(full["controlled"][0], full["original"][0]),
        note="원 코드형은 부분 묶음도 20으로 나눈다(C7). n=5120이면 32스텝이 모두 꽉 차 C7 차이가 없다.",
        sec=[round(full["original"][2], 1), round(full["controlled"][2], 1)])
    log(f"[C9 round] {res['C9_round_accumulated']}")
    del full
    ad.opt_policy.zero_grad(set_to_none=True)
    ad.opt_value.zero_grad(set_to_none=True)

    # ------------------------------------------------ 라운드 시간 분해: 통제형 업데이트(C10 재계산 포함), 기본(비결정) 설정
    ppo.restore(ad, snap0)
    t_rc = {}
    orig_recompute = ppo.recompute_old

    def timed_recompute(*args, **kw):
        t0 = time.time()
        o = orig_recompute(*args, **kw)
        torch.cuda.synchronize()
        t_rc["s"] = time.time() - t0
        return o
    ppo.recompute_old = timed_recompute
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    upd = ppo.ppo_update(ad, batch, ppo.PPOConfig(order_seed=0, chunk=a.chunk, recompute_old=True,
                                                  max_steps=a.smoke_steps or None))
    torch.cuda.synchronize()
    t_upd = time.time() - t0
    ppo.recompute_old = orig_recompute
    peak_upd = G.gpu_mem()
    # 저장 비용: 계층 1(저차원 전체) + RGB 앞 2 에피소드 무손실
    t0 = time.time()
    np.savez_compressed(out / "round0_tier1.npz", actions=r["actions"], lp_gen=r["lp_gen"], v_gen=r["v_gen"],
                        success=r["success"], grasp=r["grasp_latch"], consec=r["consec_latch"], eid=r["episode_id"])
    np.savez_compressed(out / "round0_rgb_ep0-1.npz", rgb=r["obs"][:, :2])
    t_save = time.time() - t0
    collect_s = res["collect"]["total_s"] + res["collect"]["state_record_80_s"]
    res["round_time"] = dict(collect_s=round(collect_s, 1), collect_policy_s=res["collect"]["policy_s"],
                             collect_sim_render_s=res["collect"]["sim_s"], collect_reset_s=res["collect"]["reset_s"],
                             state_record_s=res["collect"]["state_record_80_s"],
                             recompute_old_c10_s=round(t_rc.get("s", float("nan")), 1),
                             update_steps_s=round(t_upd - t_rc.get("s", 0), 1), n_steps=upd["n_steps"],
                             save_s=round(t_save, 1),
                             round_total_s=round(collect_s + t_upd + t_save, 1),
                             peak_update=peak_upd, first_minibatch=upd["first_minibatch"],
                             all_finite=all(s["finite"] for s in upd["steps"]))
    res["round_time"]["extrapolated_390_rounds_gpu_h"] = round(res["round_time"]["round_total_s"] * 390 / 3600, 1)
    res["c10_audit_W1"] = upd["c10_audit"]
    log(f"[round time] {res['round_time']}")
    res["paper_peak_gb"] = 44.4
    f1, fr = res["C9_first_step"], res["C9_round_accumulated"]
    # 허용 오차 1e-5는 4090 기준(손실 상대 차이 2e-8)과 같은 손실 기준. bf16 기울기는 같은 형식을 두 번 돌린 A/A 차이와 비교해 보고
    res["verdict"] = dict(loss_within_tol=bool(f1["loss_rel_diff"] <= TOL and fr["loss_rel_diff"] <= TOL),
                          grad_c9_diff_vs_aa_floor=[f1["grad_rel_diff"], f1["aa_floor_original_vs_repeat"]["grad_rel_diff"]],
                          note="기울기는 bf16. C9 차이가 같은 설정의 A/A 차이 이하이면 실행 잡음과 구분되지 않음. 결정적 설정(S6)에서 다시 확인")
    res["total_min"] = round((time.time() - t_all) / 60, 1)
    (out / "g2_s5.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"[verdict] {res['verdict']} saved")


if __name__ == "__main__":
    main()
