"""G-PRE 남은 작업 T2 (R-a′ 관통 검사, CPU)·T3 (평가 잡음, GPU). 규칙: pilot/GPRE_T2T3_prereg.md (실행 전 커밋).

모드 (PhysX CPU와 GPU는 한 프로세스에서 같이 못 쓰므로 별도 프로세스):
  t2      CPU: S_rec 타당 원소 21개 × (대조, R-a′, R-b, R-c) 제외 규칙 측정 + 기술 보고 값
  t3      GPU: 공개 RL 체크포인트, 실행 3조건 × 192, 칸 배정을 바꾼 평가 2 + EEPose 배치 0 재현 점검
  report  판정·요약 → <out>/gpre_t2t3.json
사용: python g2_gpre_t2t3.py <체크포인트 폴더> <s8 폴더> <s4 결과 폴더> <s8f 폴더> <출력 폴더> --mode t2|t3|report
"""
import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_gpu as G  # noqa: E402
import g2_s8_followup as F  # noqa: E402
import g2_s8_restore as S8  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

KINDS = ["control", "R-a'", "R-b", "R-c"]
PEN_MAX, SPEED_MAX = 0.005, 0.5
EXEC = [("Position", "PutOnPlateInScene25Position-v1", "test"),
        ("EEPose", "PutOnPlateInScene25EEPose-v1", "test"),
        ("PositionChangeTo", "PutOnPlateInScene25PositionChangeTo-v1", "test")]
N_EP, NB = 192, 64
SEED = 20261106


# ---------------------------------------------------------------- T2
def mode_t2(ckpt_dir, recs, valid, out):
    bridge = F.unnorm_of(ckpt_dir, "openvla-7b-rlvla-warmup", "bridge_orig")
    env = S8.Env("cpu", bridge)
    meas = {}
    t0 = time.time()
    for j in valid:
        rec = recs[j]
        tok = np.asarray(rec["tokens"][rec["s0"]])
        for k in KINDS:
            # (1) 서브스텝 1회 뒤 관통
            env.reset([rec["eid"]])
            env.set_full([rec["base"]])
            z_base = float(env.src_actor(0).pose.p[0, 2])
            env.perturb([k], [rec["obj_pose0"]])
            with G.sim_nondeterministic_ok():
                env.u.scene.px.step()
            d1, w1 = F.cpu_penetration(env)
            g_sub = bool(env.grasp_now()[0])
            # (2) 기록된 W1 행동 1스텝 뒤 관통·속도
            env.reset([rec["eid"]])
            env.set_full([rec["base"]])
            env.perturb([k], [rec["obj_pose0"]])
            env.step(torch.as_tensor(tok).reshape(1, 7))
            d2, w2 = F.cpu_penetration(env)
            vs, vt = F.speeds(env, 0)
            z1 = float(env.src_actor(0).pose.p[0, 2])
            meas[(j, k)] = dict(pen_substep_m=d1, pen_substep_with=w1, pen_1step_m=d2, pen_1step_with=w2,
                                src_speed_1step=vs, tgt_speed_1step=vt,
                                z_above_t0_m=z_base - float(np.asarray(rec["obj_pose0"])[2]),
                                grasp_after_substep=g_sub, grasp_after_1step=bool(env.grasp_now()[0]),
                                dz_1step_m=z1 - z_base, w1_gripper_token=int(tok[6]))
        print(f"[t2] elem {j} {time.time() - t0:.0f}s", flush=True)
    env.close()
    pickle.dump(dict(meas=meas, sec=time.time() - t0), open(out / "T2_cpu.pkl", "wb"))


# ---------------------------------------------------------------- T3
def s4_sets(s4dir):
    sets = json.loads((s4dir / "g2_s4_episode_sets.json").read_text())
    for lab, _env, _split in EXEC:
        re = [G.episode_id_for(f"g2-eval-{_env}-{_split}", k) for k in range(N_EP)]
        assert [int(x) for x in sets["sets"][lab]] == [int(x) for x in re], lab
        assert hashlib.sha256(json.dumps(sets["sets"][lab]).encode()).hexdigest() == sets["sha256"][lab], lab
    return sets["sets"], sets["sha256"]


def perms():
    rng = np.random.default_rng(SEED)
    return {lab: rng.permutation(N_EP) for lab, _e, _s in EXEC}


def mode_t3(ckpt_dir, s4dir, out):
    sets, _sha = s4_sets(s4dir)
    P = perms()
    epf = out / "T3_episodes.jsonl"
    done = set()
    if epf.exists():
        for line in epf.read_text().splitlines():
            d = json.loads(line)
            done.add((d["run"], d["cond"], d["batch"]))
    t0 = time.time()
    policy = G.build_policy(str(Path(ckpt_dir) / "openvla-7b-rlvla-rl"), "bridge_orig", attn="flash_attention_2", seed=0)
    G.set_top_k(policy, "K0")
    ustate = policy.vla.get_action_stats("bridge_orig")
    print(f"[load] {time.time() - t0:.0f}s attn={G.attn_impls(policy)}", flush=True)
    timing = []
    for lab, env_id, split in EXEC:
        todo = ([("repro", 0)] if lab == "EEPose" else []) + [("perm", b) for b in range(3)]
        todo = [t for t in todo if (t[0], lab, t[1]) not in done]
        if not todo:
            continue
        env = G.GpuEnv(env_id, ustate, NB, seed=0)
        for run, b in todo:
            ks = list(range(NB * b, NB * (b + 1))) if run == "repro" else [int(x) for x in P[lab][NB * b:NB * (b + 1)]]
            eids = [sets[lab][k] for k in ks]
            tm = {}
            r = G.rollout(policy, env, split, eids, deterministic=True, rng_seed=1000 + b, timing=tm)
            sel = r["selected"]
            with epf.open("a") as f:
                for i in range(NB):
                    f.write(json.dumps(dict(run=run, ckpt="rl", cond=lab, env_id=env_id, obj_set=split, batch=b, slot=i,
                                            k=ks[i], episode_id=int(eids[i]), instruction=r["instr"][i],
                                            Y=bool(r["Y"][i]), success=bool(r["success"][-1, i]),
                                            success_any=bool(r["success"][:, i].any()),
                                            grasp_latch=bool(r["grasp_latch"][-1, i]),
                                            **{kk: int(v[i]) for kk, v in sel.items()})) + "\n")
            timing.append(dict(run=run, cond=lab, batch=b, **{kk: round(v, 1) for kk, v in tm.items()}))
            print(f"[{run} {lab} b{b}] Y {r['Y'].sum()}/64 {tm['total_s']:.0f}s", flush=True)
        env.close()
    with (out / "T3_timing.jsonl").open("a") as f:
        for t in timing:
            f.write(json.dumps(t) + "\n")


# ---------------------------------------------------------------- report
def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def report_t2(out, s8f):
    m = pickle.load(open(out / "T2_cpu.pkl", "rb"))["meas"]
    f4 = pickle.load(open(s8f / "F4_cpu.pkl", "rb"))["meas"]
    elems = sorted({j for j, _k in m})
    R = {}
    for k in KINDS:
        rows, excl, speed_only = [], [], []
        for j in elems:
            x = m[(j, k)]
            c1 = x["pen_substep_m"] > PEN_MAX
            c2p = x["pen_1step_m"] > PEN_MAX
            c2v = x["src_speed_1step"] > SPEED_MAX or x["tgt_speed_1step"] > SPEED_MAX
            ex = c1 or c2p or c2v
            if ex:
                excl.append(j)
                if c2v and not (c1 or c2p):
                    speed_only.append(j)
            rows.append(dict(j=j, excluded=ex, **{kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in x.items()}))
        n = len(elems)
        d = dict(n=n, excluded=excl, n_excluded=len(excl), speed_only=speed_only,
                 drop_from_E4b=len(excl) > n / 4, rows=rows)
        if k == "R-a'":
            d["describe"] = dict(
                grasp_after_substep=sum(r["grasp_after_substep"] for r in rows),
                grasp_after_1step=sum(r["grasp_after_1step"] for r in rows),
                z_above_t0_mm=dict(median=round(1000 * float(np.median([r["z_above_t0_m"] for r in rows])), 1),
                                   max=round(1000 * float(np.max([r["z_above_t0_m"] for r in rows])), 1)),
                dz_1step_mm=dict(median=round(1000 * float(np.median([r["dz_1step_m"] for r in rows])), 1),
                                 min=round(1000 * float(np.min([r["dz_1step_m"] for r in rows])), 1)),
                w1_gripper_tokens=sorted({r["w1_gripper_token"] for r in rows}))
        if k != "R-a'":
            keys = ("pen_substep_m", "pen_1step_m", "src_speed_1step", "tgt_speed_1step")
            d["same_as_F4"] = all(m[(j, k)][kk] == f4[(j, k)][kk] for j in elems for kk in keys)
        R[k] = d
    return R


def report_t3(out, s4dir):
    s4 = [json.loads(x) for x in (s4dir / "g2_s4_episodes.jsonl").read_text().splitlines()]
    s4 = {(r["cond"], r["k"]): r for r in s4 if r["ckpt"] == "rl"}
    t3 = [json.loads(x) for x in (out / "T3_episodes.jsonl").read_text().splitlines()]
    res = dict(conds={})
    # 재현 점검
    rp = [r for r in t3 if r["run"] == "repro"]
    skip = {"run", "slot"}
    res["repro_EEPose_b0"] = dict(n=len(rp), all_fields_equal=all(
        all(r[f] == s4[(r["cond"], r["k"])][f] for f in r if f not in skip and f in s4[(r["cond"], r["k"])]) for r in rp),
        Y_equal=sum(r["Y"] == s4[(r["cond"], r["k"])]["Y"] for r in rp))
    pairs = {}
    for lab, _e, _s in EXEC:
        pr = {r["k"]: r for r in t3 if r["run"] == "perm" and r["cond"] == lab}
        assert sorted(pr) == list(range(N_EP)), lab
        y1 = np.array([s4[(lab, k)]["Y"] for k in range(N_EP)], float)
        y2 = np.array([pr[k]["Y"] for k in range(N_EP)], float)
        same_slot = sum(pr[k]["batch"] == k // NB and pr[k]["slot"] == k % NB for k in range(N_EP))
        pairs[lab] = (y1, y2)
        D = int(np.sum(y1 != y2))
        pbar = (y1.sum() + y2.sum()) / (2 * N_EP)
        b10, b01 = int(np.sum((y1 == 1) & (y2 == 0))), int(np.sum((y1 == 0) & (y2 == 1)))
        res["conds"][lab] = dict(Y1=round(100 * y1.mean(), 1), Y2=round(100 * y2.mean(), 1),
                                 diff=round(100 * (y2.mean() - y1.mean()), 1), disagree=D, only1=b10, only2=b01,
                                 mcnemar_p=round(mcnemar(b10, b01), 4),
                                 sd_eval_pp=round(100 * math.sqrt(D / (2 * N_EP ** 2)), 2),
                                 sd_binom_pp=round(100 * math.sqrt(pbar * (1 - pbar) / N_EP), 2),
                                 same_batch_and_slot=int(same_slot))

    def stat(pp):
        V = sum(np.sum(a != b) / (2 * N_EP ** 2) for a, b in pp.values())
        B = 0.0
        for a, b in pp.values():
            p = (a.sum() + b.sum()) / (2 * N_EP)
            B += p * (1 - p) / N_EP
        return 100 * math.sqrt(V) / 3, V / B if B > 0 else float("nan"), V, B

    sd, r, V, B = stat(pairs)
    rng = np.random.default_rng(SEED)
    bs_sd, bs_r = [], []
    for _ in range(10000):
        pb = {}
        for lab, (a, b) in pairs.items():
            idx = rng.integers(0, N_EP, N_EP)
            pb[lab] = (a[idx], b[idx])
        s, rr, _v, _b = stat(pb)
        bs_sd.append(s)
        bs_r.append(rr)
    crude = sum((a.mean() - b.mean()) ** 2 for a, b in pairs.values()) / 2
    res["exec_axis"] = dict(sd_eval_pp=round(sd, 3), ci95=[round(float(np.percentile(bs_sd, 2.5)), 3),
                                                          round(float(np.percentile(bs_sd, 97.5)), 3)],
                            sd_binom_at_observed_p_pp=round(100 * math.sqrt(B) / 3, 3),
                            ratio_r=round(r, 3), ratio_r_ci95=[round(float(np.percentile(bs_r, 2.5)), 3),
                                                               round(float(np.percentile(bs_r, 97.5)), 3)],
                            spec_assumption_pp=1.8,
                            t1_main_pp=math.ceil(round(10 * sd, 6)) / 10,
                            t1_sensitivity_pp=round(float(np.percentile(bs_sd, 97.5)), 2),
                            crude_check=dict(sum_sq_mean_diff_over2=round(crude, 6), sum_V=round(V, 6)))
    tm = [json.loads(x) for x in (out / "T3_timing.jsonl").read_text().splitlines()]
    res["gpu_s"] = round(sum(t["total_s"] for t in tm), 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("s8_dir")
    ap.add_argument("s4_dir")
    ap.add_argument("s8f_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--mode", required=True)
    a = ap.parse_args()
    s8dir, s4dir, s8f, out = Path(a.s8_dir), Path(a.s4_dir), Path(a.s8f_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "t2":
        recs = pickle.load(open(s8dir / "srec.pkl", "rb"))
        s8 = json.loads((s8dir / "g2_s8.json").read_text())
        valid = [b["j"] for b in s8["B_validity"]["cases"] if b["valid"]]
        mode_t2(a.ckpt_dir, recs, valid, out)
    elif a.mode == "t3":
        mode_t3(a.ckpt_dir, s4dir, out)
    elif a.mode == "report":
        R = dict(T2=report_t2(out, s8f) if (out / "T2_cpu.pkl").exists() else None,
                 T3=report_t3(out, s4dir) if (out / "T3_episodes.jsonl").exists() else None)
        (out / "gpre_t2t3.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
        if R["T2"]:
            for k, d in R["T2"].items():
                print("T2", k, "excluded", d["n_excluded"], d["excluded"], "speed_only", d["speed_only"],
                      "drop", d["drop_from_E4b"], "sameF4", d.get("same_as_F4"), d.get("describe", ""))
        if R["T3"]:
            print("T3", json.dumps(R["T3"], ensure_ascii=False))


if __name__ == "__main__":
    main()
