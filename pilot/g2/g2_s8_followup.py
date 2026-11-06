"""G2 후속 F3 (GPU 64 배치 복원 재현성)·F4 (R-a 관통 검사). 규칙은 pilot/G2_followup_prereg.md (실행 전 커밋).

모드 (PhysX CPU와 GPU는 한 프로세스에서 같이 못 쓰므로 모드마다 별도 프로세스):
  ra_cpu      F4 판정용 CPU 측정 (서브스텝 관통, 1스텝 뒤 관통·속도) + CPU 초기 전이 상태. 정책 불필요
  cpu_closed  F3 닫힌 고리 진단: RL 대조, CPU, 추가 샘플링 시드 2개
  gpu_replay  F3 재생 시험: 구성 L0·L0r·L1·L2·L0s × 교란 4종, 기록된 W1 행동 25스텝
  ra_gpu      F4 GPU 초기 전이 (교란 4종 × 행동 출처 4종)
  gpu_closed  F3 닫힌 고리 진단: RL 대조, GPU, 구성 L0(a)·L0(a) 반복·L1(a)·L0(b)
  report      위 결과를 모아 판정·요약 (g2_s8_followup.json)
사용: python g2_s8_followup.py <체크포인트 폴더> <s8 폴더> <출력 폴더> --mode ...
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_gpu as G  # noqa: E402
import g2_s8_restore as S8  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PERT = ["control", "R-a", "R-b", "R-c"]
NB = 64
L_REPLAY = S8.L_REPLAY
H = S8.H
SRCS = [("W1rec", "openvla-7b-rlvla-warmup", "bridge_orig"), ("W1", "openvla-7b-rlvla-warmup", "bridge_orig"),
        ("SFT", "openvla-7b-rlvla-sft_16k", "sft"), ("RL", "openvla-7b-rlvla-rl", "bridge_orig")]


def unnorm_of(ckpt_dir, sub, key):
    cfg = json.loads((Path(ckpt_dir) / sub / "config.json").read_text())
    return {k: np.asarray(v) for k, v in cfg["norm_stats"][key]["action"].items()}


def reset_seeded(env, eids, seeds):
    opts = {"obj_set": "train", "episode_id": torch.as_tensor(np.asarray(eids, np.int64), device=env.u.device), "reconfigure": True}
    with G.sim_nondeterministic_ok():
        env.env.reset(seed=[int(s) for s in seeds], options=opts)


def cpu_penetration(env):
    """CPU 환경 0: 집을 물체와 다른 모든 물체 사이 최대 관통 깊이 (m)와 그 상대 이름."""
    src = env.src_actor(0)._objs[0]
    best, who = 0.0, None
    with G.sim_nondeterministic_ok():
        for c in env.u.scene.px.get_contacts():
            ents = [c.bodies[0].entity, c.bodies[1].entity]
            if src not in ents or ents[0] is ents[1]:
                continue
            other = ents[1] if ents[0] is src else ents[0]
            mn = min([pt.separation for pt in c.points] + [0.0])
            if -mn > best:
                best, who = -mn, other.name
    return float(best), who


def speeds(env, i):
    return (float(torch.linalg.norm(env.src_actor(i).linear_velocity[i])),
            float(torch.linalg.norm(env.tgt_actor(i).linear_velocity[i])))


# ---------------------------------------------------------------- F4 CPU
def mode_ra_cpu(ckpt_dir, recs, out, firsts):
    bridge = unnorm_of(ckpt_dir, SRCS[0][1], "bridge_orig")
    env = S8.Env("cpu", bridge)
    meas, trans = {}, {}
    t0 = time.time()
    for j, rec in enumerate(recs):
        for k in PERT:
            # (1) 서브스텝 1회 뒤 관통
            env.reset([rec["eid"]])
            env.set_full([rec["base"]])
            env.perturb([k], [rec["obj_pose0"]])
            with G.sim_nondeterministic_ok():
                env.u.scene.px.step()
            d1, w1 = cpu_penetration(env)
            # (2) 기록된 W1 행동 1스텝 뒤 관통·속도
            env.set_unnorm(bridge)
            env.reset([rec["eid"]])
            env.set_full([rec["base"]])
            env.perturb([k], [rec["obj_pose0"]])
            env.step(torch.as_tensor(rec["tokens"][rec["s0"]]).reshape(1, 7))
            d2, w2 = cpu_penetration(env)
            vs, vt = speeds(env, 0)
            meas[(j, k)] = dict(pen_substep_m=d1, pen_substep_with=w1, pen_1step_m=d2, pen_1step_with=w2,
                                src_speed_1step=vs, tgt_speed_1step=vt)
            # 초기 전이 (행동 출처 4종)
            for name, sub, key in SRCS:
                env.set_unnorm(unnorm_of(ckpt_dir, sub, key))
                act = rec["tokens"][rec["s0"]] if name == "W1rec" else firsts[name][(j, k)]
                env.reset([rec["eid"]])
                env.set_full([rec["base"]])
                env.perturb([k], [rec["obj_pose0"]])
                env.step(torch.as_tensor(act).reshape(1, 7))
                trans[(j, k, name)] = S8.key_state(env, 0)
        print(f"[ra_cpu] {j + 1}/{len(recs)} {time.time() - t0:.0f}s", flush=True)
    env.close()
    pickle.dump(dict(meas=meas, trans=trans, sec=time.time() - t0), open(out / "F4_cpu.pkl", "wb"))


# ---------------------------------------------------------------- F4 GPU
def mode_ra_gpu(ckpt_dir, recs, out, firsts):
    env = S8.Env("gpu", unnorm_of(ckpt_dir, SRCS[0][1], "bridge_orig"), n=NB)
    trans, spd = {}, {}
    t0 = time.time()
    m = len(recs)
    for name, sub, key in SRCS:
        env.set_unnorm(unnorm_of(ckpt_dir, sub, key))
        for k in PERT:
            slots = [i % m for i in range(NB)]
            reset_seeded(env, [recs[j]["eid"] for j in slots], [0] * NB)
            env.set_full([recs[j]["base"] for j in slots])
            env.perturb([k] * NB, [recs[j]["obj_pose0"] for j in slots])
            acts = [recs[j]["tokens"][recs[j]["s0"]] if name == "W1rec" else firsts[name][(j, k)] for j in slots]
            env.step(torch.as_tensor(np.stack(acts)))
            for i in range(m):
                trans[(i, k, name)] = S8.key_state(env, i)
                spd[(i, k, name)] = speeds(env, i)
        print(f"[ra_gpu] {name} {time.time() - t0:.0f}s", flush=True)
    env.close()
    pickle.dump(dict(trans=trans, speed=spd, sec=time.time() - t0), open(out / "F4_gpu.pkl", "wb"))


# ---------------------------------------------------------------- F3 재생
def layouts(m):
    L = {}
    L["L0"] = [("rec", i % m) for i in range(NB)], [0] * NB
    L["L0r"] = L["L0"]
    L["L1"] = [("rec", (5 * i + 3) % m) for i in range(NB)], [0] * NB
    L["L2"] = [("rec", (i // 2) % m) if i % 2 == 0 else ("fill", G.episode_id_for("g2s8f-filler", i)) for i in range(NB)], [0] * NB
    L["L0s"] = L["L0"][0], list(range(NB))
    return L


def mode_gpu_replay(ckpt_dir, recs, out):
    env = S8.Env("gpu", unnorm_of(ckpt_dir, SRCS[0][1], "bridge_orig"), n=NB)
    m = len(recs)
    res = {}
    t0 = time.time()
    for lname, (slots, seeds) in layouts(m).items():
        for k in PERT:
            eids = [recs[x]["eid"] if t == "rec" else x for t, x in slots]
            reset_seeded(env, eids, seeds)
            states = []
            for i, (t, x) in enumerate(slots):
                states.append(recs[x]["base"] if t == "rec" else env.get_full(i))
            env.set_full(states)
            env.perturb([k if t == "rec" else "control" for t, _ in slots],
                        [recs[x]["obj_pose0"] if t == "rec" else np.zeros(7) for t, x in slots])
            st_hist, hs_hist, ks_hist = [], [], []
            for s in range(L_REPLAY):
                toks = [recs[x if t == "rec" else i % m]["tokens"][recs[x if t == "rec" else i % m]["s0"] + s] for i, (t, x) in enumerate(slots)]
                rgb, _ = env.step(torch.as_tensor(np.stack(toks)))
                st_hist.append(env.u.get_state().detach().cpu().numpy())
                px = rgb.cpu().numpy()
                hs_hist.append([hashlib.sha256(px[i].tobytes()).hexdigest() for i in range(NB)])
                ks_hist.append([S8.key_state(env, i) for i in range(NB)])
            st_hist = np.stack(st_hist)  # [T, NB, D]
            for i, (t, x) in enumerate(slots):
                if t != "rec":
                    continue
                res[(lname, k, i)] = dict(j=x, state=st_hist[:, i].copy(), hashes=[h[i] for h in hs_hist],
                                          key=[kk[i] for kk in ks_hist])
            print(f"[gpu_replay] {lname} {k} {time.time() - t0:.0f}s", flush=True)
    env.close()
    pickle.dump(dict(res=res, sec=time.time() - t0), open(out / "F3_replay.pkl", "wb"))


# ---------------------------------------------------------------- F3 닫힌 고리 (RL 대조)
def mode_gpu_closed(ckpt_dir, recs, out):
    name, sub, key = SRCS[3]
    pol = S8.load_policy(ckpt_dir, sub, key)
    env = S8.Env("gpu", unnorm_of(ckpt_dir, sub, key), n=NB)
    m = len(recs)
    L = layouts(m)
    runs = [("L0_a", "L0", 7100), ("L0_a_repeat", "L0", 7100), ("L1_a", "L1", 7100), ("L0_b", "L0", 7200)]
    res = {}
    t0 = time.time()
    for rname, lname, seed in runs:
        slots, seeds = L[lname]
        reset_seeded(env, [recs[x]["eid"] for _, x in slots], seeds)
        env.set_full([recs[x]["base"] for _, x in slots])
        env.perturb(["control"] * NB, [recs[x]["obj_pose0"] for _, x in slots])
        torch.manual_seed(seed)
        rgb, ins = env.obs(), [recs[x]["instr"] for _, x in slots]
        info, acts = None, []
        for t in range(H):
            a = S8.policy_tokens(pol, rgb, ins)
            acts.append(a.cpu().numpy())
            rgb, info = env.step(a)
        Y = (info["success"] & info["is_src_obj_grasped"]).cpu().numpy()
        res[rname] = dict(slots=[x for _, x in slots], Y=Y.astype(bool).tolist(), actions=np.stack(acts))
        print(f"[gpu_closed] {rname} {Y.mean():.3f} {time.time() - t0:.0f}s", flush=True)
    env.close()
    pickle.dump(dict(res=res, sec=time.time() - t0), open(out / "F3_closed_gpu.pkl", "wb"))


def mode_cpu_closed(ckpt_dir, recs, out, valid):
    name, sub, key = SRCS[3]
    pol = S8.load_policy(ckpt_dir, sub, key)
    env = S8.Env("cpu", unnorm_of(ckpt_dir, sub, key))
    res = {}
    t0 = time.time()
    for base in (2900, 4900):
        for j in valid:
            rec = recs[j]
            torch.manual_seed(base + j)
            env.reset([rec["eid"]])
            env.set_full([rec["base"]])
            env.perturb(["control"], [rec["obj_pose0"]])
            rgb, ins = env.obs(), [rec["instr"]]
            info = None
            for t in range(H):
                rgb, info = env.step(S8.policy_tokens(pol, rgb, ins))
            res[(base, j)] = bool(info["success"][0]) and bool(info["is_src_obj_grasped"][0])
        print(f"[cpu_closed] seed base {base} {np.mean([res[(base, j)] for j in valid]):.3f} {time.time() - t0:.0f}s", flush=True)
    env.close()
    pickle.dump(dict(res=res, sec=time.time() - t0), open(out / "F3_closed_cpu.pkl", "wb"))


# ---------------------------------------------------------------- 보고
def kd(a, b):
    return S8.key_diff(a, b)


def mode_report(recs, s8dir, out, valid):
    s8 = json.loads((s8dir / "g2_s8.json").read_text())
    R = {"rules": "pilot/G2_followup_prereg.md", "valid_srec": valid}
    # ---- F3 재생
    rp = pickle.load(open(out / "F3_replay.pkl", "rb"))["res"]
    ref = {}
    for (ln, k, i), v in rp.items():
        if ln == "L0" and (v["j"], k) not in ref:
            ref[(v["j"], k)] = v
    comps = []
    for (ln, k, i), v in rp.items():
        r = ref[(v["j"], k)]
        if v is r:
            continue
        st_eq = bool(np.array_equal(v["state"], r["state"]))
        px_eq = v["hashes"] == r["hashes"]
        first = None
        if not (st_eq and px_eq):
            for s in range(L_REPLAY):
                if not np.array_equal(v["state"][s], r["state"][s]) or v["hashes"][s] != r["hashes"][s]:
                    first = s
                    break
        d = [kd(v["key"][s], r["key"][s]) for s in range(L_REPLAY)]
        comps.append(dict(layout=ln, kind=k, slot=i, j=v["j"], valid=v["j"] in valid, state_equal=st_eq, pixel_equal=px_eq,
                          first_diff_step=first, src_max_m=max(x["src_m"] for x in d), tgt_max_m=max(x["tgt_m"] for x in d),
                          qpos_max_rad=max(x["qpos_rad"] for x in d)))
    cv = [c for c in comps if c["valid"]]
    by = {}
    for c in cv:
        b = by.setdefault(f"{c['layout']}|{c['kind']}", dict(n=0, identical=0, src_max_m=0.0))
        b["n"] += 1
        b["identical"] += int(c["state_equal"] and c["pixel_equal"])
        b["src_max_m"] = max(b["src_max_m"], c["src_max_m"])
    R["F3_replay"] = dict(pass_=all(c["state_equal"] and c["pixel_equal"] for c in cv), n_comparisons_valid=len(cv),
                          n_identical_valid=sum(c["state_equal"] and c["pixel_equal"] for c in cv),
                          n_comparisons_all24=len(comps), n_identical_all24=sum(c["state_equal"] and c["pixel_equal"] for c in comps),
                          by_layout_kind=by, mismatches=[c for c in comps if not (c["state_equal"] and c["pixel_equal"])][:200])
    # ---- F3 닫힌 고리
    g = pickle.load(open(out / "F3_closed_gpu.pkl", "rb"))["res"]
    c = pickle.load(open(out / "F3_closed_cpu.pkl", "rb"))["res"]
    s8rl = {(x["j"]): x for x in s8["C_link"]["RL"]["per_item"] if x["k"] == "control"}
    gpu_runs = {}
    for rn, v in g.items():
        per = {}
        for x, y in zip(v["slots"], v["Y"]):
            per.setdefault(x, []).append(y)
        gpu_runs[rn] = dict(rate_valid_slotlevel=round(100 * np.mean([y for x, y in zip(v["slots"], v["Y"]) if x in valid]), 1),
                            rate_valid_elementmean=round(100 * np.mean([np.mean(per[j]) for j in valid]), 1), per_element=per)
    same = [a == b for a, b in zip(g["L0_a"]["Y"], g["L0_a_repeat"]["Y"])]
    act_same = bool(np.array_equal(g["L0_a"]["actions"], g["L0_a_repeat"]["actions"]))
    cpu_rates = {"S8 (900+j)": round(100 * np.mean([s8rl[j]["cpu"] for j in valid]), 1)}
    for base in (2900, 4900):
        cpu_rates[f"{base}+j"] = round(100 * np.mean([c[(base, j)] for j in valid]), 1)
    R["F3_closed_loop_RL_control"] = dict(
        gpu={k: {kk: vv for kk, vv in v.items() if kk != "per_element"} for k, v in gpu_runs.items()},
        gpu_S8=round(100 * np.mean([s8rl[j]["gpu"] for j in valid]), 1),
        gpu_L0_repeat_same_Y_frac=round(float(np.mean(same)), 3), gpu_L0_repeat_actions_identical=act_same,
        cpu=cpu_rates,
        cpu_per_element={j: [s8rl[j]["cpu"], c[(2900, j)], c[(4900, j)]] for j in valid},
        gpu_per_element={j: {rn: gpu_runs[rn]["per_element"][j] for rn in gpu_runs} for j in valid})
    # ---- F4
    fc = pickle.load(open(out / "F4_cpu.pkl", "rb"))
    fg = pickle.load(open(out / "F4_gpu.pkl", "rb"))
    meas = fc["meas"]

    def excluded(j, k):
        x = meas[(j, k)]
        return x["pen_substep_m"] > 0.005 or x["pen_1step_m"] > 0.005 or x["src_speed_1step"] > 0.5 or x["tgt_speed_1step"] > 0.5
    ex = [j for j in valid if excluded(j, "R-a")]
    keep = [j for j in valid if j not in ex]
    R["F4_Ra_rule"] = dict(excluded=ex, n_excluded=len(ex), n_valid=len(valid), revisit_definition=len(ex) > len(valid) / 4,
                           per_element={j: meas[(j, "R-a")] for j in valid})
    R["F4_other_kinds_rule_applied_descriptive"] = {k: [j for j in valid if excluded(j, k)] for k in PERT if k != "R-a"}
    R["B_penetration_recheck_control_all24"] = dict(
        over_5mm=[j for j in range(len(recs)) if meas[(j, "control")]["pen_substep_m"] > 0.005 or meas[(j, "control")]["pen_1step_m"] > 0.005],
        max_substep_m=max(meas[(j, "control")]["pen_substep_m"] for j in range(len(recs))),
        max_1step_m=max(meas[(j, "control")]["pen_1step_m"] for j in range(len(recs))),
        per_element={j: meas[(j, "control")] for j in range(len(recs))})
    T = {}
    for k in PERT:
        for set_name, js in (("kept", keep), ("valid21", valid)):
            if k != "R-a" and set_name == "kept":
                continue
            for name, _, _ in SRCS:
                d = [kd(fg["trans"][(j, k, name)], fc["trans"][(j, k, name)]) for j in js]
                T[f"{k}|{set_name}|{name}"] = dict(n=len(js), src_median_m=float(np.median([x["src_m"] for x in d])),
                                                   src_max_m=float(np.max([x["src_m"] for x in d])),
                                                   n_src_over_1cm=int(sum(x["src_m"] > 0.01 for x in d)),
                                                   tgt_max_m=float(np.max([x["tgt_m"] for x in d])),
                                                   qpos_median_rad=float(np.median([x["qpos_rad"] for x in d])),
                                                   over_1cm=[j for j, x in zip(js, d) if x["src_m"] > 0.01])
    R["F4_initial_transition_cpu_vs_gpu"] = T
    R["F4_gpu_speed_Ra_valid"] = {j: fg["speed"][(j, "R-a", "W1rec")] for j in valid}
    succ = {}
    for pol in ("W1", "SFT", "RL"):
        pi = {x["j"]: x for x in s8["C_link"][pol]["per_item"] if x["k"] == "R-a"}
        succ[pol] = dict(cpu_kept=round(100 * np.mean([pi[j]["cpu"] for j in keep]), 1), gpu_kept=round(100 * np.mean([pi[j]["gpu"] for j in keep]), 1),
                         cpu_valid21=s8["C_link"][pol]["cpu"]["R-a"], gpu_valid21=s8["C_link"][pol]["gpu"]["R-a"], n_kept=len(keep))
    R["F4_Ra_success_from_S8_per_item"] = succ
    (out / "g2_s8_followup.json").write_text(json.dumps(R, indent=1, default=float, ensure_ascii=False))
    print(json.dumps({k: R[k] for k in ("F3_replay",)}, default=float)[:3000])
    print(json.dumps({k: R[k] for k in ("F3_closed_loop_RL_control",)}, default=float)[:1500])
    print(json.dumps({k: R[k] for k in ("F4_Ra_rule", "B_penetration_recheck_control_all24")}, default=float)[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("s8_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--mode", required=True)
    a = ap.parse_args()
    s8dir, out = Path(a.s8_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = pickle.load(open(s8dir / "srec.pkl", "rb"))
    s8 = json.loads((s8dir / "g2_s8.json").read_text())
    valid = [b["j"] for b in s8["B_validity"]["cases"] if b["valid"]]
    firsts = {n: pickle.load(open(s8dir / f"C_cpu_{n}.pkl", "rb"))["first"] for n in ("W1", "SFT", "RL")}
    if a.mode == "ra_cpu":
        mode_ra_cpu(a.ckpt_dir, recs, out, firsts)
    elif a.mode == "ra_gpu":
        mode_ra_gpu(a.ckpt_dir, recs, out, firsts)
    elif a.mode == "gpu_replay":
        mode_gpu_replay(a.ckpt_dir, recs, out)
    elif a.mode == "gpu_closed":
        mode_gpu_closed(a.ckpt_dir, recs, out)
    elif a.mode == "cpu_closed":
        mode_cpu_closed(a.ckpt_dir, recs, out, valid)
    elif a.mode == "report":
        mode_report(recs, s8dir, out, valid)


if __name__ == "__main__":
    main()
