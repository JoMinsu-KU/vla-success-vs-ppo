"""P4 — 저장 정확성(C6)·재렌더링 일치율·집기 접촉 상태의 복원 (W1 정책 실제 롤아웃, Windows 네이티브, 단일 프로세스).

1. W1 정책(온도 0.6, K0, 0-B LoRA 병합)으로 학습 과제 분포 에피소드 K개를 돌리며 스텝마다 전체 상태
   (평탄화 시뮬레이터 상태, 제어기 상태, 과제 상태), 관측 RGB, 행동 토큰·역정규화 행동, log-prob·value, info를 기록한다.
2. 저장 정확성: storage.RoundStore로 쓰고 다시 읽어 배열·바이트·해시 완전 일치 확인, 크기 기록.
3. 재렌더링: 디스크에서 읽은 계층 1 상태로 복원한 뒤 관측을 다시 렌더링해 저장 RGB와 비교 (스텝 표본).
4. 접촉 상태 복원: 집기가 2스텝 이상 이어진 첫 시점의 다음 상태를 디스크 상태로 복원하고, 기록된 행동 토큰
   25개를 재생해 원래 궤적(상태·집기·info·픽셀)과 비교. 대조군: pose만 복원(속도 0, 제어기·과제 상태 미복원).
사용: python test_storage_contact.py <W1 ckpt> <out_dir> [K=8]
"""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "learner"))
import p2_eval  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import storage  # noqa: E402

ENV = "PutOnPlateInScene25Main-v3"
CAM = "3rd_view_camera"
L_REPLAY = 25


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- 상태 평탄화·복원
def ctrl_leaves(d, prefix=""):
    out = []
    for k in sorted(d):
        v = d[k]
        if isinstance(v, dict):
            out += ctrl_leaves(v, prefix + k + "/")
        else:
            out.append((prefix + k, v))
    return out


def get_full_state(w):
    u = w.env.unwrapped
    sim = u.get_state()[0].float().cpu().numpy().astype(np.float32)
    leaves = ctrl_leaves(u.agent.controller.get_state())
    ctrl = np.concatenate([torch.as_tensor(v).float().reshape(-1).cpu().numpy() for _, v in leaves]).astype(np.float32) \
        if leaves else np.zeros(0, np.float32)
    es = u.episode_stats
    task = np.array([int(es["is_src_obj_grasped"][0]), int(es["consecutive_grasp"][0]), int(es["src_on_target"][0]),
                     int(u.consecutive_grasp[0]), int(u._elapsed_steps[0])], np.int64)
    return sim, ctrl, task


def set_full_state(w, sim, ctrl, task, full=True):
    u = w.env.unwrapped
    s = torch.as_tensor(sim, dtype=torch.float32).reshape(1, -1)
    if full:
        u.set_state(s)
        tmpl = u.agent.controller.get_state()
        leaves = ctrl_leaves(tmpl)
        pos, new = 0, copy.deepcopy(tmpl)
        for name, v in leaves:
            n = torch.as_tensor(v).numel()
            val = torch.as_tensor(ctrl[pos:pos + n]).reshape(torch.as_tensor(v).shape).to(torch.as_tensor(v).dtype)
            node = new
            parts = name.split("/")
            for p_ in parts[:-1]:
                node = node[p_]
            node[parts[-1]] = val
            pos += n
        u.agent.controller.set_state(new)
        es = u.episode_stats
        es["is_src_obj_grasped"] = torch.tensor([bool(task[0])])
        es["consecutive_grasp"] = torch.tensor([bool(task[1])])
        es["src_on_target"] = torch.tensor([bool(task[2])])
        u.consecutive_grasp = torch.tensor([int(task[3])], dtype=torch.int32)
        u._elapsed_steps = torch.tensor([int(task[4])], dtype=u._elapsed_steps.dtype)
    else:  # pose만: 모든 속도 0, 제어기·과제 상태는 지금 값 그대로
        st = u.get_state_dict()
        flat = torch.as_tensor(sim, dtype=torch.float32).reshape(1, -1)
        u.set_state(flat)
        sd = u.get_state_dict()
        for v in sd["actors"].values():
            v[..., 7:13] = 0
        for v in sd["articulations"].values():
            v[..., 7:13] = 0
            if v.shape[-1] > 13:
                nq = (v.shape[-1] - 13) // 2
                v[..., 13 + nq:] = 0
        u.set_state_dict(sd)
        del st


def render_now(w):
    obs = w.env.unwrapped.get_obs()
    return obs["sensor_data"][CAM]["rgb"].to(torch.uint8)[0].cpu().numpy()


def grasp_now(w):
    u = w.env.unwrapped
    name = u.carrot_names[int(u.select_carrot_ids.reshape(-1)[0])]
    return bool(u.agent.is_grasping(u.objs_carrot[name]).reshape(-1)[0].item())


# ---------------------------------------------------------------- 롤아웃 기록
def rollout(policy, w, eid, dev):
    obs, instr, _ = w.reset_with_episode("train", eid)
    sims, ctrls, tasks, rgbs = [], [], [], []
    toks, acts, lps, vals, succ, gl, cl, gn, rp = [], [], [], [], [], [], [], [], []
    for t in range(80):
        s, c, k = get_full_state(w)
        sims.append(s); ctrls.append(c); tasks.append(k); rgbs.append(obs[0].numpy())
        with torch.no_grad():
            v, a, lp = policy.get_action(dict(image=obs.to(dev), task_description=instr), deterministic=True)
        a = a.cpu()
        acts.append(w._process_action(a)[0].numpy().astype(np.float32))
        obs, r, _tr, info = w.step(a)
        toks.append(a[0].numpy()); lps.append(float(lp[0, 0])); vals.append(float(v[0, 0]))
        succ.append(bool(info["success"][0])); gl.append(bool(info["is_src_obj_grasped"][0]))
        cl.append(bool(info["consecutive_grasp"][0])); gn.append(grasp_now(w)); rp.append(float(r[0, 0]))
    s, c, k = get_full_state(w)
    sims.append(s); ctrls.append(c); tasks.append(k); rgbs.append(obs[0].numpy())
    return storage.EpisodeRecord(
        episode_id=eid, instruction=instr[0], sim_state=np.stack(sims), controller_state=np.stack(ctrls),
        task_state=np.stack(tasks), tokens=np.stack(toks).astype(np.int64), action=np.stack(acts),
        logprob_gen=np.array(lps, np.float32), value_gen=np.array(vals, np.float32), success=np.array(succ),
        grasp_latch=np.array(gl), consec_latch=np.array(cl), grasp_now=np.array(gn),
        reward_potential=np.array(rp, np.float32), rgb=np.stack(rgbs))


def main():
    ckpt, out = sys.argv[1], Path(sys.argv[2])
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    policy, unnorm = p2_eval.load_policy(ckpt, "bridge_orig", 0.6, merge_lora=True)
    policy.vla.generation_config.top_k = 0
    w = p2_eval.WinSimplerWrapper(ENV, unnorm, seed=0)
    dev = torch.device("cuda:0")
    eps = []
    t0 = time.time()
    for k in range(K):
        eid = int.from_bytes(hashlib.sha256(f"p4-storage-ep{k}".encode()).digest()[:8], "little") & ((1 << 62) - 1)
        torch.manual_seed(900 + k)
        e = rollout(policy, w, eid, dev)
        eps.append(e)
        log(f"[rollout {k}] Y={e.Y} grasp steps={int(e.grasp_now.sum())} {e.instruction}")
    res["rollout_min"] = round((time.time() - t0) / 60, 1)

    # ---------------- 2. 저장 정확성
    st = storage.RoundStore(out / "store", "W1_probe", 0, storage.StoragePolicy(full_rgb_rounds=(20,), keep_envs=2,
                                                                                 video_every=20, video_envs=2))
    t0 = time.time()
    man = st.write_round(20, eps)
    man_r, t1, instr, rgb = st.read_round(20, verify=True)
    exact = {k: bool(np.array_equal(t1[k], np.stack([getattr(e, k) for e in eps])) and
                     t1[k].dtype == np.stack([getattr(e, k) for e in eps]).dtype) for k in storage.TIER1_KEYS}
    rgb_exact = all(rgb[i].tobytes() == eps[i].rgb.tobytes() for i in range(K))
    size = st.round_bytes(20)
    tier1_bytes = (st.round_dir(20) / "tier1.npz").stat().st_size
    res["storage"] = dict(tier1_exact=exact, all_tier1_exact=all(exact.values()), rgb_bytes_exact=rgb_exact,
                          instructions_exact=instr == [e.instruction for e in eps], hashes_verified=True,
                          round_bytes=size, tier1_bytes=tier1_bytes, tier1_bytes_per_step=tier1_bytes / (K * 80),
                          rgb_bytes_per_frame=(size - tier1_bytes) / (K * 81),
                          raw_rgb_bytes_per_frame=480 * 640 * 3, sim_state_dim=int(eps[0].sim_state.shape[1]),
                          controller_state_dim=int(eps[0].controller_state.shape[1]), sec=round(time.time() - t0, 1))
    log(f"[storage] {json.dumps(res['storage'])}")

    # ---------------- 3. 재렌더링 (디스크에서 읽은 계층 1 상태 사용)
    diffs, eq = [], 0
    samples = [(i, t) for i in range(K) for t in range(0, 81, 10)]
    for i, t in samples:
        w.reset_with_episode("train", int(t1["episode_id"][i]))
        set_full_state(w, t1["sim_state"][i, t], t1["controller_state"][i, t], t1["task_state"][i, t])
        img = render_now(w)
        d = np.abs(img.astype(np.int16) - rgb[i][t].astype(np.int16))
        diffs.append(dict(ep=i, t=t, frac_pixels_diff=float((d > 0).any(-1).mean()), max_diff=int(d.max())))
        eq += int(d.max() == 0)
    res["rerender"] = dict(n=len(samples), identical_frames=eq, identical_rate=eq / len(samples),
                           mean_frac_pixels_diff=float(np.mean([x["frac_pixels_diff"] for x in diffs])),
                           max_frac_pixels_diff=float(np.max([x["frac_pixels_diff"] for x in diffs])),
                           max_pixel_diff=int(np.max([x["max_diff"] for x in diffs])), samples=diffs)
    log(f"[rerender] identical {eq}/{len(samples)}, mean frac {res['rerender']['mean_frac_pixels_diff']:.5f}, "
        f"max frac {res['rerender']['max_frac_pixels_diff']:.5f}")

    # ---------------- 4. 접촉 상태 복원
    contact = []
    for i in range(K):
        g = eps[i].grasp_now
        starts = [t for t in range(1, 80) if g[t - 1] and g[t] and t + 1 + L_REPLAY <= 80]
        if not starts:
            continue
        s0 = starts[0] + 1  # 집기가 이어진 뒤의 상태(행동 s0 직전)에서 복원
        for mode in ("full", "pose_only"):
            w.reset_with_episode("train", int(t1["episode_id"][i]))
            set_full_state(w, t1["sim_state"][i, s0], t1["controller_state"][i, s0], t1["task_state"][i, s0],
                           full=(mode == "full"))
            sim_d, gn_eq, info_eq, pix = [], 0, 0, []
            for t in range(s0, s0 + L_REPLAY):
                obs, _r, _tr, info = w.step(torch.as_tensor(t1["tokens"][i, t]).reshape(1, 7))
                s, _c, _k = get_full_state(w)
                sim_d.append(float(np.abs(s - t1["sim_state"][i, t + 1]).max()))
                gn_eq += int(grasp_now(w) == bool(t1["grasp_now"][i, t]))
                info_eq += int((bool(info["success"][0]), bool(info["is_src_obj_grasped"][0]), bool(info["consecutive_grasp"][0]))
                               == (bool(t1["success"][i, t]), bool(t1["grasp_latch"][i, t]), bool(t1["consec_latch"][i, t])))
                img = obs[0].numpy()
                pix.append(float((img != rgb[i][t + 1]).any(-1).mean()))
            rec = dict(ep=i, start=s0, mode=mode, grasp_at_start=bool(t1["grasp_now"][i, s0 - 1]),
                       max_state_diff=max(sim_d), state_diff_at_end=sim_d[-1], grasp_now_match=gn_eq / L_REPLAY,
                       info_match=info_eq / L_REPLAY, mean_frac_pixels_diff=float(np.mean(pix)),
                       Y_original=int(eps[i].Y))
            contact.append(rec)
            log(f"[contact {mode}] ep{i} start {s0} state maxdiff {rec['max_state_diff']:.2e} grasp match "
                f"{rec['grasp_now_match']:.2f} info match {rec['info_match']:.2f} pix {rec['mean_frac_pixels_diff']:.4f}")
    res["contact_restore"] = contact
    full = [c for c in contact if c["mode"] == "full"]
    res["contact_summary"] = dict(
        n_grasp_episodes=len(full),
        full_max_state_diff=max((c["max_state_diff"] for c in full), default=None),
        full_min_grasp_match=min((c["grasp_now_match"] for c in full), default=None),
        full_min_info_match=min((c["info_match"] for c in full), default=None),
        pose_only_mean_grasp_match=float(np.mean([c["grasp_now_match"] for c in contact if c["mode"] == "pose_only"]))
        if full else None)
    log(f"[contact summary] {res['contact_summary']}")
    w.env.close()
    (out / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log("SAVED")


if __name__ == "__main__":
    main()
