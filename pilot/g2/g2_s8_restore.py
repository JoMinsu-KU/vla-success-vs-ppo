"""G2 S8: 상태 복원 반복성(E4-b), S_rec 타당성, CPU–GPU 연결 평가 (명세 3.8절 E4, G2 명세 4절 S8).

교란 연산 (이 파일에서 정의, 정책 평가 전에 고정. 크기는 G2 시험용 제안값):
  R-a 집기 실패 유도: 그리퍼를 연다(그리퍼 관절 qpos와 제어기 목표를 열림 상한으로) + 물체를 그 에피소드 시작(t=0)의 정착 자세에서
      x로 +3 cm 옮긴 위치로, 속도 0.
  R-b 운반 중 자세 교란: 팔 제어기의 EE 목표 자세를 (+3 cm, −3 cm, +2 cm) 평행 이동 + z축 0.15 rad 회전 (로봇이 다음 스텝부터 그쪽으로 움직임).
  R-c 접시 이동: 접시를 y로 +5 cm 옮김, 속도 0.
  대조: 교란 없음.
기준 상태(S_rec 원소의 시작점): W1(온도 0.6, K0) 학습 조건 롤아웃에서 집기가 2스텝 이어진 첫 시점의 다음 스텝 (파일럿 P4와 같은 정의).
  세 교란과 대조는 같은 기준 상태를 쓴다 (R-b의 "집은 뒤 고정 스텝 수" = 연속 집기 2스텝으로 해석).
복원 절차 (명세 v0.11 확정): ① episode_id로 장면 재구성 ② 전체 상태(물리 평탄화 상태 + 제어기 상태 + 과제 상태) 복원 ③ 교란 1회 ④ 회복 카운터 H=40스텝.
성공 = H스텝 뒤 Y (success ∧ 집기 래치).

부분:
  A. Linux CPU 반복성 (E4-b): 기준 상태 ≥7개에서 재구성+복원 → 기록된 행동 25개 재생을 (1) 두 번 (2) 다른 에피소드를 끼운 뒤 (3) 별도 프로세스에서
     → 상태·픽셀 비트 단위 비교
  B. S_rec 24개 타당성 (사전 고정 기준, docs/G2_prereg.md): 관절 한계, 관통 깊이 < 5 mm(물체–테이블, 물체–그리퍼), 복원 뒤 1스텝 물체 속도 < 0.5 m/s,
     래치 ↔ 물리 접촉 일치. 교란 없는 복원 대조의 성공률 vs 원 궤적 성공률 (15%p 넘으면 보고)
  C. CPU–GPU 연결 평가: W1·SFT·RL × S_rec 24 × (R-a, R-b, R-c, 대조) — CPU(환경 1개)와 GPU(환경 64개 배치) 각각.
     초기 전이(같은 출발 상태에서 같은 행동 1스텝 뒤 상태 차이), 성공 판정, 정책 간 상대 순서 비교.
사용: python g2_s8_restore.py <체크포인트 폴더> <out_dir> --part A|B|C|all
"""
import argparse
import copy
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_gpu as G  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ENV = "PutOnPlateInScene25Main-v3"
H = 40
L_REPLAY = 25
M_REC = 24
CKPTS = [("W1", "openvla-7b-rlvla-warmup", "bridge_orig"), ("SFT", "openvla-7b-rlvla-sft_16k", "sft"),
         ("RL", "openvla-7b-rlvla-rl", "bridge_orig")]
PERT = ["control", "R-a", "R-b", "R-c"]


# ---------------------------------------------------------------- 환경
def key_state(env, i):
    """초기 전이 비교용: 집을 물체 위치, 접시 위치 (m), 로봇 관절 위치 (rad)."""
    return dict(src=env.src_actor(i).pose.p[i].detach().cpu().numpy().astype(np.float64),
                tgt=env.tgt_actor(i).pose.p[i].detach().cpu().numpy().astype(np.float64),
                qpos=env.u.agent.robot.get_qpos()[i].detach().cpu().numpy().astype(np.float64))


def key_diff(a, b):
    return dict(src_m=float(np.abs(a["src"] - b["src"]).max()), tgt_m=float(np.abs(a["tgt"] - b["tgt"]).max()),
                qpos_rad=float(np.abs(a["qpos"] - b["qpos"]).max()))


class Env:
    """sim_backend cpu(환경 1개) 또는 gpu(환경 B개). 원 SimlerWrapper 설정·행동 변환."""

    def __init__(self, backend, unnorm, n=1, seed=0):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from simpler_env.env.simpler_wrapper import SimlerWrapper
        self.backend, self.n, self.unnorm_state = backend, n, unnorm
        with G.sim_nondeterministic_ok():
            self.env = gym.make(id=ENV, num_envs=n, obs_mode="rgb+segmentation", control_mode=G.CONTROL_MODE,
                                sim_backend=backend, sim_config={"sim_freq": 500, "control_freq": 5},
                                max_episode_steps=200, sensor_configs={"shader_pack": "default"})
            self.env.reset(seed=[seed * 1000 + i for i in range(n)])
        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self._proc = SimlerWrapper._process_action.__get__(self)

    @property
    def u(self):
        return self.env.unwrapped

    def set_unnorm(self, unnorm):
        self.unnorm_state = unnorm

    def reset(self, eids, reconfigure=True, seed=0):
        opts = {"obj_set": "train", "episode_id": torch.as_tensor(np.asarray(eids, np.int64), device=self.u.device)}
        if reconfigure:
            opts["reconfigure"] = True
        with G.sim_nondeterministic_ok():
            obs, _ = self.env.reset(seed=[seed * 1000 + i for i in range(self.n)], options=opts)
        return obs["sensor_data"][G.CAM]["rgb"].to(torch.uint8), list(self.u.get_language_instruction())

    def obs(self):
        with G.sim_nondeterministic_ok():
            o = self.u.get_obs()
        return o["sensor_data"][G.CAM]["rgb"].to(torch.uint8)

    def step(self, tokens):
        a = self._proc(torch.as_tensor(tokens).reshape(self.n, 7).to(self.u.device))
        with G.sim_nondeterministic_ok():
            obs, _r, _te, _tr, info = self.env.step(a)
        return obs["sensor_data"][G.CAM]["rgb"].to(torch.uint8), info

    def close(self):
        with G.sim_nondeterministic_ok():
            self.env.close()

    # ------------------------------------------------ 전체 상태 (환경 i 하나씩)
    def get_full(self, i=0):
        u = self.u
        sim = u.get_state()[i].detach().cpu().clone()
        ctrl = {}
        for uid, d in u.agent.controller.get_state().items():
            ctrl[uid] = {k: torch.as_tensor(v)[i].detach().cpu().clone() for k, v in d.items()}
        es = u.episode_stats
        task = dict(grasp=bool(es["is_src_obj_grasped"][i]), consec=bool(es["consecutive_grasp"][i]),
                    on_target=bool(es["src_on_target"][i]), counter=int(u.consecutive_grasp[i]), elapsed=int(u._elapsed_steps[i]))
        return dict(sim=sim, ctrl=ctrl, task=task)

    def set_full(self, states):
        """states: 길이 n 목록 (환경마다 하나). 재구성 직후 호출."""
        u = self.u
        dev = u.device
        with G.sim_nondeterministic_ok():
            u.set_state(torch.stack([s["sim"] for s in states]).to(dev))
            tmpl = u.agent.controller.get_state()
            new = {uid: {k: torch.stack([s["ctrl"][uid][k] for s in states]).to(torch.as_tensor(v).device)
                         for k, v in d.items()} for uid, d in tmpl.items()}
            u.agent.controller.set_state(new)
            es = u.episode_stats
            es["is_src_obj_grasped"] = torch.tensor([s["task"]["grasp"] for s in states], device=dev)
            es["consecutive_grasp"] = torch.tensor([s["task"]["consec"] for s in states], device=dev)
            es["src_on_target"] = torch.tensor([s["task"]["on_target"] for s in states], device=dev)
            u.consecutive_grasp = torch.tensor([s["task"]["counter"] for s in states], dtype=torch.int32, device=dev)
            u._elapsed_steps = torch.tensor([s["task"]["elapsed"] for s in states], dtype=u._elapsed_steps.dtype,
                                            device=u._elapsed_steps.device)
            if u.gpu_sim_enabled:
                u.scene._gpu_apply_all()
                u.scene.px.gpu_update_articulation_kinematics()
                u.scene._gpu_fetch_all()

    # ------------------------------------------------ 물체 핸들
    def src_actor(self, i):
        u = self.u
        return u.objs_carrot[u.carrot_names[int(u.select_carrot_ids.reshape(-1)[i])]]

    def tgt_actor(self, i):
        u = self.u
        return u.objs_plate[u.plate_names[int(u.select_plate_ids.reshape(-1)[i])]]

    def grasp_now(self):
        u = self.u
        return np.array([bool(u.agent.is_grasping(self.src_actor(i))[i]) for i in range(self.n)])

    # ------------------------------------------------ 교란 (환경별 종류 목록, t0 물체 자세 목록)
    def perturb(self, kinds, obj_pose0):
        from mani_skill.utils.structs.pose import Pose
        u = self.u
        dev = u.device
        with G.sim_nondeterministic_ok():
            for i, k in enumerate(kinds):
                if k == "R-a":
                    act = self.src_actor(i)
                    raw = act.pose.raw_pose.clone()
                    p0 = torch.as_tensor(obj_pose0[i], dtype=raw.dtype, device=raw.device)
                    raw[i] = p0
                    raw[i, 0] += 0.03
                    act.set_pose(Pose.create_from_pq(raw[:, :3], raw[:, 3:]))
                    lv = act.linear_velocity.clone()
                    av = act.angular_velocity.clone()
                    lv[i] = 0
                    av[i] = 0
                    act.set_linear_velocity(lv)
                    act.set_angular_velocity(av)
                    gc = u.agent.controller.controllers["gripper"]
                    up = float(np.max(np.asarray(gc.config.upper)))
                    qpos = u.agent.robot.get_qpos().clone()
                    jidx = [u.agent.robot.active_joints.index(j) for j in gc.joints]
                    qpos[i, jidx] = up
                    u.agent.robot.set_qpos(qpos)
                    qv = u.agent.robot.get_qvel().clone()
                    qv[i, jidx] = 0
                    u.agent.robot.set_qvel(qv)
                    st = gc.get_state()
                    if "target_qpos" in st:
                        tq = st["target_qpos"].clone()
                        tq[i] = up
                        gc.set_state(dict(target_qpos=tq))
                    # WidowX 그리퍼 제어기는 목표 상태를 두지 않으므로(use_target 꺼짐) 구동 목표를 직접 열림으로
                    tgt = gc.qpos.clone()
                    tgt[i] = up
                    gc.set_drive_targets(tgt)
                elif k == "R-b":
                    ac = u.agent.controller.controllers["arm"]
                    tp = ac._target_pose.raw_pose.clone()
                    tp[i, 0] += 0.03
                    tp[i, 1] -= 0.03
                    tp[i, 2] += 0.02
                    half = 0.075  # 0.15 rad / 2
                    qz = torch.tensor([np.cos(half), 0, 0, np.sin(half)], dtype=tp.dtype, device=tp.device)
                    w1, x1, y1, z1 = qz
                    w2, x2, y2, z2 = tp[i, 3:]
                    tp[i, 3:] = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
                    ac._target_pose = Pose.create_from_pq(tp[:, :3], tp[:, 3:])
                elif k == "R-c":
                    act = self.tgt_actor(i)
                    raw = act.pose.raw_pose.clone()
                    raw[i, 1] += 0.05
                    act.set_pose(Pose.create_from_pq(raw[:, :3], raw[:, 3:]))
                    lv = act.linear_velocity.clone()
                    lv[i] = 0
                    act.set_linear_velocity(lv)
            if u.gpu_sim_enabled:
                u.scene._gpu_apply_all()
                u.scene.px.gpu_update_articulation_kinematics()
                u.scene._gpu_fetch_all()


def policy_tokens(policy, rgb, instr):
    with torch.no_grad():
        _v, a, _lp = policy.get_action(dict(image=rgb.to("cuda:0"), task_description=list(instr)), deterministic=True)
    return a


def load_policy(ckpt_dir, sub, unnorm):
    p = G.build_policy(str(Path(ckpt_dir) / sub), unnorm, attn="flash_attention_2", seed=0)
    G.set_top_k(p, "K0")
    p.prep_rollout()
    return p


# ---------------------------------------------------------------- S_rec 만들기 (W1 CPU 롤아웃)
def build_srec(policy, env, out, log, max_eps=60, m_rec=M_REC):
    recs = []
    k = 0
    while len(recs) < m_rec and k < max_eps:
        eid = G.episode_id_for("g2s8-srec", k)
        k += 1
        torch.manual_seed(500 + k)
        rgb, ins = env.reset([eid])
        states, toks, gn, obj0 = [], [], [], env.src_actor(0).pose.raw_pose[0].detach().cpu().clone()
        succ = grasp = None
        for t in range(80):
            states.append(env.get_full(0))
            a = policy_tokens(policy, rgb, ins)
            toks.append(a[0].cpu().numpy())
            rgb, info = env.step(a)
            gn.append(bool(env.grasp_now()[0]))
            succ, grasp = bool(info["success"][0]), bool(info["is_src_obj_grasped"][0])
        starts = [t for t in range(1, 80) if gn[t - 1] and gn[t] and t + 1 + L_REPLAY <= 80]
        if starts:
            s0 = starts[0] + 1
            recs.append(dict(k=k - 1, eid=eid, instr=ins[0], s0=s0, base=states[s0], obj_pose0=obj0.numpy(),
                             tokens=np.stack(toks), orig_Y=succ and grasp))
            log(f"[srec] {len(recs)}/{m_rec} ep{k - 1} s0={s0} orig_Y={succ and grasp}")
    with (out / "srec.pkl").open("wb") as f:
        pickle.dump(recs, f)
    return recs


# ---------------------------------------------------------------- A. 반복성
def replay(env, rec):
    env.reset([rec["eid"]])
    env.set_full([rec["base"]])
    sims, imgs = [], []
    for t in range(rec["s0"], rec["s0"] + L_REPLAY):
        rgb, _ = env.step(torch.as_tensor(rec["tokens"][t]).reshape(1, 7))
        sims.append(env.u.get_state()[0].cpu().numpy())
        imgs.append(rgb[0].cpu().numpy())
    return np.stack(sims), np.stack(imgs)


def gpu_worker(ckpt_dir, out, name):
    """CPU–GPU 연결 평가의 GPU 쪽 (별도 프로세스). 입력 C_cpu_<name>.pkl, 출력 C_gpu_<name>.pkl."""
    recs = pickle.load(open(out / "srec.pkl", "rb"))
    cin = pickle.load(open(out / f"C_cpu_{name}.pkl", "rb"))
    items, first = cin["items"], cin["first"]
    pol = load_policy(ckpt_dir, cin["sub"], cin["unnorm"])
    us = {k: np.asarray(v) for k, v in pol.vla.get_action_stats(cin["unnorm"]).items()}
    gpu = Env("gpu", us, n=64)
    gpu_res, sg = {}, {}
    for b0 in range(0, len(items), 48):
        chunk = items[b0:b0 + 48]
        fill = chunk + [chunk[i % len(chunk)] for i in range(64 - len(chunk))]
        gpu.reset([recs[j]["eid"] for j, _ in fill])
        gpu.set_full([recs[j]["base"] for j, _ in fill])
        gpu.perturb([k for _, k in fill], [recs[j]["obj_pose0"] for j, _ in fill])
        gpu.step(torch.as_tensor(np.stack([first[x] for x in fill])))
        for i, x in enumerate(chunk):
            sg[x] = key_state(gpu, i)
        gpu.reset([recs[j]["eid"] for j, _ in fill])
        gpu.set_full([recs[j]["base"] for j, _ in fill])
        gpu.perturb([k for _, k in fill], [recs[j]["obj_pose0"] for j, _ in fill])
        torch.manual_seed(1900 + b0)
        rgb, ins = gpu.obs(), [recs[j]["instr"] for j, _ in fill]
        info = None
        for t in range(H):
            rgb, info = gpu.step(policy_tokens(pol, rgb, ins))
        Yg = (info["success"] & info["is_src_obj_grasped"]).cpu().numpy()
        for i, x in enumerate(chunk):
            gpu_res[x] = bool(Yg[i])
    gpu.close()
    pickle.dump(dict(gpu_res=gpu_res, sg=sg), open(out / f"C_gpu_{name}.pkl", "wb"))


def part_A_worker(srec_path, out_path, idxs):
    """별도 프로세스에서 같은 복원·재생 (작업자 배치 변경)."""
    recs = pickle.load(open(srec_path, "rb"))
    env = Env("cpu", BRIDGE)
    res = {i: replay(env, recs[i]) for i in idxs[::-1]}  # 순서도 뒤집는다
    env.close()
    pickle.dump(res, open(out_path, "wb"))


BRIDGE = None


def main():
    global BRIDGE
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--part", default="all")
    ap.add_argument("--worker", default="")
    ap.add_argument("--gpu-worker", default="")
    ap.add_argument("--m-rec", type=int, default=M_REC, help="스모크 시험용. 본 측정은 24")
    ap.add_argument("--ckpts", default="W1,SFT,RL")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    BRIDGE = json.loads((out / "bridge_unnorm.json").read_text()) if (out / "bridge_unnorm.json").exists() else None
    if a.gpu_worker:
        gpu_worker(a.ckpt_dir, out, a.gpu_worker)
        return
    if a.worker:
        srec_path, out_path, idxs = a.worker.split(":")
        part_A_worker(srec_path, out_path, [int(x) for x in idxs.split(",")])
        return
    resf = out / "g2_s8.json"
    res = json.loads(resf.read_text()) if resf.exists() else {}

    def log(m):
        print(m, flush=True)

    def save():
        resf.write_text(json.dumps(res, indent=2, default=float))

    t_all = time.time()
    w1 = load_policy(a.ckpt_dir, CKPTS[0][1], "bridge_orig")
    BRIDGE = {k: np.asarray(v).tolist() for k, v in w1.vla.get_action_stats("bridge_orig").items()}
    (out / "bridge_unnorm.json").write_text(json.dumps(BRIDGE))
    cpu = Env("cpu", BRIDGE)
    if (out / "srec.pkl").exists():
        recs = pickle.load(open(out / "srec.pkl", "rb"))
    else:
        t0 = time.time()
        recs = build_srec(w1, cpu, out, log, m_rec=a.m_rec)
        res["srec_build"] = dict(n=len(recs), sec=round(time.time() - t0, 1),
                                 episodes_tried=recs[-1]["k"] + 1 if recs else None)
        save()

    if a.part in ("A", "all"):
        idxs = list(range(min(8, len(recs))))
        A = []
        for i in idxs:
            s1, p1 = replay(cpu, recs[i])
            s2, p2 = replay(cpu, recs[i])
            other = recs[(i + 1) % len(recs)]
            cpu.reset([other["eid"]])
            for t in range(10):
                cpu.step(torch.as_tensor(other["tokens"][t]).reshape(1, 7))
            s3, p3 = replay(cpu, recs[i])
            A.append(dict(i=i, twice=bool(np.array_equal(s1, s2) and np.array_equal(p1, p2)),
                          after_other=bool(np.array_equal(s1, s3) and np.array_equal(p1, p3)), ref=(s1, p1)))
        wp = out / "A_worker.pkl"
        cmd = [sys.executable, __file__, a.ckpt_dir, str(out), "--worker", f"{out / 'srec.pkl'}:{wp}:{','.join(map(str, idxs))}"]
        subprocess.run(cmd, check=True)
        wr = pickle.load(open(wp, "rb"))
        for r in A:
            s, p = wr[r["i"]]
            r["other_process_reversed_order"] = bool(np.array_equal(r["ref"][0], s) and np.array_equal(r["ref"][1], p))
            del r["ref"]
        res["A_repeatability"] = dict(cases=A, all_identical=all(r["twice"] and r["after_other"] and r["other_process_reversed_order"] for r in A))
        log(f"[A] {res['A_repeatability']['all_identical']} {A}")
        save()

    if a.part in ("B", "all"):
        B = []
        for j, rec in enumerate(recs):
            cpu.reset([rec["eid"]])
            cpu.set_full([rec["base"]])
            u = cpu.u
            ql = u.agent.robot.get_qlimits()[0].cpu().numpy()
            q = u.agent.robot.get_qpos()[0].cpu().numpy()
            in_lim = bool(np.all(q >= ql[:, 0] - 1e-4) and np.all(q <= ql[:, 1] + 1e-4))
            # 관통: CPU PhysX 접촉점의 separation (음수 = 관통 깊이)
            src = cpu.src_actor(0)._objs[0]
            links = [lk._objs[0].entity for lk in u.agent.robot.get_links()]
            depth_tab = depth_grip = 0.0
            with G.sim_nondeterministic_ok():
                for c in u.scene.px.get_contacts():
                    ents = {c.bodies[0].entity, c.bodies[1].entity}
                    if src not in ents:
                        continue
                    other = (ents - {src}).pop() if len(ents) == 2 else None
                    mn = min([pt.separation for pt in c.points] + [0.0])
                    if other is not None and other in links:
                        depth_grip = max(depth_grip, -mn)
                    elif other is not None and "arena" in other.name:
                        depth_tab = max(depth_tab, -mn)
            latch = rec["base"]["task"]["grasp"]
            # 접촉력은 물리 스텝 뒤에만 계산되므로(복원 직후에는 0) 물리 집기 판정은 복원 뒤 1스텝(기록된 행동)에서 한다
            rgb, info1 = cpu.step(torch.as_tensor(rec["tokens"][rec["s0"]]).reshape(1, 7))
            phys = bool(cpu.grasp_now()[0])
            latch = latch and bool(info1["is_src_obj_grasped"][0])
            v1 = float(torch.linalg.norm(cpu.src_actor(0).linear_velocity[0]))
            ok = in_lim and depth_tab < 0.005 and depth_grip < 0.005 and v1 < 0.5 and (latch == phys)
            B.append(dict(j=j, joint_limits_ok=in_lim, penetration_table_m=depth_tab, penetration_gripper_m=depth_grip,
                          obj_speed_after_1step=v1, latch=latch, physical_grasp=phys, valid=ok))
        res["B_validity"] = dict(cases=B, n_valid=sum(b["valid"] for b in B), n=len(B))
        log(f"[B] valid {res['B_validity']['n_valid']}/{len(B)}")
        save()

    if a.part in ("C", "all"):
        valid = [b["j"] for b in res["B_validity"]["cases"] if b["valid"]] if "B_validity" in res else list(range(len(recs)))
        C = res.get("C_link", {})
        for name, sub, unnorm in [c for c in CKPTS if c[0] in a.ckpts.split(",")]:
            if name in C:
                continue
            pol = w1 if name == "W1" else load_policy(a.ckpt_dir, sub, unnorm)
            us = {k: np.asarray(v) for k, v in pol.vla.get_action_stats(unnorm).items()}
            cpu.set_unnorm(us)
            items = [(j, k) for j in range(len(recs)) for k in PERT]
            # CPU
            t0 = time.time()
            cpu_res, first = {}, {}
            for j, k in items:
                rec = recs[j]
                torch.manual_seed(900 + j)
                cpu.reset([rec["eid"]])
                cpu.set_full([rec["base"]])
                cpu.perturb([k], [rec["obj_pose0"]])
                rgb, ins = cpu.obs(), [rec["instr"]]
                info = None
                for t in range(H):
                    a_ = policy_tokens(pol, rgb, ins)
                    if t == 0:
                        first[(j, k)] = a_[0].cpu().numpy()
                    rgb, info = cpu.step(a_)
                cpu_res[(j, k)] = bool(info["success"][0]) and bool(info["is_src_obj_grasped"][0])
            t_cpu = time.time() - t0
            # 초기 전이 비교용 CPU 상태 (복원 + 교란 + CPU 첫 행동 1스텝)
            sc = {}
            for j, k in items:
                rec = recs[j]
                cpu.reset([rec["eid"]])
                cpu.set_full([rec["base"]])
                cpu.perturb([k], [rec["obj_pose0"]])
                cpu.step(torch.as_tensor(first[(j, k)]).reshape(1, 7))
                sc[(j, k)] = key_state(cpu, 0)
            # GPU: PhysX GPU 모드는 CPU PhysX를 쓴 프로세스에서 켤 수 없으므로 별도 프로세스에서 실행
            cin, cout = out / f"C_cpu_{name}.pkl", out / f"C_gpu_{name}.pkl"
            pickle.dump(dict(items=items, first=first, unnorm=unnorm, sub=sub), open(cin, "wb"))
            t0 = time.time()
            subprocess.run([sys.executable, __file__, a.ckpt_dir, str(out), "--gpu-worker", name], check=True)
            t_gpu = time.time() - t0
            g = pickle.load(open(cout, "rb"))
            gpu_res = g["gpu_res"]
            trans = [key_diff(g["sg"][x], sc[x]) for x in items]

            def rate(d, k, js):
                v = [d[(j, k)] for j in js]
                return round(100 * sum(v) / max(1, len(v)), 1)
            C[name] = dict(cpu={k: rate(cpu_res, k, valid) for k in PERT}, gpu={k: rate(gpu_res, k, valid) for k in PERT},
                           agreement=round(100 * np.mean([cpu_res[x] == gpu_res[x] for x in items if x[0] in valid]), 1),
                           initial_transition_cpu_vs_gpu={c: dict(median=float(np.median([t[c] for t in trans])), max=float(np.max([t[c] for t in trans])))
                                                             for c in ("src_m", "tgt_m", "qpos_rad")},
                           sec_cpu=round(t_cpu, 1), sec_gpu=round(t_gpu, 1), n_states=len(valid),
                           per_item=[dict(j=j, k=k, cpu=cpu_res[(j, k)], gpu=gpu_res[(j, k)]) for j, k in items])
            log(f"[C {name}] cpu {C[name]['cpu']} gpu {C[name]['gpu']} agree {C[name]['agreement']}% "
                f"trans {C[name]['initial_transition_cpu_vs_gpu']}")
            res["C_link"] = C
            save()
            if name != "W1":
                del pol
                torch.cuda.empty_cache()
        # 교란 없는 복원 대조 vs 원 궤적 (W1)
        ctrl = [C["W1"]["per_item"][i]["cpu"] for i in range(len(C["W1"]["per_item"])) if C["W1"]["per_item"][i]["k"] == "control"
                and C["W1"]["per_item"][i]["j"] in valid]
        orig = [recs[j]["orig_Y"] for j in valid]
        res["B_control_vs_original"] = dict(control_restored_W1_cpu=round(100 * np.mean(ctrl), 1),
                                            original_trajectory=round(100 * np.mean(orig), 1),
                                            report_flag=bool(abs(np.mean(ctrl) - np.mean(orig)) > 0.15))
        order = {be: sorted(list(C), key=lambda n: -C[n][be]["control"]) for be in ("cpu", "gpu")}
        res["C_relative_order_control"] = order
        log(f"[B ctrl vs orig] {res['B_control_vs_original']} order {order}")
    cpu.close()
    res["total_min"] = round((time.time() - t_all) / 60, 1) + res.get("total_min", 0)
    save()
    log("SAVED")


if __name__ == "__main__":
    main()
