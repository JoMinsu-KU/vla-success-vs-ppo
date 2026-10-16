"""P4: 상태 복원 시험 (E4-b, 연구 명세 v0.9 3.8절·P4 표 '상태 복원'·'시뮬레이터 재현성'). 모델 불필요.

전체 상태 = 물리 상태(env.get_state_dict: 모든 actor·articulation의 pose·속도·관절) + 로봇 제어기 상태
(agent.get_state: EE 목표 pose 등) + 과제 상태(episode_stats 래치, 연속 집기 카운터, 경과 스텝).

시험:
  0. 기준 재현성: 같은 에피소드·같은 행동열을 처음부터 두 번 → 차이 (CPU 시뮬레이션 결정성)
  1. 전체 상태 복원: 15스텝 뒤 저장 → 25스텝 진행(T1) → 복원 → 같은 25스텝 재생(T2) → T1과 비교
  2. pose만 복원(속도·제어기·과제 상태 제외) → 같은 재생 → 차이가 커지는지
  3. 과제 상태(래치·카운터·경과 스텝)가 복원 전후 같은지
  4. 시간: 70스텝 시점 상태를 복원한 뒤 별도 카운터로 H = 40스텝을 끝까지 진행할 수 있는지
     (max_episode_steps를 넉넉히 둔 환경에서 경과 스텝은 과제 상태로 보존)
사용: python test_state_restore.py <out_json>
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import p2_eval  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

out_json = Path(sys.argv[1])
ENV = "PutOnPlateInScene25Main-v3"
BRIDGE = {"q01": [-0.0298, -0.0418, -0.0262, -0.0807, -0.0942, -0.1876, 0.0],
          "q99": [0.0293, 0.0404, 0.0396, 0.0779, 0.0861, 0.1910, 1.0],
          "mask": [True, True, True, True, True, True, False]}  # 형식만 맞춘 대역 (행동 크기 범위용)


class Wrapper(p2_eval.WinSimplerWrapper):
    def __init__(self, max_steps=80):
        self.args = None
        self.unnorm_state = BRIDGE
        self.num_envs = 1
        self.env = gym.make(ENV, num_envs=1, obs_mode="rgb+segmentation",
                            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                            sim_backend="cpu", render_backend="cpu", sim_config={"sim_freq": 500, "control_freq": 5},
                            max_episode_steps=max_steps, sensor_configs={"shader_pack": "default"})
        self.env.reset(seed=[0])
        self.reward_old = torch.zeros(1, 1)
        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0


def actions(seed, n):
    """그리퍼 열고 닫기가 섞이도록 무작위 행동 토큰 (물체 접촉이 생기게 z를 낮추는 쪽으로 약간 치우침)."""
    g = np.random.default_rng(seed)
    toks = g.integers(32000 - 256 + 40, 32000 - 40, size=(n, 7))
    toks[:, 2] = g.integers(32000 - 120, 32000 - 60, size=n)  # z 성분
    toks[:, 6] = np.where(np.arange(n) % 10 < 5, 31800, 31900)  # 그리퍼 열기/닫기 반복
    return torch.as_tensor(toks)


def snapshot(w):
    u = w.env.unwrapped
    return dict(sim=copy.deepcopy(u.get_state_dict()), agent=copy.deepcopy(u.agent.get_state()),
                episode_stats={k: v.clone() for k, v in u.episode_stats.items()},
                consecutive_grasp=u.consecutive_grasp.clone(), elapsed=u._elapsed_steps.clone(),
                reward_old=w.reward_old.clone())


def restore(w, s, full=True):
    u = w.env.unwrapped
    if full:
        u.set_state_dict(s["sim"])
        u.agent.set_state(s["agent"])
        u.episode_stats = {k: v.clone() for k, v in s["episode_stats"].items()}
        u.consecutive_grasp = s["consecutive_grasp"].clone()
        u._elapsed_steps = s["elapsed"].clone()
        w.reward_old = s["reward_old"].clone()
    else:  # pose만: actor pose와 관절 위치만, 속도 0, 제어기·과제 상태는 복원하지 않음
        sim = copy.deepcopy(s["sim"])
        for v in sim["actors"].values():
            v[..., 7:13] = 0  # [pose 7, 선속도 3, 각속도 3]
        for v in sim["articulations"].values():
            v[..., 7:13] = 0  # 루트 속도
            if v.shape[-1] > 13:  # [루트 pose 7, 루트 속도 6, qpos n, qvel n]
                n = (v.shape[-1] - 13) // 2
                v[..., 13 + n:] = 0
        u.set_state_dict(sim)


def run(w, acts):
    u = w.env.unwrapped
    traj = []
    for a in acts:
        obs, _r, _t, info = w.step(a.reshape(1, 7))
        name = u.carrot_names[int(u.select_carrot_ids[0])]
        traj.append(dict(qpos=u.agent.robot.get_qpos()[0].clone(), obj=u.objs_carrot[name].pose.raw_pose[0].clone(),
                         img=obs[0].clone(), success=bool(info["success"][0]), grasp=bool(info["is_src_obj_grasped"][0]),
                         consec=bool(info["consecutive_grasp"][0]), elapsed=int(u._elapsed_steps[0])))
    return traj


def diff(t1, t2):
    return dict(max_qpos=float(max((a["qpos"] - b["qpos"]).abs().max() for a, b in zip(t1, t2))),
                max_obj_pose=float(max((a["obj"] - b["obj"]).abs().max() for a, b in zip(t1, t2))),
                max_pixel=int(max((a["img"].int() - b["img"].int()).abs().max() for a, b in zip(t1, t2))),
                frac_pixels_diff=float(np.mean([(a["img"] != b["img"]).float().mean().item() for a, b in zip(t1, t2)])),
                info_identical=all((a["success"], a["grasp"], a["consec"], a["elapsed"]) ==
                                   (b["success"], b["grasp"], b["consec"], b["elapsed"]) for a, b in zip(t1, t2)))


res = {}
EID = p2_eval.episode_id_for(11, 0)
A1, A2 = actions(1, 15), actions(2, 25)

# 0. 기준 재현성
w = Wrapper()
w.reset_with_episode("train", EID)
base1 = run(w, torch.cat([A1, A2]))
w.reset_with_episode("train", EID)
base2 = run(w, torch.cat([A1, A2]))
res["0_rerun_from_reset"] = diff(base1, base2)

# 1·3. 전체 복원
w.reset_with_episode("train", EID)
run(w, A1)
s = snapshot(w)
T1 = run(w, A2)
restore(w, s, full=True)
after = snapshot(w)
res["3_task_state_restored"] = dict(
    latches_equal=all(torch.equal(s["episode_stats"][k], after["episode_stats"][k]) for k in s["episode_stats"]),
    counter_equal=bool(torch.equal(s["consecutive_grasp"], after["consecutive_grasp"])),
    elapsed_equal=bool(torch.equal(s["elapsed"], after["elapsed"])), elapsed=int(s["elapsed"][0]))
T2 = run(w, A2)
res["1_full_restore_replay"] = diff(T1, T2)

# 2. pose만 복원
restore(w, s, full=False)
T3 = run(w, A2)
res["2_pose_only_restore_replay"] = diff(T1, T3)
res["episode_grasped_in_T1"] = any(x["grasp"] for x in T1)
w.env.close()

# 4. 70스텝 복원 + 별도 카운터 H=40
w = Wrapper(max_steps=400)
w.reset_with_episode("train", EID)
run(w, actions(3, 70))
s70 = snapshot(w)
restore(w, s70, full=True)
t = run(w, actions(4, 40))
res["4_restore_at_70_then_H40"] = dict(steps_run=len(t), elapsed_first=t[0]["elapsed"], elapsed_last=t[-1]["elapsed"],
                                      completed=len(t) == 40)
w.env.close()

for k, v in res.items():
    print(k, v, flush=True)
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
