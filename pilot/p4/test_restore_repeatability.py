"""P4 — 접촉 상태 복원의 재현성과 원 궤적과의 차이 분해 (모델 불필요, test_storage_contact.py가 저장한 자료 사용).

E4-b에 필요한 성질은 "복원 뒤 원 궤적과 같음"이 아니라 "같은 저장 상태를 복원하면 매번 같은 결과"다
(모든 정책이 같은 출발 상태에서 평가되도록). 그래서:
  1. 같은 저장 상태를 두 번(그리고 다른 에피소드를 거친 뒤 한 번 더) 복원해 같은 행동을 재생 → 비트 단위 비교
  2. 원 궤적과의 차이를 성분별(물체 위치·자세, 물체 속도, 로봇 관절 위치, 관절 속도)로 나눠 스텝별로 기록
사용: python test_restore_repeatability.py <store_dir(test_storage_contact 출력)> <out_json>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "learner"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import p2_eval  # noqa: E402
import storage  # noqa: E402
import test_storage_contact as tsc  # noqa: E402

L = 25
BRIDGE = {"q01": [-0.0298, -0.0418, -0.0262, -0.0807, -0.0942, -0.1876, 0.0],
          "q99": [0.0293, 0.0404, 0.0396, 0.0779, 0.0861, 0.1910, 1.0],
          "mask": [True, True, True, True, True, True, False]}


def layout(w):
    """평탄화 상태 벡터에서 성분별 인덱스 (actors: pose 7 + vel 6, articulations: 루트 13 + qpos + qvel)."""
    sd = w.env.unwrapped.get_state_dict()
    u = w.env.unwrapped
    name = u.carrot_names[int(u.select_carrot_ids.reshape(-1)[0])]
    idx, pos = {}, 0
    for k, v in sd["actors"].items():
        n = v.shape[-1]
        if k == u.objs_carrot[name].name:
            idx["object_pose"] = list(range(pos, pos + 7))
            idx["object_vel"] = list(range(pos + 7, pos + 13))
        pos += n
    for k, v in sd["articulations"].items():
        n = v.shape[-1]
        nq = (n - 13) // 2
        idx["robot_qpos"] = list(range(pos + 13, pos + 13 + nq))
        idx["robot_qvel"] = list(range(pos + 13 + nq, pos + n))
        pos += n
    return idx


def replay(w, t1, i, s0, eid):
    w.reset_with_episode("train", eid)
    tsc.set_full_state(w, t1["sim_state"][i, s0], t1["controller_state"][i, s0], t1["task_state"][i, s0])
    states, imgs, gns = [], [], []
    for t in range(s0, s0 + L):
        obs, *_ = w.step(torch.as_tensor(t1["tokens"][i, t]).reshape(1, 7))
        states.append(tsc.get_full_state(w)[0])
        imgs.append(obs[0].numpy())
        gns.append(tsc.grasp_now(w))
    return np.stack(states), np.stack(imgs), gns


def main():
    store_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    st = storage.RoundStore(store_dir / "store", "W1_probe", 0)
    _man, t1, _instr, _rgb = st.read_round(20)
    w = p2_eval.WinSimplerWrapper("PutOnPlateInScene25Main-v3", BRIDGE, seed=0)
    res = []
    K = len(t1["episode_id"])
    for i in range(K):
        g = t1["grasp_now"][i]
        starts = [t for t in range(1, 80) if g[t - 1] and g[t] and t + 1 + L <= 80]
        if not starts:
            continue
        s0, eid = starts[0] + 1, int(t1["episode_id"][i])
        a_states, a_imgs, a_g = replay(w, t1, i, s0, eid)
        b_states, b_imgs, b_g = replay(w, t1, i, s0, eid)
        # 다른 에피소드를 한 번 돌린 뒤 다시 (환경 내부 상태가 남아 영향을 주는지)
        other = (i + 1) % K
        w.reset_with_episode("train", int(t1["episode_id"][other]))
        for t in range(10):
            w.step(torch.as_tensor(t1["tokens"][other, t]).reshape(1, 7))
        c_states, c_imgs, c_g = replay(w, t1, i, s0, eid)
        lay = layout(w)
        orig = t1["sim_state"][i, s0 + 1:s0 + 1 + L]
        by_comp = {k: [float(np.abs(a_states[t, ix] - orig[t, ix]).max()) for t in range(L)] for k, ix in lay.items()}
        rec = dict(ep=i, start=s0,
                   repeat_identical_state=bool(np.array_equal(a_states, b_states)),
                   repeat_identical_pixels=bool(np.array_equal(a_imgs, b_imgs)),
                   repeat_after_other_episode_identical=bool(np.array_equal(a_states, c_states) and np.array_equal(a_imgs, c_imgs)),
                   repeat_grasp_identical=(a_g == b_g == c_g),
                   vs_original_by_component_step1={k: v[0] for k, v in by_comp.items()},
                   vs_original_by_component_step5={k: v[4] for k, v in by_comp.items()},
                   vs_original_by_component_step25={k: v[-1] for k, v in by_comp.items()})
        res.append(rec)
        print(json.dumps(rec), flush=True)
    w.env.close()
    summary = dict(n=len(res), all_repeat_identical=all(r["repeat_identical_state"] and r["repeat_identical_pixels"]
                                                       and r["repeat_after_other_episode_identical"] for r in res))
    out_json.write_text(json.dumps(dict(summary=summary, episodes=res), indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
