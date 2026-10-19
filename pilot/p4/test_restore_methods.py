"""P4 — E4-b 공통 상태를 재현 가능하게 만드는 방법 비교 (접촉 상태, 모델 불필요).

test_restore_repeatability.py에서 상태 벡터 복원은 접촉 상태에서 반복 재현되지 않았다 (물리 엔진 내부 상태가 남음).
두 방법을 시험한다. 각 방법으로 같은 출발 상태를 세 번 만들고(사이에 다른 에피소드를 끼움) 같은 행동 25개를 재생해
비트 단위로 같은지, 그리고 원 궤적과 같은지 본다.
  (a) reconfigure: 장면을 새로 만든 뒤(reset options reconfigure=True) 상태 벡터 복원
  (b) replay: 리셋 뒤 원 행동 토큰을 출발 시점까지 재생해 상태를 다시 만듦 (S_rec = episode_id + 행동열)
사용: python test_restore_methods.py <store_dir> <out_json>
"""
import json
import sys
import time
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
from test_restore_repeatability import BRIDGE, L  # noqa: E402


def start_state(w, t1, i, s0, method):
    eid = int(t1["episode_id"][i])
    if method == "reconfigure":
        w.env.reset(options={"obj_set": "train", "episode_id": torch.tensor([eid]), "reconfigure": True})
        tsc.set_full_state(w, t1["sim_state"][i, s0], t1["controller_state"][i, s0], t1["task_state"][i, s0])
    elif method == "replay":
        w.reset_with_episode("train", eid)
        for t in range(s0):
            w.step(torch.as_tensor(t1["tokens"][i, t]).reshape(1, 7))
    else:
        raise ValueError(method)


def run(w, t1, i, s0, method):
    t0 = time.time()
    start_state(w, t1, i, s0, method)
    t_start = time.time() - t0
    states, imgs = [], []
    for t in range(s0, s0 + L):
        obs, *_ = w.step(torch.as_tensor(t1["tokens"][i, t]).reshape(1, 7))
        states.append(tsc.get_full_state(w)[0])
        imgs.append(obs[0].numpy())
    return np.stack(states), np.stack(imgs), t_start


def disturb(w, t1, j):
    w.reset_with_episode("train", int(t1["episode_id"][j]))
    for t in range(15):
        w.step(torch.as_tensor(t1["tokens"][j, t]).reshape(1, 7))


def main():
    store_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    st = storage.RoundStore(store_dir / "store", "W1_probe", 0)
    _m, t1, _i, rgb = st.read_round(20)
    w = p2_eval.WinSimplerWrapper("PutOnPlateInScene25Main-v3", BRIDGE, seed=0)
    K = len(t1["episode_id"])
    out = []
    for i in range(K):
        g = t1["grasp_now"][i]
        starts = [t for t in range(1, 80) if g[t - 1] and g[t] and t + 1 + L <= 80]
        if not starts:
            continue
        s0 = starts[0] + 1
        orig = t1["sim_state"][i, s0 + 1:s0 + 1 + L]
        rec = dict(ep=i, start=s0)
        for method in ("reconfigure", "replay"):
            runs = []
            for r in range(3):
                disturb(w, t1, (i + 1 + r) % K)
                runs.append(run(w, t1, i, s0, method))
            same_state = all(np.array_equal(runs[0][0], x[0]) for x in runs[1:])
            same_pix = all(np.array_equal(runs[0][1], x[1]) for x in runs[1:])
            rec[method] = dict(
                repeat_identical=bool(same_state and same_pix),
                repeat_max_state_diff=float(max(np.abs(runs[0][0] - x[0]).max() for x in runs[1:])),
                matches_original_state=bool(np.array_equal(runs[0][0], orig)),
                vs_original_max_state_diff=float(np.abs(runs[0][0] - orig).max()),
                matches_original_pixels=bool(np.array_equal(runs[0][1], rgb[i][s0 + 1:s0 + 1 + L])),
                sec_to_build_start=round(float(np.mean([x[2] for x in runs])), 2))
        out.append(rec)
        print(json.dumps(rec), flush=True)
    w.env.close()
    summ = {m: dict(all_repeat_identical=all(r[m]["repeat_identical"] for r in out),
                    all_match_original=all(r[m]["matches_original_state"] and r[m]["matches_original_pixels"] for r in out),
                    mean_sec_to_build_start=float(np.mean([r[m]["sec_to_build_start"] for r in out])))
            for m in ("reconfigure", "replay")}
    out_json.write_text(json.dumps(dict(summary=summ, episodes=out), indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
