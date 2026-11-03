"""G2 S2: H200에서 ManiSkill GPU 시뮬레이션 + 렌더링이 되는지 확인 (go/no-go 1).

원 RL4VLA 래퍼(simpler_wrapper.py)와 같은 설정: sim_backend "gpu", obs_mode "rgb+segmentation",
같은 제어 모드·sim/control 주파수·shader pack, 기본 render backend.
  A. SAPIEN이 보는 렌더 장치
  B. 학습 환경 PutOnPlateInScene25Main-v3, num_envs = 1, 8, 64: 생성·리셋·80스텝 시간, 프레임 통계, GPU 메모리
  C. 평가 조건 16개 (num_envs 64): 생성·리셋·5스텝, 첫 프레임 저장
사용: python g2_01_render.py <출력 폴더> [--gpu-index 0]
결과: <출력 폴더>/g2_s2_render.json, PNG
"""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
gpu_index = int(sys.argv[sys.argv.index("--gpu-index") + 1]) if "--gpu-index" in sys.argv else 0

import gymnasium as gym  # noqa: E402
import sapien  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import mani_skill.envs  # noqa: E402,F401  (RL4VLA 포크 환경 등록)

CONDITIONS = [
    ("PutOnPlateInScene25Carrot-v1", "train", "ID"),
    ("PutOnPlateInScene25VisionImage-v1", "test", "vision"),
    ("PutOnPlateInScene25VisionTexture03-v1", "test", "vision"),
    ("PutOnPlateInScene25VisionTexture05-v1", "test", "vision"),
    ("PutOnPlateInScene25VisionWhole03-v1", "test", "vision"),
    ("PutOnPlateInScene25VisionWhole05-v1", "test", "vision"),
    ("PutOnPlateInScene25Carrot-v1", "test", "semantic"),
    ("PutOnPlateInScene25Plate-v1", "test", "semantic"),
    ("PutOnPlateInScene25Instruct-v1", "test", "semantic"),
    ("PutOnPlateInScene25MultiCarrot-v1", "test", "semantic"),
    ("PutOnPlateInScene25MultiPlate-v1", "test", "semantic"),
    ("PutOnPlateInScene25Position-v1", "test", "execution"),
    ("PutOnPlateInScene25EEPose-v1", "test", "execution"),
    ("PutOnPlateInScene25PositionChangeTo-v1", "test", "execution"),
    ("PutOnPlateInScene25MultiCarrot-v1", "train", "train-variant"),
    ("PutOnPlateInScene25MultiPlate-v1", "train", "train-variant"),
]
CAM = "3rd_view_camera"
res = {"gpu_index": gpu_index}


def gpu_mem_used_mib():
    try:
        s = subprocess.run(["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        return int(s.splitlines()[0])
    except Exception:
        return None


def make(env_id, n):
    # simpler_wrapper.py (RL4VLA de46e01)와 같은 설정
    return gym.make(id=env_id, num_envs=n, obs_mode="rgb+segmentation",
                    control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
                    sim_backend="gpu", sim_config={"sim_freq": 500, "control_freq": 5},
                    max_episode_steps=80, sensor_configs={"shader_pack": "default"})


def frame_stats(obs):
    rgb = obs["sensor_data"][CAM]["rgb"]
    a = rgb.detach().cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
    return a, dict(shape=list(a.shape), mean=float(a.mean()), std=float(a.std()),
                   nonzero=float(np.count_nonzero(a) / a.size))


# A. 렌더 장치
try:
    res["sapien_device_summary"] = str(sapien.render.get_device_summary())
except Exception as e:
    res["sapien_device_summary"] = f"unavailable: {e!r}"
res["torch_cuda"] = dict(available=torch.cuda.is_available(), n=torch.cuda.device_count(),
                         name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("[A]", res["sapien_device_summary"], res["torch_cuda"], flush=True)

# B. 학습 환경, 환경 수별
res["train_env"] = []
for n in (1, 8, 64):
    rec = dict(num_envs=n, ok=False)
    env = None
    try:
        m0 = gpu_mem_used_mib()
        t0 = time.time()
        env = make("PutOnPlateInScene25Main-v3", n)
        obs, info = env.reset(seed=[i for i in range(n)])
        rec["make_reset_s"] = round(time.time() - t0, 2)
        arr, rec["frame"] = frame_stats(obs)
        Image.fromarray(arr[0].astype(np.uint8)).save(out / f"train_env_n{n}_first.png")
        act = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        t0 = time.time()
        for _ in range(80):
            obs, rew, term, trunc, info = env.step(act)
        torch.cuda.synchronize()
        dt = time.time() - t0
        rec["steps80_s"] = round(dt, 2)
        rec["env_steps_per_s"] = round(80 * n / dt, 1)
        t0 = time.time()
        env.reset(options={"reconfigure": True})
        rec["reconfigure_reset_s"] = round(time.time() - t0, 2)
        rec["gpu_mem_delta_mib"] = (gpu_mem_used_mib() or 0) - (m0 or 0)
        rec["info_keys"] = sorted(info.keys())
        rec["ok"] = rec["frame"]["std"] > 1.0
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    finally:
        if env is not None:
            env.close()
        torch.cuda.empty_cache()
    res["train_env"].append(rec)
    print("[B]", json.dumps({k: v for k, v in rec.items() if k != "traceback"}), flush=True)

# C. 평가 조건 16개, 환경 64개
res["conditions"] = []
for idx, (env_id, split, axis) in enumerate(CONDITIONS, 1):
    rec = dict(idx=idx, env_id=env_id, split=split, axis=axis, ok=False)
    env = None
    try:
        t0 = time.time()
        env = make(env_id, 64)
        obs, info = env.reset(seed=list(range(64)), options={"obj_set": split})
        act = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for _ in range(5):
            obs, *_ = env.step(act)
        rec["s"] = round(time.time() - t0, 2)
        arr, rec["frame"] = frame_stats(obs)
        Image.fromarray(arr[0].astype(np.uint8)).save(out / f"cond{idx:02d}_{env_id.replace('PutOnPlateInScene25', '')}_{split}.png")
        rec["instruction"] = str(env.unwrapped.get_language_instruction()[0])[:80] \
            if hasattr(env.unwrapped, "get_language_instruction") else None
        rec["ok"] = rec["frame"]["std"] > 1.0
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-2000:]
    finally:
        if env is not None:
            env.close()
        torch.cuda.empty_cache()
    res["conditions"].append(rec)
    print("[C]", json.dumps({k: v for k, v in rec.items() if k != "traceback"}), flush=True)

res["verdict"] = dict(train_env_64_ok=bool(res["train_env"][-1]["ok"]),
                      conditions_ok=sum(r["ok"] for r in res["conditions"]))
(out / "g2_s2_render.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print("[verdict]", res["verdict"], "saved", out / "g2_s2_render.json", flush=True)
