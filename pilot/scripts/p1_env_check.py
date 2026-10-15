"""P1: 원 논문 평가 조건 16개의 환경 생성·렌더링 확인 (연구 명세 v0.9, 6.3절 P1 절차 3).

Windows 네이티브(rlvla_win)용. 원 SimplerEnv 래퍼(simpler_wrapper.py)와 같은 설정을 쓰되,
Windows에서 되는 경로로만 바꾼다:
  - sim_backend "gpu" -> "cpu" (Windows는 GPU 병렬 시뮬레이션 미지원)
  - render_backend "cpu" (RTX 4090 Vulkan으로 렌더링, 결과는 CPU 메모리로 읽음. "gpu"는 CUDA 연동에서 충돌)
  - num_envs 64 -> 1

사용: python p1_env_check.py <출력 폴더> [스텝 수]
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

out_dir = Path(sys.argv[1])
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
out_dir.mkdir(parents=True, exist_ok=True)

import gymnasium as gym
import mani_skill.envs  # noqa: F401  환경 등록 (RL4VLA 포크)
import torch

# 1.2절 표와 README "Task definition" 순서. (환경 ID, split, 축, 설명)
CONDITIONS = [
    ("PutOnPlateInScene25Carrot-v1", "train", "ID", "similar to training setting"),
    ("PutOnPlateInScene25VisionImage-v1", "test", "vision", "unseen table"),
    ("PutOnPlateInScene25VisionTexture03-v1", "test", "vision", "dynamic texture (weak)"),
    ("PutOnPlateInScene25VisionTexture05-v1", "test", "vision", "dynamic texture (strong)"),
    ("PutOnPlateInScene25VisionWhole03-v1", "test", "vision", "dynamic noise (weak)"),
    ("PutOnPlateInScene25VisionWhole05-v1", "test", "vision", "dynamic noise (strong)"),
    ("PutOnPlateInScene25Carrot-v1", "test", "semantic", "unseen objects"),
    ("PutOnPlateInScene25Plate-v1", "test", "semantic", "unseen receptacles"),
    ("PutOnPlateInScene25Instruct-v1", "test", "semantic", "unseen instructions"),
    ("PutOnPlateInScene25MultiCarrot-v1", "test", "semantic", "multi-object (both unseen)"),
    ("PutOnPlateInScene25MultiPlate-v1", "test", "semantic", "multi-receptacle (both unseen)"),
    ("PutOnPlateInScene25Position-v1", "test", "execution", "unseen position"),
    ("PutOnPlateInScene25EEPose-v1", "test", "execution", "unseen robot init pose"),
    ("PutOnPlateInScene25PositionChangeTo-v1", "test", "execution", "mid-episode object reposition"),
    ("PutOnPlateInScene25MultiCarrot-v1", "train", "train-variant", "multi-object (both seen)"),
    ("PutOnPlateInScene25MultiPlate-v1", "train", "train-variant", "distractive receptacle"),
]

CAMERA = "3rd_view_camera"
results = []

for idx, (env_id, split, axis, desc) in enumerate(CONDITIONS, start=1):
    tag = f"{idx:02d}_{env_id.replace('PutOnPlateInScene25', '')}_{split}"
    rec = {"idx": idx, "env_id": env_id, "split": split, "axis": axis, "desc": desc, "ok": False}
    print(f"===== [{idx:02d}/16] {env_id} ({split}) - {desc}", flush=True)
    env = None
    try:
        t0 = time.time()
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="rgb+segmentation",
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            sim_backend="cpu",
            render_backend="cpu",
            sim_config={"sim_freq": 500, "control_freq": 5},
            max_episode_steps=80,
            sensor_configs={"shader_pack": "default"},
        )
        env.reset(seed=[0])
        rec["make_s"] = round(time.time() - t0, 2)

        t0 = time.time()
        obs, info = env.reset(options={"obj_set": split})
        rec["reset_s"] = round(time.time() - t0, 2)
        rec["instruction"] = env.unwrapped.get_language_instruction()

        rgb = obs["sensor_data"][CAMERA]["rgb"]
        arr = rgb.to(torch.uint8).cpu().numpy()[0]
        rec["frame_shape"] = list(arr.shape)
        rec["frame_mean"] = round(float(arr.mean()), 2)
        rec["frame_std"] = round(float(arr.std()), 2)
        Image.fromarray(arr).save(out_dir / f"{tag}.png")

        # 무동작(0) 행동으로 몇 스텝 진행해 step 경로와 info 키를 확인한다.
        action = torch.zeros(env.action_space.shape, dtype=torch.float32)
        t0 = time.time()
        for _ in range(n_steps):
            obs, _r, _term, trunc, info = env.step(action)
        dt = time.time() - t0
        rec["step_fps"] = round(n_steps / dt, 1) if dt > 0 else None
        rec["info_keys"] = sorted(k for k in info.keys() if not k.startswith("_"))
        for k in ("success", "is_src_obj_grasped", "consecutive_grasp"):
            if k in info:
                rec[k] = bool(info[k].reshape(-1)[0].item())
        rec["ok"] = True
        print(f"  ok  make {rec['make_s']}s  reset {rec['reset_s']}s  step {rec['step_fps']} FPS  "
              f"frame {rec['frame_shape']} mean {rec['frame_mean']}  instr: {rec['instruction']}", flush=True)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc(limit=6)
        print("  FAIL", rec["error"], flush=True)
        print(rec["traceback"], flush=True)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    results.append(rec)

(out_dir / "p1_env_check.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

# 16개 프레임을 4x4 격자 한 장으로 모은다 (축소).
thumbs = []
for rec in results:
    p = out_dir / f"{rec['idx']:02d}_{rec['env_id'].replace('PutOnPlateInScene25', '')}_{rec['split']}.png"
    img = Image.open(p).resize((320, 240)) if p.exists() else Image.new("RGB", (320, 240), (40, 40, 40))
    thumbs.append(img)
grid = Image.new("RGB", (320 * 4, 240 * 4))
for i, img in enumerate(thumbs):
    grid.paste(img, ((i % 4) * 320, (i // 4) * 240))
grid.save(out_dir / "p1_grid_4x4.jpg", quality=85)

n_ok = sum(r["ok"] for r in results)
print(f"SUMMARY: {n_ok}/16 conditions created, reset, rendered and stepped", flush=True)
