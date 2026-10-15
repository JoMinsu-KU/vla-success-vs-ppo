"""P1-b: ManiSkill 카메라 프레임 렌더링 시험 (연구 명세 v0.9, 6.0절).

사용: python p1b_render_test.py <라벨> <출력 폴더>
환경 변수 VK_ICD_FILENAMES로 Vulkan 드라이버(ICD)를 고른다.
"""
import sys
import time
from pathlib import Path

label, out_dir = sys.argv[1], Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

import numpy as np
import sapien

try:
    print("[device]", sapien.render.get_device_summary())
except Exception as e:  # 버전에 따라 없을 수 있음
    print("[device] summary unavailable:", repr(e))

import gymnasium as gym
import mani_skill.envs  # noqa: F401  환경 등록

t0 = time.time()
import os
render_backend = os.environ.get("P1B_RENDER_BACKEND", "gpu")  # lavapipe는 "cpu"
print("[render_backend]", render_backend)
res = os.environ.get("P1B_RES")  # 예: "640x480" (연구 입력 해상도)
extra = {}
if res:
    w, h = map(int, res.split("x"))
    extra["sensor_configs"] = dict(width=w, height=h)
print("[resolution]", res or "default")
env = gym.make("PickCube-v1", obs_mode="rgb", num_envs=1, sim_backend="cpu",
               render_backend=render_backend, **extra)
obs, _ = env.reset(seed=0)
print(f"[make+reset] {time.time() - t0:.2f}s")

cams = obs["sensor_data"]
for name, data in cams.items():
    rgb = data["rgb"]
    arr = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
    arr = arr[0] if arr.ndim == 4 else arr
    print(f"[frame] cam={name} shape={arr.shape} dtype={arr.dtype} "
          f"mean={arr.mean():.2f} std={arr.std():.2f} nonzero={np.count_nonzero(arr) / arr.size:.3f}")
    from PIL import Image
    Image.fromarray(arr.astype(np.uint8)).save(out_dir / f"{label}_{name}.png")

n = 50
t0 = time.time()
for _ in range(n):
    obs, *_ = env.step(env.action_space.sample())
dt = time.time() - t0
print(f"[steps] {n} steps with rgb obs: {dt:.2f}s -> {n / dt:.1f} FPS")
env.close()
print("render ok")
