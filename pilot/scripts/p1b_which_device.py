"""P1-b (Windows): render_backend="cpu"일 때 SAPIEN이 실제로 어떤 Vulkan 장치로 렌더링하는지 확인한다.

640x480에서 400스텝을 돌리는 동안 다른 창에서 nvidia-smi를 찍어 4090 사용 여부를 본다.
"""
import os
import time

import sapien

d = sapien.Device("cpu")
print("[sapien.Device('cpu')]", d, [a for a in dir(d) if not a.startswith("_")], flush=True)
for attr in ("name", "pci_string", "is_cpu", "is_cuda", "can_render", "cuda_id"):
    if hasattr(d, attr):
        v = getattr(d, attr)
        print(f"  {attr} =", v() if callable(v) else v, flush=True)

import gymnasium as gym
import mani_skill.envs  # noqa: F401

env = gym.make("PickCube-v1", obs_mode="rgb", num_envs=1, sim_backend="cpu",
               render_backend="cpu", sensor_configs=dict(width=640, height=480))
env.reset(seed=0)
n = int(os.environ.get("N_STEPS", "400"))
t0 = time.time()
for _ in range(n):
    env.step(env.action_space.sample())
dt = time.time() - t0
print(f"[steps 640x480] {n} steps: {dt:.2f}s -> {n / dt:.1f} FPS", flush=True)
env.close()
