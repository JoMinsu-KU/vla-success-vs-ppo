"""P2: 공개 체크포인트 재현 평가 — Windows 네이티브, 환경 1개 (연구 명세 v0.9, 6.3절 P2).

원 RL4VLA 코드를 고치지 않고 그대로 가져다 쓴다:
  - 정책: simpler_env.policies.openvla.openvla_train.OpenVLAPolicy (predict_action_batch, 온도 0.6 샘플링,
    생성 설정 기본값 = top_k 50 포함. P2는 원 코드형 경로이므로 C4 통제를 적용하지 않는다)
  - 행동 변환·보상·리셋·스텝: simpler_env.env.simpler_wrapper.SimlerWrapper의 메서드
  - 성공 판정: 원 평가와 같은 info["success"]의 마지막 스텝 값 (info["episode"])

Windows 때문에 바꾸는 것 (모두 기록):
  1. sim_backend "gpu" -> "cpu", render_backend "cpu", num_envs 64 -> 1  (P1 기록 참고)
  2. attn_implementation "flash_attention_2" -> "sdpa"  (flash-attn은 Windows 미설치)
  3. prismatic.vla.datasets 대역 (win_compat.py, 추론에 쓰지 않는 학습 데이터 파이프라인)
  4. 초기 상태: 원 코드는 torch.randint(CUDA RNG)로 episode_id를 뽑아 재현 불가.
     여기서는 (seed, 에피소드 번호)에서 결정적으로 만든 큰 정수를 episode_id로 넘긴다
     (환경이 episode_id % ltt로 같은 균등 분포를 만든다).

사용 (예):
  python p2_eval.py --ckpt gen-robot/openvla-7b-rlvla-rl --unnorm-key bridge_orig \
      --env-id PutOnPlateInScene25Carrot-v1 --obj-set train --seed 0 --n-episodes 64 --out results.jsonl
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", r"D:\VLA_ReinforceLearning\hf_cache")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import win_compat  # noqa: E402

win_compat.install_prismatic_datasets_stub()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402,F401
from simpler_env.env.simpler_wrapper import SimlerWrapper  # noqa: E402

EPISODE_LEN = 80


# ---------------------------------------------------------------- 정책 (원 코드 + attention 교체)
def load_policy(ckpt: str, unnorm_key: str, temperature_eval: float):
    from prismatic.extern.hf import modeling_prismatic as mp
    from simpler_env.policies.openvla import openvla_train as ot

    cls = mp.OpenVLAForActionPredictionWithValueHead
    if not getattr(cls, "_p2_attn_patched", False):
        orig = cls.from_pretrained.__func__

        @classmethod
        def from_pretrained_sdpa(c, *a, **kw):
            if kw.get("attn_implementation") == "flash_attention_2":
                kw["attn_implementation"] = "sdpa"
            return orig(c, *a, **kw)

        cls.from_pretrained = from_pretrained_sdpa
        cls._p2_attn_patched = True

    args = SimpleNamespace(
        vla_path=ckpt, vla_unnorm_key=unnorm_key, vla_load_path="", vla_lora_rank=32,
        vla_lr=1e-4, vla_vhlr=3e-3, vla_optim_beta1=0.9, vla_optim_beta2=0.999,
        vla_temperature=1.0, vla_temperature_eval=temperature_eval,
    )
    policy = ot.OpenVLAPolicy(args, device_id=0)
    policy.prep_rollout()
    unnorm_state = policy.vla.get_action_stats(unnorm_key)
    return policy, unnorm_state


# ---------------------------------------------------------------- 환경 (원 래퍼 메서드 + Windows 설정)
class WinSimplerWrapper(SimlerWrapper):
    """SimlerWrapper와 같은 설정이되 sim/render backend와 환경 수만 Windows용으로 바꾼다."""

    def __init__(self, env_id: str, unnorm_state: dict, seed: int):
        self.args = SimpleNamespace(num_envs=1, episode_len=EPISODE_LEN, env_id=env_id, seed=seed)
        self.unnorm_state = unnorm_state
        self.num_envs = 1
        self.env = gym.make(
            id=env_id,
            num_envs=1,
            obs_mode="rgb+segmentation",
            control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
            sim_backend="cpu",          # 원: "gpu"
            render_backend="cpu",       # 원: 기본("gpu")
            sim_config={"sim_freq": 500, "control_freq": 5},
            max_episode_steps=EPISODE_LEN,
            sensor_configs={"shader_pack": "default"},
        )
        self.env.reset(seed=[seed * 1000])  # 원: [seed*1000 + i for i in range(64)]
        self.reward_old = torch.zeros(1, 1, dtype=torch.float32)
        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0

    def reset_with_episode(self, obj_set: str, episode_id: int):
        obs, info = self.env.reset(options={"obj_set": obj_set,
                                            "episode_id": torch.tensor([episode_id], dtype=torch.int64)})
        obs_image = obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        instruction = self.env.unwrapped.get_language_instruction()
        self.reward_old = torch.zeros(1, 1, dtype=torch.float32)
        return obs_image, instruction, info


def episode_id_for(seed: int, k: int) -> int:
    """(seed, 에피소드 번호) → 결정적인 63비트 정수. 환경이 % ltt로 균등 선택한다."""
    h = hashlib.sha256(f"p2-seed{seed}-ep{k}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 62) - 1)


def selected_ids(env) -> dict:
    u = env.unwrapped
    out = {}
    for name in ("select_carrot_ids", "select_plate_ids", "select_overlay_ids", "select_pos_ids",
                 "select_quat_ids", "select_extra_ids"):
        if hasattr(u, name):
            out[name.replace("select_", "")] = int(getattr(u, name).reshape(-1)[0].item())
    return out


P2_TASKS = [  # 6.3절 P2 핵심 과제 4개 (Carrot-v1 train = 학습 조건, test = 새 물체)
    ("PutOnPlateInScene25Carrot-v1", "train"),
    ("PutOnPlateInScene25Carrot-v1", "test"),
    ("PutOnPlateInScene25EEPose-v1", "test"),
    ("PutOnPlateInScene25PositionChangeTo-v1", "test"),
]


def done_episodes(out: Path, ckpt, env_id, obj_set, seed) -> set:
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if (r["ckpt"], r["env_id"], r["obj_set"], r["seed"]) == (ckpt, env_id, obj_set, seed):
                done.add(r["episode"])
    return done


def run_cell(policy, unnorm_state, a, env_id, obj_set, seed, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [k for k in range(a.n_episodes) if k not in done_episodes(out, a.ckpt, env_id, obj_set, seed)]
    if not todo:
        print(f"[skip] {env_id}/{obj_set} s{seed} already complete", flush=True)
        return
    wrapper = WinSimplerWrapper(env_id, unnorm_state, seed)
    dev = torch.device("cuda:0")
    for k in todo:
        eid = episode_id_for(seed, k)
        torch.manual_seed(seed * 100_000 + k)  # 행동 샘플링(온도 0.6) 재현용
        t0 = time.time()
        obs_img, instruction, _ = wrapper.reset_with_episode(obj_set, eid)
        frames = [obs_img[0].numpy()] if k < a.n_videos else None
        success_any = False
        t_policy = 0.0
        ep_info = None
        for t in range(EPISODE_LEN):
            tp = time.time()
            with torch.no_grad():
                _v, action, _lp = policy.get_action(
                    dict(image=obs_img.to(dev), task_description=instruction), deterministic=True)
            t_policy += time.time() - tp
            obs_img, _r, trunc, info = wrapper.step(action.cpu())
            success_any = success_any or bool(info["success"].reshape(-1)[0].item())
            if frames is not None:
                frames.append(obs_img[0].numpy())
            if "episode" in info:
                ep_info = info["episode"]
        assert ep_info is not None, "episode did not truncate at step 80"
        rec = {
            "ckpt": a.ckpt, "unnorm_key": a.unnorm_key, "env_id": env_id, "obj_set": obj_set,
            "seed": seed, "episode": k, "episode_id_raw": eid, **selected_ids(wrapper.env),
            "instruction": instruction[0],
            "success": bool(ep_info["success"][0]),                       # 원 평가 기준 (마지막 스텝)
            "grasped_last": bool(ep_info["is_src_obj_grasped"][0]),
            "consecutive_grasp_last": bool(ep_info["consecutive_grasp"][0]),
            "success_and_grasp_last": bool(ep_info["success"][0] and ep_info["is_src_obj_grasped"][0]),  # v0.9 Y 후보
            "success_any_step": success_any,
            "sec": round(time.time() - t0, 2), "sec_policy": round(t_policy, 2),
            "temperature_eval": a.temperature_eval,
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{Path(a.ckpt).name} {env_id}/{obj_set} s{seed} ep{k:02d}] success={rec['success']} "
              f"any={success_any} {rec['sec']}s (policy {rec['sec_policy']}s) :: {instruction[0]}", flush=True)
        if frames is not None and a.video_dir:
            from mani_skill.utils.visualization.misc import images_to_video
            vd = Path(a.video_dir)
            vd.mkdir(parents=True, exist_ok=True)
            images_to_video(frames, str(vd),
                            f"{Path(a.ckpt).name}_{env_id}_{obj_set}_s{seed}_ep{k}_succ{int(rec['success'])}",
                            fps=10, verbose=False)
    wrapper.env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--unnorm-key", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tasks", default="p2", help="'p2' = 핵심 4과제, 또는 'ENV_ID:split,ENV_ID:split'")
    p.add_argument("--n-episodes", type=int, default=64)
    p.add_argument("--temperature-eval", type=float, default=0.6)
    p.add_argument("--out", required=True)
    p.add_argument("--video-dir", default="")
    p.add_argument("--n-videos", type=int, default=0)
    a = p.parse_args()

    tasks = P2_TASKS if a.tasks == "p2" else [tuple(x.split(":")) for x in a.tasks.split(",")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pending = [(e, o, s) for s in a.seeds for (e, o) in tasks
               if len(done_episodes(out, a.ckpt, e, o, s)) < a.n_episodes]
    if not pending:
        print("nothing to do", flush=True)
        return
    t0 = time.time()
    policy, unnorm_state = load_policy(a.ckpt, a.unnorm_key, a.temperature_eval)
    print(f"[load] {a.ckpt} in {time.time() - t0:.1f}s, "
          f"cuda mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    for env_id, obj_set, seed in pending:
        run_cell(policy, unnorm_state, a, env_id, obj_set, seed, out)


if __name__ == "__main__":
    main()
