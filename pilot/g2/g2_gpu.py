"""G2 Linux GPU 경로 공통 모듈 (S4~S9). Windows 파일럿의 p2_eval·batched_eval·win_compat를 대체한다.

원 RL4VLA 코드를 그대로 쓰는 부분:
  - 정책: simpler_env.policies.openvla.openvla_train.OpenVLAPolicy (get_action, evaluate_actions, _preprocess_obs)
  - 행동 변환: simpler_env.env.simpler_wrapper.SimlerWrapper._process_action
  - 환경 설정: SimlerWrapper와 같은 gym.make 인자 (GPU 시뮬레이션, rgb+segmentation, 같은 제어 모드·주파수·shader)
바꾸는 부분 (통제 경로, 명세 3.8절·C4):
  - 초기 상태: GPU 난수가 아니라 고정 episode_id (SHA-256 유도)를 env.reset(options["episode_id"])로 넘긴다
  - 배치(환경 64개)마다 장면 재구성(reconfigure) → 모든 에피소드가 새로 만든 장면에서 시작
  - top_k: K0(0) 또는 K50, top_p 1.0을 명시
  - attention 경로: 기본 flash_attention_2(원 코드), 인자로 sdpa 선택 가능. 실제 구현을 numerics_report로 기록
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
sys.path.insert(0, str(PILOT / "learner"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

T_EP = 80
CAM = "3rd_view_camera"
CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


def episode_id_for(tag: str, k: int) -> int:
    """(태그, 에피소드 번호) → 결정적 62비트 정수. 환경이 % ltt로 균등 선택한다 (P2와 같은 방식)."""
    h = hashlib.sha256(f"{tag}-ep{k}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 62) - 1)


# ---------------------------------------------------------------- 정책
def build_policy(ckpt: str, unnorm_key: str = "bridge_orig", attn: str = "flash_attention_2", seed: int = 0,
                 lr: float = 1e-4, vh_lr: float = 3e-3, temp_train: float = 1.0, temp_eval: float = 0.6):
    """원 OpenVLAPolicy. vla_load_path=""이면 원 PPO 레시피의 새 LoRA(gaussian A·0 B)를 붙인다 (C2).
    LoRA 초기화 시드 = seed (torch.manual_seed 뒤 생성). attn: 'flash_attention_2'(원 코드) 또는 'sdpa'."""
    from prismatic.extern.hf import modeling_prismatic as mp
    from simpler_env.policies.openvla import openvla_train as ot
    cls = mp.OpenVLAForActionPredictionWithValueHead
    if not hasattr(cls, "_g2_orig_fp"):
        cls._g2_orig_fp = cls.from_pretrained.__func__
    want = attn

    @classmethod
    def fp(c, *a, **kw):
        if kw.get("attn_implementation") == "flash_attention_2" and want != "flash_attention_2":
            kw["attn_implementation"] = want
        return cls._g2_orig_fp(c, *a, **kw)
    cls.from_pretrained = fp
    args = SimpleNamespace(vla_path=ckpt, vla_unnorm_key=unnorm_key, vla_load_path="", vla_lora_rank=32,
                           vla_lr=lr, vla_vhlr=vh_lr, vla_optim_beta1=0.9, vla_optim_beta2=0.999,
                           vla_temperature=temp_train, vla_temperature_eval=temp_eval)
    torch.manual_seed(seed)
    policy = ot.OpenVLAPolicy(args, device_id=0)
    return policy


def set_top_k(policy, mode: str):
    """K50: transformers 기본 top_k 50 (원 코드형). K0: top-k 절단 해제 (C4 기본안)."""
    gc = policy.vla.generation_config
    gc.top_k = 50 if mode == "K50" else 0
    gc.top_p = 1.0


def attn_impls(policy) -> list:
    s = set()
    for m in policy.vla.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            s.add(str(cfg._attn_implementation))
    return sorted(s)


# ---------------------------------------------------------------- 환경
import contextlib  # noqa: E402


@contextlib.contextmanager
def sim_nondeterministic_ok():
    """G2 S6 실측: torch 2.2 결정적 모드에서 ManiSkill GPU 시뮬레이션의 인덱스 대입(actor.set_pose 등)이
    내부 assert로 실패한다. 결정적 설정(C11)은 정책·학습기 경로에 적용하고, 환경 호출 동안만 플래그를 끈다."""
    on, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    if on:
        torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        if on:
            torch.use_deterministic_algorithms(True, warn_only=warn)


class GpuEnv:
    """SimlerWrapper와 같은 설정의 GPU 병렬 환경. 원 _process_action을 그대로 쓴다."""

    def __init__(self, env_id: str, unnorm_state: dict, num_envs: int = 64, seed: int = 0):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
        from simpler_env.env.simpler_wrapper import SimlerWrapper
        self.env_id, self.num_envs, self.unnorm_state = env_id, num_envs, unnorm_state
        with sim_nondeterministic_ok():
            self._make(gym, env_id, num_envs, seed)
        bins = np.linspace(-1, 1, 256)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self._proc = SimlerWrapper._process_action.__get__(self)

    def _make(self, gym, env_id, num_envs, seed):
        self.env = gym.make(id=env_id, num_envs=num_envs, obs_mode="rgb+segmentation", control_mode=CONTROL_MODE,
                            sim_backend="gpu", sim_config={"sim_freq": 500, "control_freq": 5},
                            max_episode_steps=T_EP, sensor_configs={"shader_pack": "default"})
        self.env.reset(seed=[seed * 1000 + i for i in range(num_envs)])

    @property
    def u(self):
        return self.env.unwrapped

    def reset(self, obj_set: str, episode_ids, reconfigure: bool = True, seed: int | None = None):
        dev = self.u.device
        eids = torch.as_tensor(np.asarray(episode_ids, dtype=np.int64), device=dev)
        assert len(eids) == self.num_envs
        opts = {"obj_set": obj_set, "episode_id": eids}
        if reconfigure:
            opts["reconfigure"] = True
        # numpy RandomState 한계(2³²) 안으로. 2³² 미만이던 값(G2 전부, E1 시드 1~12)은 그대로 (본실험 튜닝 시드 100에서 넘침)
        s = None if seed is None else [(int(seed) * 1000 + i) % (2 ** 32) for i in range(self.num_envs)]
        with sim_nondeterministic_ok():
            obs, info = self.env.reset(seed=s, options=opts)
        rgb = obs["sensor_data"][CAM]["rgb"].to(torch.uint8)
        instr = list(self.u.get_language_instruction())
        return rgb, instr

    def selected(self) -> dict:
        out = {}
        for n in ("select_carrot_ids", "select_plate_ids", "select_overlay_ids", "select_pos_ids", "select_quat_ids"):
            if hasattr(self.u, n):
                out[n.replace("select_", "")] = getattr(self.u, n).reshape(-1).cpu().numpy().astype(int)
        return out

    def step(self, tokens: torch.Tensor):
        action = self._proc(tokens)
        with sim_nondeterministic_ok():
            obs, _r, _term, trunc, info = self.env.step(action)
        rgb = obs["sensor_data"][CAM]["rgb"].to(torch.uint8)
        si = dict(success=info["success"].reshape(-1).bool().cpu().numpy(),
                  grasp_latch=info["is_src_obj_grasped"].reshape(-1).bool().cpu().numpy(),
                  consec_latch=info["consecutive_grasp"].reshape(-1).bool().cpu().numpy(),
                  truncated=trunc.reshape(-1).bool().cpu().numpy())
        return rgb, si

    def close(self):
        with sim_nondeterministic_ok():
            self.env.close()


# ---------------------------------------------------------------- 롤아웃
@torch.no_grad()
def rollout(policy, env: GpuEnv, obj_set: str, episode_ids, deterministic: bool, rng_seed: int,
            record_obs: bool = False, reconfigure: bool = True, timing: dict | None = None):
    """환경 수(64)만큼의 에피소드 한 배치를 80스텝 진행한다. deterministic=True면 평가 온도(0.6), False면 수집 온도(1.0).
    반환 dict: actions [T,N,7] int64, lp_gen [T,N], v_gen [T,N], success/grasp_latch/consec_latch [T,N] bool,
    instr [N], selected ids, (record_obs) obs [T,N,480,640,3] uint8 (CPU)."""
    N = env.num_envs
    t0 = time.time()
    rgb, instr = env.reset(obj_set, episode_ids, reconfigure=reconfigure, seed=rng_seed)
    t_reset = time.time() - t0
    torch.manual_seed(rng_seed)
    out = dict(actions=np.zeros((T_EP, N, 7), np.int64), lp_gen=np.zeros((T_EP, N), np.float32),
               v_gen=np.zeros((T_EP, N), np.float32), success=np.zeros((T_EP, N), bool),
               grasp_latch=np.zeros((T_EP, N), bool), consec_latch=np.zeros((T_EP, N), bool),
               instr=instr, episode_id=np.asarray(episode_ids, np.int64), selected=env.selected())
    if record_obs:
        out["obs"] = np.zeros((T_EP, N, 480, 640, 3), np.uint8)
    policy.prep_rollout()
    t_pol = t_sim = 0.0
    for t in range(T_EP):
        if record_obs:
            out["obs"][t] = rgb.cpu().numpy()
        tp = time.time()
        v, a, lp = policy.get_action(dict(image=rgb, task_description=instr), deterministic)
        torch.cuda.synchronize()
        t_pol += time.time() - tp
        out["actions"][t] = a.cpu().numpy()
        out["lp_gen"][t] = lp[:, 0].float().cpu().numpy()
        out["v_gen"][t] = v[:, 0].float().cpu().numpy()
        ts = time.time()
        rgb, si = env.step(a)
        t_sim += time.time() - ts
        out["success"][t], out["grasp_latch"][t], out["consec_latch"][t] = si["success"], si["grasp_latch"], si["consec_latch"]
        if t == T_EP - 1:
            assert si["truncated"].all(), "episodes did not truncate at step 80"
    out["Y"] = out["success"][-1] & out["grasp_latch"][-1]
    if timing is not None:
        timing.update(reset_s=t_reset, policy_s=t_pol, sim_s=t_sim, total_s=time.time() - t0)
    return out


def to_roundbatch(r: dict):
    """rollout 결과 → ppo.RoundBatch (extra에 obs·instr)."""
    import ppo
    return ppo.RoundBatch(actions=r["actions"], logprob_gen=r["lp_gen"], value_gen=r["v_gen"], success=r["success"],
                          grasp_latch=r["grasp_latch"], consec_latch=r["consec_latch"],
                          extra=dict(obs=r.get("obs"), instr=r["instr"]))


def gpu_mem() -> dict:
    return dict(alloc_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
                peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                reserved_gib=round(torch.cuda.memory_reserved() / 2**30, 2))
