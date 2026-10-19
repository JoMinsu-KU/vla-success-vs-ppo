"""궤적 저장 모듈 (연구 명세 v0.10, 3.2절 C6, 7.4절 계층 저장).

원 코드는 렌더링 경로에서 mp4와 통계만 남기고, `eval_ms3_collect.py`는 성공 스텝 6개 이상인 에피소드만 저장했다.
이 모듈은 모든 온라인 갈래가 같은 형식으로 라운드를 저장하고, 성공 정의는 C5의 Y로 통일한다.

계층 (7.4절):
  1 저차원 기록 — 모든 라운드·모든 스텝: 시뮬레이터 상태(평탄화), 제어기·과제 상태, 행동 토큰, 역정규화 행동,
    저장 log-prob(생성 경로), value, success·집기·연속 집기, 보상(원 잠재함수·terminal), Y, episode_id, 지시문.
  2 학습 관측 RGB — 무손실(zlib) 압축. 보존 규칙: 지정 라운드(기본 20·150·300)는 전체, 그 밖의 라운드는
    앞 `keep_envs`개 에피소드만. (현재 라운드 학습 입력은 학습기가 메모리에서 쓰고, 끝나면 이 규칙으로만 남긴다.)
  3 정성 분석 영상 — `video_every` 라운드마다 앞 `video_envs`개 에피소드 mp4 (픽셀 보존 아님).
모든 파일은 SHA-256과 함께 라운드 manifest(JSON)에 기록하고, 읽을 때 해시를 검증한다.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

TIER1_KEYS = ("sim_state", "controller_state", "task_state", "tokens", "action", "logprob_gen", "value_gen",
              "success", "grasp_latch", "consec_latch", "grasp_now", "reward_potential")


@dataclass
class StoragePolicy:
    full_rgb_rounds: tuple = (20, 150, 300)
    keep_envs: int = 2
    video_every: int = 40
    video_envs: int = 4


@dataclass
class EpisodeRecord:
    """한 에피소드의 기록. 배열의 첫 차원은 스텝 T (state 계열은 T+1: 각 행동 전 상태 + 마지막 상태)."""
    episode_id: int
    instruction: str
    sim_state: np.ndarray          # [T+1, S] float32  env.get_state() 평탄화
    controller_state: np.ndarray   # [T+1, C] float32  EE 목표 pose 등
    task_state: np.ndarray         # [T+1, K] int64    래치 3개, 연속 집기 카운터, 경과 스텝
    tokens: np.ndarray             # [T, 7] int64
    action: np.ndarray             # [T, 7] float32 역정규화 실행 명령
    logprob_gen: np.ndarray        # [T] float32
    value_gen: np.ndarray          # [T] float32
    success: np.ndarray            # [T] bool
    grasp_latch: np.ndarray        # [T] bool
    consec_latch: np.ndarray       # [T] bool
    grasp_now: np.ndarray          # [T] bool
    reward_potential: np.ndarray   # [T] float32 원 잠재함수 변화량
    rgb: np.ndarray | None = None  # [T+1, H, W, 3] uint8 (계층 2)
    meta: dict = field(default_factory=dict)

    @property
    def Y(self) -> int:
        return int(self.success[-1] and self.grasp_latch[-1])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class RoundStore:
    def __init__(self, root, arm: str, seed: int, policy: StoragePolicy | None = None):
        self.root = Path(root) / arm / f"seed{seed}"
        self.policy = policy or StoragePolicy()
        self.root.mkdir(parents=True, exist_ok=True)

    def round_dir(self, rnd: int) -> Path:
        return self.root / f"round{rnd:04d}"

    def keep_rgb(self, rnd: int, ep_index: int) -> bool:
        return rnd in self.policy.full_rgb_rounds or ep_index < self.policy.keep_envs

    def write_round(self, rnd: int, episodes: list[EpisodeRecord], extra_meta: dict | None = None) -> dict:
        d = self.round_dir(rnd)
        d.mkdir(parents=True, exist_ok=True)
        files = {}
        t1 = {k: np.stack([getattr(e, k) for e in episodes]) for k in TIER1_KEYS}
        t1["episode_id"] = np.array([e.episode_id for e in episodes], np.int64)
        t1["Y"] = np.array([e.Y for e in episodes], np.int64)
        p = d / "tier1.npz"
        np.savez_compressed(p, **t1)
        files["tier1.npz"] = _sha256(p)
        (d / "instructions.json").write_text(json.dumps([e.instruction for e in episodes], ensure_ascii=False),
                                             encoding="utf-8")
        files["instructions.json"] = _sha256(d / "instructions.json")
        kept = []
        for i, e in enumerate(episodes):
            if e.rgb is not None and self.keep_rgb(rnd, i):
                p = d / f"rgb_ep{i:03d}.npz"
                np.savez_compressed(p, rgb=e.rgb)
                files[p.name] = _sha256(p)
                kept.append(i)
        if self.policy.video_every and rnd % self.policy.video_every == 0:
            try:
                from mani_skill.utils.visualization.misc import images_to_video
                for i, e in enumerate(episodes[: self.policy.video_envs]):
                    if e.rgb is not None:
                        images_to_video(list(e.rgb), str(d), f"video_ep{i:03d}_Y{e.Y}", fps=10, verbose=False)
                        files[f"video_ep{i:03d}_Y{e.Y}.mp4"] = _sha256(d / f"video_ep{i:03d}_Y{e.Y}.mp4")
            except Exception as ex:  # 영상은 보조 자료라 실패해도 기록만
                files["video_error"] = repr(ex)
        manifest = dict(round=rnd, n_episodes=len(episodes), rgb_kept_episodes=kept, files=files,
                        storage_policy=asdict(self.policy), meta=extra_meta or {})
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest

    def read_round(self, rnd: int, verify: bool = True):
        d = self.round_dir(rnd)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if verify:
            for name, h in manifest["files"].items():
                if name.endswith((".npz", ".json", ".mp4")) and _sha256(d / name) != h:
                    raise IOError(f"hash mismatch: {d / name}")
        with np.load(d / "tier1.npz") as z:
            t1 = {k: z[k] for k in z.files}
        instr = json.loads((d / "instructions.json").read_text(encoding="utf-8"))
        rgb = {}
        for i in manifest["rgb_kept_episodes"]:
            with np.load(d / f"rgb_ep{i:03d}.npz") as z:
                rgb[i] = z["rgb"]
        return manifest, t1, instr, rgb

    def round_bytes(self, rnd: int) -> int:
        return sum(p.stat().st_size for p in self.round_dir(rnd).iterdir())
