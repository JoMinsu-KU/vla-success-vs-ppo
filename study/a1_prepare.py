"""A1 시연 데이터 전처리 (study/A1_design.md 1절): 모션 플래너 npz → 학습 프레임 memmap + 행동 토큰.

원 tfds 빌더(RL4VLA openvla/rlds_dataset_builder/sft_dataset/sft_dataset.py)의 규칙을 그대로 옮긴다:
  - 파일 이름 정렬, 학습 = 끝 16개를 뺀 목록의 앞 16,384개, 검증 = [16,384 : 16,400]
  - filter_small_actions(이동 0.01, 회전 0.06, 그리퍼 전환 유지)로 행동·이미지를 거른 뒤,
    거른 인덱스 i에 대해 원본 info[i]["success"]로 연속 성공을 세어 6이 되면 그 스텝까지 넣고 멈춘다
    (원 코드가 거른 인덱스로 원본 info를 읽는 방식까지 그대로)
  - 이미지: JPEG 저장 없이 렌더 원본 uint8 480×640×3 (설계 문서의 결정)
  - 행동 → 토큰 (C1): bridge_orig q01·q99로 [−1, 1] 정규화(마스크 False인 그리퍼 차원은 그대로) → [−1, 1] 자르기
    → np.digitize(x, linspace(−1, 1, 256)) → 토큰 32000 − bin  (원 ActionTokenizer)
출력 (<out>/<split>/): frames.u8 (memmap [M,480,640,3]), tokens.npy [M,7] int64, ep.npy [M] int32, instr.json (에피소드별),
  files.json, meta.json (프레임 수, 제거 수, SHA-256: 토큰·프레임·파일 목록)
사용: python study/a1_prepare.py <npz 폴더> <출력 폴더> <W1 체크포인트> [--procs 4] [--split train|val|both]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

N_TRAIN, SPARE = 16384, 16
BINS = np.linspace(-1, 1, 256)
VOCAB = 32000


def filter_small_actions(actions, pos_thresh=0.01, rot_thresh=0.06, check_gripper=True):
    """원 sft_dataset.py의 함수 그대로."""
    actions = np.asarray(actions)
    N = actions.shape[0]
    valid_mask = np.zeros(N, dtype=bool)
    for i in range(N):
        act = actions[i]
        delta_xyz = act[:3]
        delta_euler = act[3:6]
        gripper = act[6]
        pos_movement = np.linalg.norm(delta_xyz)
        rot_movement = np.linalg.norm(delta_euler)
        if pos_thresh is None and rot_thresh is None:
            is_valid = True
        elif pos_thresh is None:
            is_valid = (rot_movement > rot_thresh)
        elif rot_thresh is None:
            is_valid = (pos_movement > pos_thresh)
        else:
            is_valid = (pos_movement > pos_thresh) or (rot_movement > rot_thresh)
        if check_gripper and i > 0 and actions[i - 1][6] != gripper:
            is_valid = True
        valid_mask[i] = is_valid
    return valid_mask


def split_files(files: list) -> dict:
    """원 _generate_examples(16384, spare=16) / (16, start=16384)와 같은 선택."""
    assert len(files) >= N_TRAIN + SPARE, len(files)
    train = files[:-SPARE][:N_TRAIN]
    start = min(N_TRAIN, len(files) - SPARE)
    val = files[start:start + SPARE]
    assert len(train) == N_TRAIN and len(val) == SPARE
    return dict(train=train, val=val)


def tokenize(actions: np.ndarray, stats: dict) -> np.ndarray:
    a = np.asarray(actions, np.float64)
    lo, hi = np.asarray(stats["q01"], np.float64), np.asarray(stats["q99"], np.float64)
    mask = np.asarray(stats.get("mask", np.ones_like(lo, bool)), bool)
    x = np.where(mask, np.clip(2 * (a - lo) / (hi - lo + 1e-8) - 1, -1, 1), a)  # 원 normalize_action_and_proprio(BOUNDS_Q99)
    x = np.clip(x, -1.0, 1.0)                                                       # 원 ActionTokenizer.__call__
    return (VOCAB - np.digitize(x, BINS)).astype(np.int64)


def detokenize(tokens: np.ndarray, stats: dict) -> np.ndarray:
    """환경의 역변환 (SimlerWrapper._process_action과 같은 식) — 왕복 시험용."""
    centers = (BINS[:-1] + BINS[1:]) / 2.0
    d = np.clip(VOCAB - np.asarray(tokens) - 1, 0, 254)
    xn = centers[d]
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", np.ones_like(lo, bool)), bool)
    return np.where(mask, 0.5 * (xn + 1) * (hi - lo) + lo, xn)


def episode(path: str):
    data = np.load(path, allow_pickle=True)["arr_0"].tolist()
    ins = data["instruction"]
    ins = ins.tolist()[0] if isinstance(ins, np.ndarray) else ins
    if isinstance(ins, list):
        ins = ins[0]
    actions = np.asarray(data["action"])
    images = np.asarray([np.asarray(img) for img in data["image"]])
    mask = filter_small_actions(actions)
    actions, images = actions[mask], images[mask]
    keep, success_count = 0, 0
    for i in range(len(actions)):
        keep = i + 1
        success_count = success_count + 1 if data["info"][i]["success"] else 0  # 원 코드: 거른 인덱스로 원본 info
        if success_count >= 6:
            break
    return str(ins), actions[:keep].astype(np.float32), images[:keep].astype(np.uint8), int((~mask).sum()), int(mask.size)


def build(files: list, out: Path, stats: dict, procs: int):
    out.mkdir(parents=True, exist_ok=True)
    fr = out / "frames.u8"
    if fr.exists():
        fr.unlink()
    h_fr, h_tok = hashlib.sha256(), hashlib.sha256()
    toks, eps, instr = [], [], []
    removed = total = 0
    with fr.open("ab") as f, mp.get_context("spawn").Pool(procs) as pool:
        for k, (ins, acts, imgs, rm, tot) in enumerate(pool.imap(episode, files, chunksize=4)):
            assert imgs.shape[1:] == (480, 640, 3), imgs.shape
            b = np.ascontiguousarray(imgs).tobytes()
            f.write(b)
            h_fr.update(b)
            t = tokenize(acts, stats)
            h_tok.update(t.tobytes())
            toks.append(t)
            eps.append(np.full(len(t), k, np.int32))
            instr.append(ins)
            removed += rm
            total += tot
            if (k + 1) % 500 == 0:
                print(f"[{out.name}] {k + 1}/{len(files)}", flush=True)
    tokens = np.concatenate(toks)
    np.save(out / "tokens.npy", tokens)
    np.save(out / "ep.npy", np.concatenate(eps))
    (out / "instr.json").write_text(json.dumps(instr, ensure_ascii=False))
    names = [Path(x).name for x in files]
    (out / "files.json").write_text(json.dumps(names))
    meta = dict(episodes=len(files), frames=int(len(tokens)), shape=[int(len(tokens)), 480, 640, 3],
                actions_before_filter=total, removed_by_filter=removed,
                sha256=dict(frames=h_fr.hexdigest(), tokens=h_tok.hexdigest(),
                            files=hashlib.sha256(json.dumps(names).encode()).hexdigest()))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_dir")
    ap.add_argument("out_dir")
    ap.add_argument("w1_ckpt")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    a = ap.parse_args()
    stats = json.loads((Path(a.w1_ckpt) / "config.json").read_text())["norm_stats"]["bridge_orig"]["action"]
    files = sorted(glob.glob(str(Path(a.npz_dir) / "*.npz")))
    sp = split_files(files)
    for name in (["train", "val"] if a.split == "both" else [a.split]):
        build(sp[name], Path(a.out_dir) / name, stats, a.procs)


if __name__ == "__main__":
    main()
