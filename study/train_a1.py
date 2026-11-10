"""A1 시연 SFT 재학습 1회 (study/A1_design.md 2절).

W1 + 온라인 갈래와 같은 새 LoRA(초기화 시드 = 학습 시드), 행동 토큰 7개 NLL 평균 (A2⁺와 같은 evaluate_actions 경로),
AdamW(lr 5e-4, β 0.9·0.999, weight decay 0.01, 상수), 클리핑 없음, 전역 배치 32, 60,000스텝, 결정적 math 경로.
표본 순서: 에폭 e마다 학습 프레임 전체를 시드 1,000·seed + e로 섞어 32개씩 (프레임 수가 32로 나누어떨어지지 않으면
에폭 끝의 남은 표본은 다음 에폭으로 넘기지 않고 버린다).
재시작: <W>/runs/<run_id>/ckpt/latest.pt (LoRA + Adam 상태 + 스텝)에서 이어 간다.
맹검·공개: 손실은 <W>/blinded/<run_id>/a1_loss.jsonl, 공개는 진행 스텝 수와 상태, 끝난 뒤 GPU시간.
사용: python study/train_a1.py --run-id r… --seed 1 --data <a1 데이터>/train [--steps 60000] [--numerics math]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--steps", type=int, default=60_000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--numerics", default="math", choices=["math", "default"])
ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--latest-every", type=int, default=5000)
ap.add_argument("--log-every", type=int, default=100)
ap.add_argument("--stop-after", type=int, default=0, help="시험용: 이 스텝까지만 하고 멈춤")
a = ap.parse_args()

W = C.work_dir()
C.ensure_dirs(W)
RUN = W / "runs" / a.run_id
BL = W / "blinded" / a.run_id
(RUN / "ckpt").mkdir(parents=True, exist_ok=True)
BL.mkdir(parents=True, exist_ok=True)
PUB = W / "public" / "runs" / f"{a.run_id}.json"

import numerics  # noqa: E402

if a.numerics == "math":
    numerics.enable_deterministic(strict=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import g2_gpu as G  # noqa: E402
import g2_learn as L  # noqa: E402


def pub(status, done, **kw):
    C.write_json(PUB, dict(run_id=a.run_id, status=status, rounds_done=done, rounds=a.steps, unit="steps", updated=C.now(), **kw))


def save(path: Path, extra: dict):
    tmp = path.with_suffix(".tmp")
    L.save_trainable(tmp, policy, extra=extra)
    os.replace(tmp, path)


D = Path(a.data)
meta = json.loads((D / "meta.json").read_text())
M = int(meta["frames"])
frames = np.memmap(D / "frames.u8", dtype=np.uint8, mode="r", shape=(M, 480, 640, 3))
tokens = np.load(D / "tokens.npy")
ep = np.load(D / "ep.npy")
instr = json.loads((D / "instr.json").read_text())
assert len(tokens) == M == len(ep)
per_epoch = M // a.batch


def batch_indices(step: int) -> np.ndarray:
    e, k = divmod(step, per_epoch)
    order = np.random.default_rng(1000 * a.seed + e).permutation(M)
    return order[k * a.batch:(k + 1) * a.batch]


attn = "sdpa" if a.numerics == "math" else "flash_attention_2"
policy = G.build_policy(C.w1_ckpt(), C.CFG["unnorm_key"], attn=attn, seed=a.seed, lr=a.lr)
state = dict(step=0, gpu_s=0.0)
latest = RUN / "ckpt" / "latest.pt"
if latest.exists():
    ex = L.load_trainable(latest, policy, load_optim=True)
    state.update({k: ex[k] for k in state if k in ex})
start = int(state["step"])
rows = [r for r in C.read_jsonl(BL / "a1_loss.jsonl") if r["step"] <= start]
(BL / "a1_loss.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
C.write_json(BL / f"meta_start{start:06d}.json", dict(args=vars(a), start_step=start, data_meta=meta, frames_per_epoch=per_epoch,
                                                       numerics=numerics.numerics_report(policy.vla)))
if start >= a.steps:
    pub("done", start)
    sys.exit(0)
pub("running", start)
params = list(policy.params_vla)
opt = policy.vla_optimizer  # AdamW(lr, β 0.9·0.999), weight decay 기본 0.01 — 원 SFT와 같음
dev = torch.device("cuda:0")
policy.prep_training()
t_seg, acc, n_acc = time.time(), 0.0, 0
status = "running"
for step in range(start, a.steps):
    idx = np.sort(batch_indices(step))  # 같은 표본 집합, 메모리 읽기 순서만 정렬
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for j in range(0, len(idx), a.chunk):
        ii = idx[j:j + a.chunk]
        x = dict(image=torch.from_numpy(np.ascontiguousarray(frames[ii])).to(dev), task_description=[instr[int(ep[i])] for i in ii])
        act = torch.from_numpy(tokens[ii]).to(dev)
        lp, _ent, _v = policy.evaluate_actions(x, act)
        loss = (-lp.float() / 7.0).sum() / len(idx)
        loss.backward()
        tot += float(loss)
    if not math.isfinite(tot):
        status = "diverged_nonfinite"
        C.append_jsonl(BL / "a1_loss.jsonl", dict(step=step + 1, loss=tot, finite=False))
        break
    opt.step()
    acc += tot
    n_acc += 1
    if (step + 1) % a.log_every == 0:
        C.append_jsonl(BL / "a1_loss.jsonl", dict(step=step + 1, loss=acc / n_acc, s_per_step=round((time.time() - t_seg) / n_acc, 3)))
        state["gpu_s"] += time.time() - t_seg
        t_seg, acc, n_acc = time.time(), 0.0, 0
        pub("running", step + 1)
    state["step"] = step + 1
    if (step + 1) % a.latest_every == 0 or (step + 1) == a.steps:
        save(latest, dict(state))
    if a.stop_after and step + 1 >= a.stop_after:
        break

state["gpu_s"] += time.time() - t_seg
if status == "running" and state["step"] >= a.steps:
    status = "done"
if status != "running":
    save(latest, dict(state))
    save(RUN / "ckpt" / "final.pt", dict(state))
    (BL / "sha256_a1_loss.jsonl.txt").write_text(C.sha256_file(BL / "a1_loss.jsonl") + "\n")
    pub(status, state["step"], gpu_h_total=round(state["gpu_s"] / 3600, 2))
else:
    save(latest, dict(state))
    pub("paused", state["step"])
