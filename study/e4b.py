"""E4-b 공통 상태 회복 평가 (study/E4b_design.md, 실행 전 고정). CPU PhysX 환경 1개, 정책은 GPU.

사용: python study/e4b.py --run-id r… --ckpt <학습 상태 .pt> [--base-ckpt <W1 또는 W2 폴더>] [--name final] [--smoke]
입력: S_rec·타당 원소는 $VLA_SREC_DIR (기본 ~/vla_g2/s8: srec.pkl, g2_s8.json), R-c 제외는 T2 결과(저장소 artifacts).
출력: <W>/blinded/<run_id>/e4b_<name>.jsonl (맹검), <W>/public/e4b/<run_id>_<name>.json (진행 상태)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SREC_SHA = "e0da26c550027e0548177696b64101df21fe910c1665e5de0ea234aae4357f5c"
KINDS = ["control", "R-a'", "R-b", "R-c"]
H = 40
REPS = 3
HEIGHT_MM = 5.0

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--base-ckpt", default="")
ap.add_argument("--name", default="final")
ap.add_argument("--smoke", action="store_true", help="시험용: 원소 2개, 반복 1회")
a = ap.parse_args()

W = C.work_dir()
C.ensure_dirs(W)
OUT = W / "blinded" / a.run_id / f"e4b_{a.name}.jsonl"
PUB = W / "public" / "e4b" / f"{a.run_id}_{a.name}.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
PUB.parent.mkdir(parents=True, exist_ok=True)

SREC_DIR = Path(os.environ.get("VLA_SREC_DIR", str(Path.home() / "vla_g2" / "s8"))).expanduser()
srec_f = SREC_DIR / "srec.pkl"
if hashlib.sha256(srec_f.read_bytes()).hexdigest() != SREC_SHA:
    raise SystemExit(f"S_rec 해시가 다르다: {srec_f}")
recs = pickle.load(open(srec_f, "rb"))
valid = [b["j"] for b in json.loads((SREC_DIR / "g2_s8.json").read_text())["B_validity"]["cases"] if b["valid"]]
t2 = json.loads((C.REPO / "pilot" / "artifacts" / "g2" / "gpre_t23" / "gpre_t2t3.json").read_text())["T2"]
excluded = {k: set(t2[k]["excluded"]) for k in KINDS}
height = {r["j"]: 1000 * r["z_above_t0_m"] for r in t2["R-a'"]["rows"]}
assert len(valid) == 21 and excluded["R-c"] == {5, 19} and all(not excluded[k] for k in ("control", "R-a'", "R-b"))

reps = 1 if a.smoke else REPS
elems = valid[:2] if a.smoke else valid
todo = [(j, k, r) for r in range(reps) for k in KINDS for j in elems if j not in excluded[k]]
done = {(x["j"], x["kind"], x["rep"]) for x in C.read_jsonl(OUT)}
todo = [t for t in todo if t not in done]


def pub(status):
    C.write_json(PUB, dict(run_id=a.run_id, name=a.name, status=status, done=len(done), total=len(done) + len(todo), updated=C.now()))


pub("running" if todo else "done")
if not todo:
    sys.exit(0)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import g2_gpu as G  # noqa: E402
import g2_learn as L  # noqa: E402
import g2_s8_restore as S8  # noqa: E402

base = a.base_ckpt or C.w1_ckpt()
policy = G.build_policy(base, C.CFG["unnorm_key"], attn="flash_attention_2", seed=0)
G.set_top_k(policy, C.CFG["top_k"])
if a.ckpt != "W1":
    L.load_trainable(Path(a.ckpt), policy, load_optim=False)
policy.prep_rollout()
env = S8.Env("cpu", policy.vla.get_action_stats(C.CFG["unnorm_key"]))
t0 = time.time()
for n_done, (j, k, r) in enumerate(todo):
    rec = recs[j]
    torch.manual_seed(1_000_000 + 10_000 * r + 100 * KINDS.index(k) + j)
    env.reset([rec["eid"]])
    env.set_full([rec["base"]])
    env.perturb([k], [rec["obj_pose0"]])
    rgb, ins = env.obs(), [rec["instr"]]
    info = None
    for _t in range(H):
        rgb, info = env.step(S8.policy_tokens(policy, rgb, ins))
    succ, latch = bool(info["success"][0]), bool(info["is_src_obj_grasped"][0])
    C.append_jsonl(OUT, dict(j=j, kind=k, rep=r, Y=succ and latch, success=succ, latch=latch,
                             height_ge_5mm=bool(height[j] >= HEIGHT_MM)))
    done.add((j, k, r))
    if (n_done + 1) % 10 == 0:
        pub("running")
        print(f"[e4b] {len(done)} {time.time() - t0:.0f}s", flush=True)
env.close()
(OUT.parent / f"sha256_{OUT.name}.txt").write_text(C.sha256_file(OUT) + "\n")
todo = []
pub("done")
