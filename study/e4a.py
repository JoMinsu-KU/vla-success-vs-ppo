"""E4-a 배포형 교란 평가 (study/E4a_design.md — 제안, 승인 전에는 실행 계획에 넣지 않는다).

GPU 64개 배치, E1 평가 경로. 각 환경에서 집기 래치가 처음 켜진 스텝 직후(R-a′·R-c) 또는 그 2스텝 뒤(R-b)에
그 환경에만 교란을 한 번 걸고, 관측을 다시 렌더링한 뒤 원래 80스텝을 끝까지 진행한다.
사용: python study/e4a.py --run-id r… --ckpt <학습 상태 .pt | W1> [--name final] [--batches 3]
출력: <W>/blinded/<run_id>/e4a_<name>.jsonl, <W>/public/e4a/<run_id>_<name>.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

KINDS = ["control", "R-a'", "R-b", "R-c"]
DELAY = {"control": None, "R-a'": 0, "R-b": 2, "R-c": 0}  # 래치가 처음 켜진 스텝 뒤 몇 스텝에 거는가
COND = ("ID", "PutOnPlateInScene25Carrot-v1", "train")

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--name", default="final")
ap.add_argument("--batches", type=int, default=3)
a = ap.parse_args()

W = C.work_dir()
C.ensure_dirs(W)
OUT = W / "blinded" / a.run_id / f"e4a_{a.name}.jsonl"
PUB = W / "public" / "e4a" / f"{a.run_id}_{a.name}.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
PUB.parent.mkdir(parents=True, exist_ok=True)
rows = C.read_jsonl(OUT)
done = {(r["kind"], r["batch"]) for r in rows if sum(1 for x in rows if x["kind"] == r["kind"] and x["batch"] == r["batch"]) >= 64}
todo = [(k, b) for k in KINDS for b in range(a.batches) if (k, b) not in done]
C.write_json(PUB, dict(run_id=a.run_id, name=a.name, status="running" if todo else "done", done=len(done), total=len(KINDS) * a.batches,
                       updated=C.now()))
if not todo:
    sys.exit(0)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import g2_gpu as G  # noqa: E402
import g2_learn as L  # noqa: E402
import g2_s8_restore as S8  # noqa: E402


class Shim:
    """g2_s8_restore.Env.perturb를 G.GpuEnv에 쓰기 위한 얇은 연결 (같은 교란 코드)."""

    def __init__(self, genv):
        self.genv = genv

    @property
    def u(self):
        return self.genv.u

    src_actor = S8.Env.src_actor
    tgt_actor = S8.Env.tgt_actor
    _open_gripper = S8.Env._open_gripper
    perturb = S8.Env.perturb


policy = G.build_policy(C.w1_ckpt(), C.CFG["unnorm_key"], attn="flash_attention_2", seed=0)
G.set_top_k(policy, C.CFG["top_k"])
if a.ckpt != "W1":
    L.load_trainable(Path(a.ckpt), policy, load_optim=False)
env = G.GpuEnv(COND[1], policy.vla.get_action_stats(C.CFG["unnorm_key"]), 64, seed=0)
shim = Shim(env)
ids = [G.episode_id_for(f"main-e4a-{COND[0]}-{COND[1]}-{COND[2]}", k) for k in range(64 * a.batches)]
t0 = time.time()
for kind, b in todo:
    eids = ids[64 * b:64 * (b + 1)]
    rng_seed = 1000 + b
    rgb, instr = env.reset(COND[2], eids, reconfigure=True, seed=rng_seed)
    torch.manual_seed(rng_seed)
    policy.prep_rollout()
    first_grasp = np.full(64, -1)
    fired = np.full(64, -1)
    succ = latch = None
    with torch.no_grad():
        for t in range(G.T_EP):
            _v, act, _lp = policy.get_action(dict(image=rgb, task_description=instr), True)
            rgb, si = env.step(act)
            succ, latch = np.asarray(si["success"]), np.asarray(si["grasp_latch"])
            newly = (first_grasp < 0) & latch
            first_grasp[newly] = t
            if DELAY[kind] is not None and t < G.T_EP - 1:
                due = (fired < 0) & (first_grasp >= 0) & (t == first_grasp + DELAY[kind])
                if due.any():
                    kinds = [kind if due[i] else "none" for i in range(64)]
                    shim.perturb(kinds, [None] * 64)
                    fired[due] = t
                    with G.sim_nondeterministic_ok():
                        rgb = env.u.get_obs()["sensor_data"][G.CAM]["rgb"].to(torch.uint8)
    with OUT.open("a", encoding="utf-8") as f:
        import json
        for i in range(64):
            f.write(json.dumps(dict(kind=kind, batch=b, k=64 * b + i, episode_id=int(eids[i]), Y=bool(succ[i] and latch[i]),
                                    reached=bool(fired[i] >= 0) if kind != "control" else bool(first_grasp[i] >= 0),
                                    first_grasp_step=int(first_grasp[i]), fired_step=int(fired[i]))) + "\n")
    done.add((kind, b))
    C.write_json(PUB, dict(run_id=a.run_id, name=a.name, status="running", done=len(done), total=len(KINDS) * a.batches, updated=C.now()))
    print(f"[e4a] {kind} b{b} {time.time() - t0:.0f}s", flush=True)
env.close()
(OUT.parent / f"sha256_{OUT.name}.txt").write_text(C.sha256_file(OUT) + "\n")
C.write_json(PUB, dict(run_id=a.run_id, name=a.name, status="done", done=len(done), total=len(KINDS) * a.batches, updated=C.now()))
