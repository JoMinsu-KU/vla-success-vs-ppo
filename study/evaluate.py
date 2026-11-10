"""본실험 평가 (기본 수치 경로: flash_attention_2, K0, 온도 0.6, 배치마다 장면 재구성, 성공 = Y).

묶음(--suite):
  idval : 튜닝용 ID 검증. 학습 환경(Main-v3, train)에서 512 에피소드(64 × 8). 고정 episode_id 가운데
          튜닝·E1 학습에 쓰는 초기 상태(episode_id mod ltt)와 겹치는 것은 건너뛴다 (명세 3.5절).
  e1    : 평가 조건 16개 × 192 에피소드 (조건마다 64 × 3). 모든 갈래·시드·체크포인트가 같은 고정 집합 (명세 3.8절).
          G2 S4 집합과 다른 새 태그(main-eval-…)로 만든다 (G-PRE 초안 8절).
  e2b   : E2-b 분기 평가 (명세 3.6절): 학습 조건(ID)과 실행 3과제, 조건당 128 에피소드 = E1 고정 집합의 앞 128개 (64 × 2).
칸 배정: 에피소드 k → 배치 ⌊k/64⌋, 칸 k mod 64, 샘플링 시드 1000 + 배치 (G2 S4·T3와 같음).
맹검: 에피소드별 결과는 <W>/blinded/<run_id>/eval_<suite>_<name>.jsonl에만. 공개는 진행 상태만.
재시작: 이미 기록된 (조건, 배치)는 건너뛴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--suite", required=True, choices=["idval", "e1", "e2b"])
ap.add_argument("--ckpt", required=True, help="학습 상태 .pt 경로 또는 W1 (학습 전)")
ap.add_argument("--name", required=True, help="체크포인트 이름 (예: r150, final)")
ap.add_argument("--conds", default="", help="쉼표 구분 조건 이름만 (시험용)")
ap.add_argument("--batches", type=int, default=0, help="조건당 배치 수 제한 (시험용)")
a = ap.parse_args()

W = C.work_dir()
C.ensure_dirs(W)
OUT = W / "blinded" / a.run_id / f"eval_{a.suite}_{a.name}.jsonl"
PUB = W / "public" / "evals" / f"{a.run_id}_{a.suite}_{a.name}.json"

import numpy as np  # noqa: E402
import torch  # noqa: E402

import g2_gpu as G  # noqa: E402
import g2_learn as L  # noqa: E402


def train_residues(ltt: int) -> set:
    """튜닝·E1 학습에 쓰는(쓸) 모든 초기 상태의 episode_id mod ltt."""
    tags = [f"tune-seed{C.CFG['tune_seed']}-round{r}" for r in range(C.CFG["rounds_tune"])]
    tags += [f"e1-seed{s}-round{r}" for s in C.CFG["e1_seeds_all"] for r in range(C.CFG["rounds_e1"])]
    return {G.episode_id_for(t, k) % ltt for t in tags for k in range(C.CFG["n_envs"])}


def idval_ids(env, n: int) -> list:
    u = env.u
    ltt = 16 * len(u.plate_names) * 16 * len(u.xyz_configs) * len(u.quat_configs)  # obj_set train
    used = train_residues(ltt)
    out, k = [], 0
    while len(out) < n:
        e = G.episode_id_for("main-idval", k)
        if e % ltt not in used:
            out.append(e)
        k += 1
    return out, dict(ltt=ltt, skipped=k - n, train_residues=len(used))


def main():
    rows = C.read_jsonl(OUT)
    done = {(c, b) for c, b in {(r["cond"], r["batch"]) for r in rows}
            if sum(1 for r in rows if r["cond"] == c and r["batch"] == b) >= 64}  # 64개가 다 있는 배치만 끝난 것으로
    if a.suite == "idval":
        conds = [("IDval", C.CFG["train_env"], C.CFG["train_obj_set"], "idval")]
        n_ep = C.CFG["idval_episodes"]
    elif a.suite == "e2b":
        conds = [tuple(c) for c in C.CFG["eval_conditions"] if c[0] == "ID" or c[3] == "execution"]
        n_ep = 128
    else:
        conds = [tuple(c) for c in C.CFG["eval_conditions"]]
        n_ep = C.CFG["eval_episodes_per_cond"]
    if a.conds:
        keep = set(a.conds.split(","))
        conds = [c for c in conds if c[0] in keep]
    n_b = n_ep // 64 if not a.batches else min(a.batches, n_ep // 64)
    todo = [(c, b) for c in conds for b in range(n_b) if (c[0], b) not in done]
    C.write_json(PUB, dict(run_id=a.run_id, suite=a.suite, name=a.name, status="running" if todo else "done",
                           batches_done=len(done), batches_total=len(conds) * n_b, updated=C.now()))
    if not todo:
        return
    policy = G.build_policy(C.w1_ckpt(), C.CFG["unnorm_key"], attn="flash_attention_2", seed=0)
    G.set_top_k(policy, C.CFG["top_k"])
    if a.ckpt != "W1":
        L.load_trainable(Path(a.ckpt), policy, load_optim=False)
    ustate = policy.vla.get_action_stats(C.CFG["unnorm_key"])
    env, cur, sets, info = None, None, {}, {}
    for (lab, env_id, split, _axis), b in todo:
        if env_id != cur:
            if env is not None:
                env.close()
            env, cur = G.GpuEnv(env_id, ustate, 64, seed=0), env_id
        if lab not in sets:
            if a.suite == "idval":
                sets[lab], info[lab] = idval_ids(env, n_ep)
            else:
                sets[lab] = [G.episode_id_for(f"main-eval-{lab}-{env_id}-{split}", k) for k in range(n_ep)]  # e2b: E1 집합의 앞 128개
            C.write_json(W / "public" / "eval_sets" / f"{a.suite}_{lab}.json",
                         dict(sha256=hashlib.sha256(json.dumps(sets[lab]).encode()).hexdigest(), n=len(sets[lab]), **info.get(lab, {})))
        eids = sets[lab][64 * b:64 * (b + 1)]
        tm = {}
        r = G.rollout(policy, env, split, eids, deterministic=True, rng_seed=1000 + b, timing=tm)
        sel = r["selected"]
        with OUT.open("a", encoding="utf-8") as f:
            for i in range(64):
                f.write(json.dumps(dict(cond=lab, env_id=env_id, obj_set=split, batch=b, k=64 * b + i, episode_id=int(eids[i]),
                                        Y=bool(r["Y"][i]), success=bool(r["success"][-1, i]), success_any=bool(r["success"][:, i].any()),
                                        grasp_latch=bool(r["grasp_latch"][-1, i]), eval_s=round(tm["total_s"], 1),
                                        **{k: int(v[i]) for k, v in sel.items()})) + "\n")
        done.add((lab, b))
        C.write_json(PUB, dict(run_id=a.run_id, suite=a.suite, name=a.name, status="running", batches_done=len(done),
                               batches_total=len(conds) * n_b, updated=C.now()))
    env.close()
    (OUT.parent / f"sha256_{OUT.name}.txt").write_text(C.sha256_file(OUT) + "\n")
    C.write_json(PUB, dict(run_id=a.run_id, suite=a.suite, name=a.name, status="done", batches_done=len(done),
                           batches_total=len(conds) * n_b, updated=C.now()))


if __name__ == "__main__":
    main()
