"""W2 강화 웜업 만들기 (study/W2_design.md, W2 학습 전 고정).

단계 (각각 별도 프로세스, 끝난 단계는 건너뜀):
  check0      W1의 학습 조건 수집 성공률 (참고값)
  N마다       train (train_a1.py, 2,000스텝) → merge (LoRA를 W1에 병합, value head 제외) → mergecheck → check
  선택        사다리 256·512·1,024·2,048에서 성공률 ≥ 70%인 첫 N, 없으면 2,048
사용: python study/build_w2.py --data <W2 전처리 폴더(n256…)> --out <W2 체크포인트 상위 폴더> [--smoke]
      python study/build_w2.py --stage merge|mergecheck|check --n N ...   (내부용)
공개 출력: <W>/public/w2/check_n<N>.json, mergecheck_n<N>.json, <W>/public/w2_selection.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

LADDER = [256, 512, 1024, 2048]
TARGET = 0.70
STEPS = 2000

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stage", default="all", choices=["all", "merge", "mergecheck", "check"])
ap.add_argument("--n", type=int, default=0, help="0 = W1 (check 단계 참고값)")
ap.add_argument("--smoke", action="store_true", help="시험용: 사다리 256만, 20스텝, 측정 1배치")
a = ap.parse_args()

W = C.work_dir()
C.ensure_dirs(W)
PUBD = W / "public" / "w2"
PUBD.mkdir(parents=True, exist_ok=True)
OUTD = Path(a.out).expanduser()
OUTD.mkdir(parents=True, exist_ok=True)
W1 = Path(C.w1_ckpt())


def w2_dir(n: int) -> Path:
    return OUTD / f"w2_n{n}"


def rid(n: int) -> str:
    return f"w2n{n}" + ("s" if a.smoke else "")


# ---------------------------------------------------------------- 단계: 병합
def stage_merge(n: int):
    import torch
    import g2_gpu as G
    import g2_learn as L
    policy = G.build_policy(str(W1), C.CFG["unnorm_key"], attn="sdpa", seed=0)
    L.load_trainable(W / "runs" / rid(n) / "ckpt" / "final.pt", policy, load_optim=False)
    merged = policy.vla.merge_and_unload()
    sd = {k: v for k, v in merged.state_dict().items() if "value_head" not in k}
    idx_w1 = json.loads((W1 / "model.safetensors.index.json").read_text())["weight_map"]
    missing, extra = set(idx_w1) - set(sd), set(sd) - set(idx_w1)
    if missing or extra:
        raise SystemExit(f"병합 상태의 키가 W1과 다르다: 빠짐 {sorted(missing)[:5]}, 추가 {sorted(extra)[:5]}")
    tmp = w2_dir(n).with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    merged.save_pretrained(tmp, state_dict=sd, safe_serialization=True, max_shard_size="5GB")
    dst = w2_dir(n)
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    for f in W1.iterdir():  # 설정·처리기·토크나이저는 W1 그대로
        if f.is_file() and not f.name.startswith("model") and f.suffix != ".safetensors":
            shutil.copy2(f, dst / f.name)
    for f in tmp.iterdir():
        if f.name.endswith(".safetensors") or f.name == "model.safetensors.index.json":
            shutil.move(str(f), dst / f.name)
    shutil.rmtree(tmp, ignore_errors=True)
    (dst / "W2_SOURCE.json").write_text(json.dumps(dict(base=str(W1), lora_run=rid(n), demos=n, steps=STEPS, created=C.now())))
    del merged, policy
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- 단계: 병합 검사 (W1 + LoRA 대 병합 W2)
def stage_mergecheck(n: int):
    import numpy as np
    import torch
    import g2_gpu as G
    import g2_learn as L
    ref_f = W / "scratch" / f"w2_mergecheck_n{n}.pt"
    ref_f.parent.mkdir(parents=True, exist_ok=True)
    pol = G.build_policy(str(W1), C.CFG["unnorm_key"], attn="sdpa", seed=0)
    G.set_top_k(pol, C.CFG["top_k"])
    L.load_trainable(W / "runs" / rid(n) / "ckpt" / "final.pt", pol, load_optim=False)
    env = G.GpuEnv(C.CFG["train_env"], pol.vla.get_action_stats(C.CFG["unnorm_key"]), 64, seed=0)
    rec = G.rollout(pol, env, C.CFG["train_obj_set"], [G.episode_id_for("w2-mergecheck", k) for k in range(64)],
                    deterministic=False, rng_seed=777, record_obs=True)
    env.close()
    probe = L.Probe(rec["obs"], rec["instr"], rec["actions"], 64, seed=0)
    del rec
    pol.prep_training()
    probe.set_ref(pol)
    del pol
    torch.cuda.empty_cache()
    pol2 = G.build_policy(str(w2_dir(n)), C.CFG["unnorm_key"], attn="sdpa", seed=0)  # 새 LoRA는 B = 0이라 효과 없음
    pol2.prep_training()
    kl = probe.kl(pol2)
    C.write_json(PUBD / f"mergecheck_n{n}.json", dict(n=n, kl_lora_vs_merged=kl, passed=bool(kl < 0.01), checked=C.now()))
    if not kl < 0.01:
        raise SystemExit(f"병합 검사 실패: KL {kl}")


# ---------------------------------------------------------------- 단계: 학습 조건 수집 성공률
def stage_check(n: int):
    import numpy as np
    import g2_gpu as G
    base = str(W1) if n == 0 else str(w2_dir(n))
    pol = G.build_policy(base, C.CFG["unnorm_key"], attn="flash_attention_2", seed=0)
    G.set_top_k(pol, C.CFG["top_k"])
    env = G.GpuEnv(C.CFG["train_env"], pol.vla.get_action_stats(C.CFG["unnorm_key"]), 64, seed=0)
    ys = []
    for b in range(1 if a.smoke else 8):
        eids = [G.episode_id_for("w2-check", 64 * b + k) for k in range(64)]
        r = G.rollout(pol, env, C.CFG["train_obj_set"], eids, deterministic=False, rng_seed=1000 + b)
        ys += [bool(y) for y in r["Y"]]
    env.close()
    C.write_json(PUBD / f"check_n{n}.json", dict(n=n, base=base, episodes=len(ys), success=float(np.mean(ys)), checked=C.now()))


def sub(stage: str, n: int) -> int:
    cmd = [sys.executable, __file__, "--data", a.data, "--out", a.out, "--stage", stage, "--n", str(n)] + (["--smoke"] if a.smoke else [])
    with (W / "logs" / f"w2_{stage}_n{n}.log").open("a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ)).returncode


def orchestrate():
    if C.read_json(W / "public" / "w2_selection.json"):
        print("W2 already selected")
        return
    ladder = LADDER[:1] if a.smoke else LADDER
    steps = 20 if a.smoke else STEPS
    if not (PUBD / "check_n0.json").exists() and sub("check", 0) != 0:
        raise SystemExit("W1 check 실패")
    rows = []
    for n in ladder:
        st = C.read_json(W / "public" / "runs" / f"{rid(n)}.json", {}).get("status")
        if st != "done":
            cmd = [sys.executable, str(C.STUDY / "train_a1.py"), "--run-id", rid(n), "--seed", "0", "--data", str(Path(a.data) / f"n{n}"),
                   "--steps", str(steps), "--numerics", "math", "--latest-every", "500" if not a.smoke else "10"]
            with (W / "logs" / f"{rid(n)}.log").open("a") as f:
                if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode != 0:
                    raise SystemExit(f"W2 학습 실패 n={n}")
        if not (w2_dir(n) / "W2_SOURCE.json").exists() and sub("merge", n) != 0:
            raise SystemExit(f"병합 실패 n={n}")
        if not C.read_json(PUBD / f"mergecheck_n{n}.json", {}).get("passed") and sub("mergecheck", n) != 0:
            raise SystemExit(f"병합 검사 실패 n={n}")
        if not (PUBD / f"check_n{n}.json").exists() and sub("check", n) != 0:
            raise SystemExit(f"성공률 측정 실패 n={n}")
        succ = C.read_json(PUBD / f"check_n{n}.json")["success"]
        rows.append(dict(n=n, success=succ))
        C.public_log(W, f"W2 ladder n={n}: collection success {succ:.3f}")
        if succ >= TARGET:
            break
    pick = next((r for r in rows if r["success"] >= TARGET), None)
    chosen = pick or rows[-1]
    C.write_json(W / "public" / "w2_selection.json", dict(target=TARGET, ladder=rows, chosen_n=chosen["n"], reached_target=pick is not None,
                                                         w1_success=C.read_json(PUBD / "check_n0.json")["success"],
                                                         w2_dir=str(w2_dir(chosen["n"])), decided=C.now()))
    C.public_log(W, f"W2 selected n={chosen['n']} (target reached: {pick is not None})")


if __name__ == "__main__":
    if a.stage == "all":
        orchestrate()
    elif a.stage == "merge":
        stage_merge(a.n)
    elif a.stage == "mergecheck":
        stage_mergecheck(a.n)
    else:
        stage_check(a.n)
