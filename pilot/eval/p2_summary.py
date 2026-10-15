"""P2 결과 집계: 체크포인트 × 과제별 진행 상황, 성공률, 6.4절 재현 목표 대비 판정.

사용: python p2_summary.py <p2_results.jsonl>
실행 도중에도 돌릴 수 있다 (끝난 에피소드만 센다).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

N_TARGET = 3 * 64  # 시드 3 x 64 에피소드

# 6.4절 재현 목표: (원 논문 평균 %, 허용 범위 하한, 상한)
TARGETS = {
    ("sft_16k", "Carrot-v1/train"): (78.1, 68.1, 88.1),
    ("sft_16k", "Carrot-v1/test"): (45.3, 35.3, 55.3),
    ("sft_16k", "EEPose-v1/test"): (33.9, 23.5, 44.3),
    ("sft_16k", "PositionChangeTo-v1/test"): (28.6, 18.6, 38.6),
    ("rl", "Carrot-v1/train"): (93.8, 83.8, 100.0),
    ("rl", "Carrot-v1/test"): (71.4, 61.4, 81.4),
    ("rl", "EEPose-v1/test"): (79.7, 66.9, 92.5),
    ("rl", "PositionChangeTo-v1/test"): (74.5, 64.5, 84.5),
}
TASK_ORDER = ["Carrot-v1/train", "Carrot-v1/test", "EEPose-v1/test", "PositionChangeTo-v1/test"]
CKPT_ORDER = ["rl", "sft_16k", "warmup"]


def short_ckpt(path: str) -> str:
    return Path(path).name.replace("openvla-7b-rlvla-", "")


def main():
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"no results yet: {path}")
        return
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    cells = defaultdict(list)
    secs = []
    for r in rows:
        task = f"{r['env_id'].replace('PutOnPlateInScene25', '')}/{r['obj_set']}"
        cells[(short_ckpt(r["ckpt"]), task)].append(r)
        secs.append(r["sec"])

    total_target = len(CKPT_ORDER) * len(TASK_ORDER) * N_TARGET
    mean_sec = sum(secs) / len(secs) if secs else float("nan")
    remaining = total_target - len(rows)
    print(f"episodes done: {len(rows)}/{total_target}  mean {mean_sec:.1f}s/episode  "
          f"remaining ~{remaining * mean_sec / 3600:.1f} h\n")

    hdr = f"{'ckpt':8} {'task':26} {'n':>7} {'SR%':>6} {'±SE':>5} {'seed0/1/2 SR%':>18} {'any%':>6} {'S&G%':>6}  target (range)  verdict"
    print(hdr)
    print("-" * len(hdr))
    for ck in CKPT_ORDER:
        for task in TASK_ORDER:
            rs = cells.get((ck, task), [])
            n = len(rs)
            if n == 0:
                print(f"{ck:8} {task:26} {0:>3}/{N_TARGET}")
                continue
            sr = 100 * sum(r["success"] for r in rs) / n
            se = 100 * math.sqrt(max(sr / 100 * (1 - sr / 100), 1e-12) / n)
            by_seed = []
            for s in (0, 1, 2):
                rss = [r for r in rs if r["seed"] == s]
                by_seed.append(f"{100 * sum(r['success'] for r in rss) / len(rss):.0f}" if rss else "-")
            any_ = 100 * sum(r["success_any_step"] for r in rs) / n
            sg = 100 * sum(r["success_and_grasp_last"] for r in rs) / n
            tgt = TARGETS.get((ck, task))
            if tgt:
                tg = f"{tgt[0]:.1f} ({tgt[1]:.1f}-{tgt[2]:.1f})"
                verdict = ("in range" if tgt[1] <= sr <= tgt[2] else "OUT") + ("" if n >= N_TARGET else " (partial)")
            else:
                tg, verdict = "-", "(no target)"
            print(f"{ck:8} {task:26} {n:>3}/{N_TARGET} {sr:6.1f} {se:5.1f} {'/'.join(by_seed):>18} {any_:6.1f} {sg:6.1f}  {tg:15} {verdict}")
        print()

    # 실행 축 방향 점검 (6.3절 P2 통과 기준: 실행 2과제에서 RL > SFT)
    for task in ("EEPose-v1/test", "PositionChangeTo-v1/test"):
        a, b = cells.get(("rl", task), []), cells.get(("sft_16k", task), [])
        if a and b:
            ra = 100 * sum(r["success"] for r in a) / len(a)
            rb = 100 * sum(r["success"] for r in b) / len(b)
            print(f"direction {task}: RL {ra:.1f} vs SFT {rb:.1f} -> {'RL > SFT' if ra > rb else 'RL <= SFT (check!)'}")
    print("\nSR = 원 평가 기준(마지막 스텝 success). any = 한 번이라도 성공. S&G = 마지막 스텝 success ∧ grasp (v0.9 Y 후보).")


if __name__ == "__main__":
    main()
