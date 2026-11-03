"""G2 S3 집계: g2_s3_eval_original.sh 결과(stats.yaml)를 조건 16개 × 체크포인트별로 모은다.
원 기준 = 마지막 스텝 success (info["episode"]), Y = success ∧ is_src_obj_grasped(래치).
RL − SFT 격차, 6.4절 재현 목표 대조, (같은 시드에서 초기 상태가 같으면) 정확 이항 McNemar.
사용: python g2_s3_summary.py <S3 출력 폴더>   → <폴더>/g2_s3_summary.json, 표 출력
"""
import json
import math
import sys
from pathlib import Path

import yaml

OUT = Path(sys.argv[1])
CONDS = [  # (라벨, env_id, obj_set, 축)
    ("ID (Carrot train)", "PutOnPlateInScene25Carrot-v1", "train", "ID"),
    ("VisionImage", "PutOnPlateInScene25VisionImage-v1", "test", "vision"),
    ("VisionTexture03", "PutOnPlateInScene25VisionTexture03-v1", "test", "vision"),
    ("VisionTexture05", "PutOnPlateInScene25VisionTexture05-v1", "test", "vision"),
    ("VisionWhole03", "PutOnPlateInScene25VisionWhole03-v1", "test", "vision"),
    ("VisionWhole05", "PutOnPlateInScene25VisionWhole05-v1", "test", "vision"),
    ("New object (Carrot test)", "PutOnPlateInScene25Carrot-v1", "test", "semantic"),
    ("Plate", "PutOnPlateInScene25Plate-v1", "test", "semantic"),
    ("Instruct", "PutOnPlateInScene25Instruct-v1", "test", "semantic"),
    ("MultiCarrot test", "PutOnPlateInScene25MultiCarrot-v1", "test", "semantic"),
    ("MultiPlate test", "PutOnPlateInScene25MultiPlate-v1", "test", "semantic"),
    ("Position", "PutOnPlateInScene25Position-v1", "test", "execution"),
    ("EEPose", "PutOnPlateInScene25EEPose-v1", "test", "execution"),
    ("PositionChangeTo", "PutOnPlateInScene25PositionChangeTo-v1", "test", "execution"),
    ("MultiCarrot train", "PutOnPlateInScene25MultiCarrot-v1", "train", "train-variant"),
    ("MultiPlate train", "PutOnPlateInScene25MultiPlate-v1", "train", "train-variant"),
]
# 6.4절 재현 목표 (원 논문 값, 허용 범위) — P2_results.md와 같은 표
TARGETS = {("rl", "ID (Carrot train)"): (93.8, 83.8, 100), ("rl", "New object (Carrot test)"): (71.4, 61.4, 81.4),
           ("rl", "EEPose"): (79.7, 66.9, 92.5), ("rl", "PositionChangeTo"): (74.5, 64.5, 84.5),
           ("sft", "ID (Carrot train)"): (78.1, 68.1, 88.1), ("sft", "New object (Carrot test)"): (45.3, 35.3, 55.3),
           ("sft", "EEPose"): (33.9, 23.5, 44.3), ("sft", "PositionChangeTo"): (28.6, 18.6, 38.6)}
PAPER_GAP = {"ID (Carrot train)": 15.7, "New object (Carrot test)": 26.1, "EEPose": 45.8, "PositionChangeTo": 45.9}
P2_4090 = {("rl", "ID (Carrot train)"): 87.0, ("rl", "New object (Carrot test)"): 69.3, ("rl", "EEPose"): 78.6,
           ("rl", "PositionChangeTo"): 79.7, ("sft", "ID (Carrot train)"): 85.4, ("sft", "New object (Carrot test)"): 55.7,
           ("sft", "EEPose"): 33.9, ("sft", "PositionChangeTo"): 30.7}


def load(name, env, split, seed):
    d = OUT / "runs" / name / env / f"s{seed}"
    f = sorted(d.glob(f"wandb/offline-run-*/glob/vis_0_{split}/stats.yaml"))
    if not f:
        return None
    s = yaml.safe_load(f[-1].read_text())
    li = s["last_info"]
    n = len(li)
    succ = [bool(li[i]["success"]) for i in range(n)]
    grasp = [bool(li[i]["is_src_obj_grasped"]) for i in range(n)]
    return dict(success=succ, Y=[a and b for a, b in zip(succ, grasp)], instr=[s["instruction"][i] for i in range(n)])


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


res = {}
for lab, env, split, axis in CONDS:
    row = dict(env_id=env, obj_set=split, axis=axis)
    per = {}
    for name in ("rl", "sft", "warmup"):
        seeds = {s: load(name, env, split, s) for s in (0, 1, 2)}
        seeds = {s: v for s, v in seeds.items() if v}
        if not seeds:
            continue
        allS = sum((v["success"] for v in seeds.values()), [])
        allY = sum((v["Y"] for v in seeds.values()), [])
        p = sum(allS) / len(allS)
        per[name] = seeds
        row[name] = dict(n=len(allS), seeds=sorted(seeds), success=round(100 * p, 1),
                         se=round(100 * math.sqrt(p * (1 - p) / len(allS)), 1),
                         per_seed=[round(100 * sum(v["success"]) / len(v["success"]), 1) for v in seeds.values()],
                         Y=round(100 * sum(allY) / len(allY), 1))
        t = TARGETS.get((name, lab))
        if t:
            row[name].update(paper=t[0], range=[t[1], t[2]], in_range=t[1] <= 100 * p <= t[2],
                             p2_4090=P2_4090.get((name, lab)))
    if "rl" in row and "sft" in row:
        row["gap_rl_minus_sft"] = round(row["rl"]["success"] - row["sft"]["success"], 1)
        row["paper_gap"] = PAPER_GAP.get(lab)
        common = sorted(set(per["rl"]) & set(per["sft"]))
        same_init = all(per["rl"][s]["instr"] == per["sft"][s]["instr"] for s in common)
        b = c = 0
        for s in common:
            for x, y in zip(per["rl"][s]["success"], per["sft"][s]["success"]):
                b += x and not y
                c += y and not x
        row["paired"] = dict(seeds=common, same_instructions_all_seeds=same_init, rl_only=b, sft_only=c,
                             mcnemar_p=mcnemar(b, c) if same_init else None)
    res[lab] = row

(OUT / "g2_s3_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"{'condition':26s} {'RL':>14s} {'SFT':>14s} {'WU':>14s} {'gap':>6s} {'paper':>6s}  McNemar")
for lab, r in res.items():
    f = lambda k: (f"{r[k]['success']:5.1f}±{r[k]['se']:.1f} n{r[k]['n']}" if k in r else "-")  # noqa: E731
    pp = r.get("paired", {})
    print(f"{lab:26s} {f('rl'):>14s} {f('sft'):>14s} {f('warmup'):>14s} {str(r.get('gap_rl_minus_sft', '-')):>6s} "
          f"{str(r.get('paper_gap') or ''):>6s}  {pp.get('mcnemar_p', '')}")
