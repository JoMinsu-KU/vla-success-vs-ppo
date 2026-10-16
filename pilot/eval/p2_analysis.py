"""P2 세부 분석: 시드 간 차이의 원인, 체크포인트 짝지은 비교, 성공 기준 차이, 소요 시간.

사용: python p2_analysis.py <p2_results.jsonl>
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
for r in rows:
    r["ck"] = Path(r["ckpt"]).name.replace("openvla-7b-rlvla-", "")
    r["task"] = f"{r['env_id'].replace('PutOnPlateInScene25', '')}/{r['obj_set']}"
TASKS = ["Carrot-v1/train", "Carrot-v1/test", "EEPose-v1/test", "PositionChangeTo-v1/test"]
CKS = ["rl", "sft_16k", "warmup"]
idx = {(r["ck"], r["task"], r["seed"], r["episode"]): r for r in rows}


def sr(rs, key="success"):
    return 100 * sum(r[key] for r in rs) / len(rs) if rs else float("nan")


print("== 1. 시드 간 차이: 관측 SD vs 이항 잡음만 있을 때의 SD (64 에피소드)")
print(f"{'ckpt':8} {'task':26} {'seed SRs':>16} {'obs SD':>7} {'binom SD':>8}")
for ck in CKS:
    for t in TASKS:
        s = [sr([r for r in rows if r["ck"] == ck and r["task"] == t and r["seed"] == k]) for k in (0, 1, 2)]
        m = sum(s) / 3
        sd = math.sqrt(sum((x - m) ** 2 for x in s) / 2)
        bsd = 100 * math.sqrt((m / 100) * (1 - m / 100) / 64)
        print(f"{ck:8} {t:26} {'/'.join(f'{x:.0f}' for x in s):>16} {sd:7.1f} {bsd:8.1f}")

print("\n== 2. 같은 초기 상태에서의 체크포인트 비교 (에피소드 짝: 같은 seed·episode)")
print(f"{'task':26} {'pair':14} {'diff %p':>8} {'both':>5} {'A only':>6} {'B only':>6} {'neither':>7} {'McNemar p':>9}")
for t in TASKS:
    for a, b in (("rl", "sft_16k"), ("rl", "warmup"), ("sft_16k", "warmup")):
        n11 = n10 = n01 = n00 = 0
        for s in (0, 1, 2):
            for k in range(64):
                x, y = idx[(a, t, s, k)]["success"], idx[(b, t, s, k)]["success"]
                n11 += x and y; n10 += x and not y; n01 += (not x) and y; n00 += (not x) and (not y)
        n = n10 + n01
        # 정확 이항 양측 McNemar
        from math import comb
        kmin = min(n10, n01)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(kmin + 1)) / 2 ** n) if n else 1.0
        print(f"{t:26} {a + '-' + b:14} {100 * (n10 - n01) / 192:8.1f} {n11:5} {n10:6} {n01:6} {n00:7} {p:9.2g}")

print("\n== 3. 물체별 성공률 (Carrot-v1/test, 새 물체 16~24): 시드 차이의 원인 확인")
by_obj = defaultdict(lambda: defaultdict(list))
names = {}
for r in rows:
    if r["task"] == "Carrot-v1/test":
        by_obj[r["carrot_ids"]][r["ck"]].append(r["success"])
        names[r["carrot_ids"]] = r["instruction"].replace("put ", "").replace(" on plate", "")
seed_obj = defaultdict(lambda: defaultdict(int))
for r in rows:
    if r["task"] == "Carrot-v1/test" and r["ck"] == "rl":
        seed_obj[r["seed"]][r["carrot_ids"]] += 1
print(f"{'id':>3} {'object':16} {'n/ckpt':>6} " + " ".join(f"{c:>8}" for c in CKS) + "   count in seed0/1/2")
for o in sorted(by_obj):
    n = len(by_obj[o]["rl"])
    print(f"{o:>3} {names[o][:16]:16} {n:>6} " + " ".join(f"{100 * sum(by_obj[o][c]) / len(by_obj[o][c]):8.0f}" for c in CKS)
          + "   " + "/".join(str(seed_obj[s][o]) for s in (0, 1, 2)))

print("\n== 4. 성공 기준별 차이 (%p, 전체 192 에피소드)")
print(f"{'ckpt':8} {'task':26} {'last':>6} {'any-last':>9} {'last-S&G':>9}")
for ck in CKS:
    for t in TASKS:
        rs = [r for r in rows if r["ck"] == ck and r["task"] == t]
        print(f"{ck:8} {t:26} {sr(rs):6.1f} {sr(rs, 'success_any_step') - sr(rs):9.1f} {sr(rs) - sr(rs, 'success_and_grasp_last'):9.1f}")

print("\n== 5. 소요 시간")
for ck in CKS:
    rs = [r for r in rows if r["ck"] == ck]
    tot = sum(r["sec"] for r in rs)
    pol = sum(r["sec_policy"] for r in rs)
    print(f"{ck:8} episodes {len(rs)}  total {tot / 3600:.2f} h  mean {tot / len(rs):.1f} s/ep  policy share {100 * pol / tot:.0f}%")
