"""P3 분석: 수집 성공률, 물체·테이블별 90% 구간, 0-성공 라운드, 집기만 성공, 회복 성공, 설정 간 짝지은 차이,
그리고 사전 고정 규칙 두 개의 판정.

규칙 1 (v0.9 C4·P3, top-k 처리 방식):
  같은 초기 상태에서 p̂(T1_K50) − p̂(T1_K0)의 90% 구간(짝지은 비율 차, Newcombe-Wald 근사) 하한이 3%p를 넘거나,
  K0만 웜업 전환 규칙에 걸리면 → K50, 그 밖이면 → K0.
규칙 2 (3.7절 웜업 전환): 고른 방식의 수집 성공률로
  (a) 전체 90% Clopper–Pearson 상한 < 5%, 또는
  (b) 물체별 90% 상한 < 8%인 물체 8종 이상, 또는 테이블별로 같은 조건 → 주 조건을 W2로.
P3를 Windows(본실험과 다른 렌더러·OS·환경 수)에서 쟀으므로 결과는 잠정치이고 G2에서 다시 잰다.

사용: python p3_analyze.py <p3_results.jsonl> [out_json]
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

rows = [json.loads(l) for l in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
out_json = Path(sys.argv[2]) if len(sys.argv) > 2 else None
by = defaultdict(list)
for r in rows:
    by[r["setting"]].append(r)


def cp90(k, n):
    lo = stats.beta.ppf(0.05, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.95, k + 1, n - k) if k < n else 1.0
    return lo, hi


def recovered(r):
    """회복 성공: 마지막에 성공했고, 집기 구간이 둘 이상(집었다가 3스텝 이상 놓친 뒤 다시 집음)."""
    if not r["success"]:
        return False
    segs = re.findall(r"1+", r["grasp_trace"])
    gaps = re.findall(r"(?<=1)0{3,}(?=1)", r["grasp_trace"])
    return len(segs) >= 2 and len(gaps) >= 1


def summarize(name):
    rs = by[name]
    n = len(rs)
    k = sum(r["success"] for r in rs)
    lo, hi = cp90(k, n)
    res = dict(setting=name, n=n, successes=k, rate=k / n, cp90=(lo, hi))
    for key, label in (("obj", "obj_name"), ("table", "table")):
        groups = defaultdict(list)
        for r in rs:
            groups[r[label] if key == "obj" else r["table"]].append(r["success"])
        per = {}
        for g, v in sorted(groups.items(), key=lambda x: str(x[0])):
            kk, nn = sum(v), len(v)
            per[str(g)] = dict(n=nn, k=kk, rate=kk / nn, cp90_upper=cp90(kk, nn)[1])
        res[f"per_{key}"] = per
        res[f"n_{key}_upper_below_8pct"] = sum(1 for v in per.values() if v["cp90_upper"] < 0.08)
    # 라운드 = 실행 순서대로 64개씩 (실제 수집 라운드와 같은 크기)
    rounds = [rs[i:i + 64] for i in range(0, n - n % 64, 64)]
    res["zero_success_rounds_observed"] = sum(1 for g in rounds if not any(r["success"] for r in g))
    res["n_rounds"] = len(rounds)
    res["p_zero_success_round_binomial"] = (1 - k / n) ** 64 if n else None
    res["expected_successes_per_round"] = 64 * k / n if n else None
    res["grasp_only_failures"] = sum(1 for r in rs if r["grasped_latch"] and not r["success"])
    res["recovery_successes"] = sum(1 for r in rs if recovered(r))
    res["success_any_step"] = sum(r["success_any_step"] for r in rs)
    res["success_and_grasp"] = sum(r["success_and_grasp"] for r in rs)
    return res


def paired_diff(a, b):
    """같은 p3_index끼리 짝지어 p̂(a) − p̂(b)와 90% 구간 (짝지은 비율 차, Wald)."""
    ia = {r["p3_index"]: r["success"] for r in by[a]}
    ib = {r["p3_index"]: r["success"] for r in by[b]}
    keys = sorted(set(ia) & set(ib))
    n = len(keys)
    n10 = sum(1 for i in keys if ia[i] and not ib[i])
    n01 = sum(1 for i in keys if ib[i] and not ia[i])
    d = (n10 - n01) / n
    var = ((n10 + n01) / n - d ** 2) / n
    z = stats.norm.ppf(0.95)
    return dict(pair=f"{a} - {b}", n=n, a_only=n10, b_only=n01, diff=d,
                ci90=(d - z * math.sqrt(var), d + z * math.sqrt(var)))


def warmup_switch(s):
    a = s["cp90"][1] < 0.05
    b_obj = s["n_obj_upper_below_8pct"] >= 8
    b_tab = s["n_table_upper_below_8pct"] >= 8
    return dict(overall_upper_below_5pct=bool(a), objects_rule=bool(b_obj), tables_rule=bool(b_tab), switch_to_W2=bool(a or b_obj or b_tab))


summ = {name: summarize(name) for name in by}
print(f"{'setting':8} {'n':>5} {'SR%':>6} {'90% CP':>15} {'0-succ rounds':>14} {'P(0-succ) binom':>16} "
      f"{'grasp-only fail':>15} {'recovery succ':>13} {'obj<8%':>6} {'tab<8%':>6}")
for name, s in summ.items():
    print(f"{name:8} {s['n']:5} {100 * s['rate']:6.1f} {100 * s['cp90'][0]:6.1f}-{100 * s['cp90'][1]:5.1f}% "
          f"{s['zero_success_rounds_observed']:>8}/{s['n_rounds']:<5} {s['p_zero_success_round_binomial']:16.2e} "
          f"{s['grasp_only_failures']:15} {s['recovery_successes']:13} {s['n_obj_upper_below_8pct']:6} {s['n_table_upper_below_8pct']:6}")

diffs = []
for a, b in (("T1_K50", "T1_K0"), ("T06_K50", "T1_K50")):
    if a in by and b in by:
        d = paired_diff(a, b)
        diffs.append(d)
        print(f"paired {d['pair']}: {100 * d['diff']:+.1f}%p  90% CI ({100 * d['ci90'][0]:+.1f}, {100 * d['ci90'][1]:+.1f})  "
              f"discordant {d['a_only']}/{d['b_only']}  n={d['n']}")

decision = {}
if "T1_K0" in summ and "T1_K50" in summ:
    sw0, sw50 = warmup_switch(summ["T1_K0"]), warmup_switch(summ["T1_K50"])
    d = paired_diff("T1_K50", "T1_K0")
    use_k50 = d["ci90"][0] > 0.03 or (sw0["switch_to_W2"] and not sw50["switch_to_W2"])
    chosen = "T1_K50" if use_k50 else "T1_K0"
    sw = sw50 if use_k50 else sw0
    decision = dict(rule1_topk=("K50" if use_k50 else "K0"), rule1_basis=dict(paired_ci90_lower=d["ci90"][0],
                    k0_triggers_switch=sw0["switch_to_W2"], k50_triggers_switch=sw50["switch_to_W2"]),
                    rule2_warmup=("W2" if sw["switch_to_W2"] else "W1"), rule2_basis=sw, chosen_setting=chosen,
                    provisional=True, note="Windows native measurement; re-measure at G2 (spec 3.7)")
    print(f"\nRULE 1 (top-k): {decision['rule1_topk']}  (paired CI lower {100 * d['ci90'][0]:+.1f}%p vs +3%p; "
          f"K0 triggers W2 rule: {sw0['switch_to_W2']}, K50: {sw50['switch_to_W2']})")
    print(f"RULE 2 (warm-up): main condition {decision['rule2_warmup']}  basis {sw}  [provisional]")

if out_json:
    out_json.write_text(json.dumps(dict(summary=summ, paired=diffs, decision=decision), indent=2, ensure_ascii=False),
                        encoding="utf-8")
