"""G-PRE 통계 시뮬레이션 2차 (6차 외부 검토 11·8번 반영, 연구 명세 v0.12 4.5절).

1차(gpre_sim.py)와 다른 점:
  a. 필요 n 결정: 맹검 SD를 0.1 단위로 반올림한 값으로 계산한 **사전 고정 표**(SD 구간 → n)를 쓴다.
     1차는 캐시 키만 반올림하고 계산은 최초 입력 SD로 해서, 실행 순서에 따라 같은 구간의 n이 달라질 수 있었다.
  b. 순열 검정: 12,870개 분할 **전체 열거** (1차는 사전식 목록의 10개마다 하나를 골라 균일 표집이 아니었다).
  c. 최종 α 0.05 · 0.045 · 0.04에서 경계 1종 오류와 검정력.
  d. 학습 초기화를 시드 블록으로 공유할 때(C2): 블록 내 상관 ρ에 따른 Welch 대 블록 짝지은 t.
  e. 난수 시드를 1차와 다르게 둔다 (독립 확인).

사용: python gpre_sim_v2.py <출력 폴더> [반복 수]
"""
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(sys.argv[1])
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
OUT.mkdir(parents=True, exist_ok=True)
EVAL_SD = 1.8
DELTA = 5.0
rng = np.random.default_rng(20261020)


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(c - h, 4), round(c + h, 4)


# ---------------------------------------------------------------- 사전 고정 n 표
def required_n(sd, delta, power=0.8, alpha=0.05, n_max=200):
    for n in range(2, n_max + 1):
        df = 2 * n - 2
        se = sd * math.sqrt(2 / n)
        if 1 - stats.nct.cdf(stats.t.ppf(1 - alpha, df), df, delta / se) >= power:
            return n
    return n_max


SD_GRID = np.round(np.arange(0.0, 30.01, 0.1), 1)
N_TABLE = {float(s): int(min(12, max(8, required_n(max(s, 0.05), DELTA)))) for s in SD_GRID}


def n_from_blinded(s):
    key = float(min(round(s, 1), 30.0))
    return N_TABLE[key]


def blinded_sd(vals):  # Δ_plan = 0 → 보정 항 없음 (Kieser–Friede)
    return float(np.std(vals, ddof=1))


# ---------------------------------------------------------------- 자료 생성 (블록 상관 ρ)
def draw_pairs(true_diff, sd, n, rho):
    cov = sd * sd * np.array([[1.0, rho], [rho, 1.0]])
    z = rng.multivariate_normal([true_diff, 0.0], cov, size=n)
    return z[:, 0], z[:, 1]


def welch_lower(a, b, alpha):
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1) / n1, np.var(b, ddof=1) / n2
    se = math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return float(np.mean(a) - np.mean(b)) - stats.t.ppf(1 - alpha, df) * se


def paired_lower(a, b, alpha):
    d = a - b
    n = len(d)
    return float(np.mean(d)) - stats.t.ppf(1 - alpha, n - 1) * float(np.std(d, ddof=1)) / math.sqrt(n)


def procedure(true_diff, sd, rho, alphas, reps, rule="adaptive", n_fixed=8):
    """반환: {α: (Welch 기각 수, 짝지은 t 기각 수)}, 평균 n."""
    rej = {a: [0, 0] for a in alphas}
    ns = []
    for _ in range(reps):
        a4, b4 = draw_pairs(true_diff, sd, 4, rho)
        n = n_from_blinded(blinded_sd(np.concatenate([a4, b4]))) if rule == "adaptive" else n_fixed
        if n > 4:
            a_, b_ = draw_pairs(true_diff, sd, n - 4, rho)
            a, b = np.concatenate([a4, a_]), np.concatenate([b4, b_])
        else:
            a, b = a4, b4
        for al in alphas:
            rej[al][0] += welch_lower(a, b, al) > -DELTA
            rej[al][1] += paired_lower(a, b, al) > -DELTA
        ns.append(n)
    return rej, float(np.mean(ns))


results = dict(reps=REPS, seed=20261020, delta=DELTA, eval_sd=EVAL_SD,
               n_table={f"{s:.1f}": n for s, n in N_TABLE.items() if s <= 15}, procedure=[], permutation=[])
ALPHAS = (0.05, 0.045, 0.04)

print("== 1. 적응형 절차 (4+4 → 맹검 SD → 사전 고정 표 n 8~12), δ 5, Δ_plan 0", flush=True)
for sigma in (3.0, 4.0, 5.0, 6.0):
    sd = math.sqrt(sigma ** 2 + EVAL_SD ** 2)
    for rho in (0.0, 0.3, 0.6, -0.3):
        reps = REPS if rho == 0.0 else REPS // 4
        t1, mn = procedure(-DELTA, sd, rho, ALPHAS, reps)
        pw, mn0 = procedure(0.0, sd, rho, ALPHAS, reps)
        row = dict(sigma=sigma, rho=rho, reps=reps, mean_n_at_0=round(mn0, 2))
        for al in ALPHAS:
            for j, name in enumerate(("welch", "paired")):
                row[f"type1_{name}_a{al}"] = t1[al][j] / reps
                row[f"type1_{name}_a{al}_ci95"] = wilson(t1[al][j], reps)
                row[f"power0_{name}_a{al}"] = pw[al][j] / reps
        if rho == 0.0:  # 고정 n 대조
            for nf in (8, 12):
                f1, _ = procedure(-DELTA, sd, 0.0, (0.05,), reps, rule="fixed", n_fixed=nf)
                row[f"type1_welch_fixed_n{nf}"] = f1[0.05][0] / reps
        results["procedure"].append(row)
        print(f"  σ {sigma:.0f} ρ {rho:+.1f} mean_n {mn0:5.2f} | T1 Welch " +
              " ".join(f"α{al}:{t1[al][0] / reps:.4f}" for al in ALPHAS) +
              " | T1 paired " + " ".join(f"α{al}:{t1[al][1] / reps:.4f}" for al in ALPHAS) +
              " | pow Welch/paired α0.05 " + f"{pw[0.05][0] / reps:.3f}/{pw[0.05][1] / reps:.3f}"
              + (f" | fixed n8 {row['type1_welch_fixed_n8']:.4f} n12 {row['type1_welch_fixed_n12']:.4f}"
                 if rho == 0.0 else ""), flush=True)

# ---------------------------------------------------------------- 2. 순열 검정, 전체 열거
print("\n== 2. 비열등성 순열 검정 (δ 이동 + studentized Welch, 8 대 8, 12,870개 분할 전체)", flush=True)
MASKS = np.zeros((12870, 16), bool)
for k, c in enumerate(combinations(range(16), 8)):
    MASKS[k, list(c)] = True


def t_all(z):
    zz = np.broadcast_to(z, MASKS.shape)
    x = zz[MASKS].reshape(-1, 8)
    y = zz[~MASKS].reshape(-1, 8)
    return (x.mean(1) - y.mean(1)) / np.sqrt(x.var(1, ddof=1) / 8 + y.var(1, ddof=1) / 8)


for sds in ((4.4, 4.4), (3.0, 6.0), (6.0, 3.0)):
    reps = max(2000, REPS // 4)
    k = 0
    for _ in range(reps):
        a = rng.normal(-DELTA, sds[0], 8) + DELTA
        b = rng.normal(0.0, sds[1], 8)
        z = np.concatenate([a, b])
        ts = t_all(z)
        t_obs = ts[0]  # MASKS[0] = 앞 8개 = 관측 배정 (전체 열거라 관측 분할이 분모에 포함)
        p = float(np.mean(ts >= t_obs))
        k += p <= 0.05
    results["permutation"].append(dict(sds=sds, reps=reps, type1=k / reps, ci95=wilson(k, reps)))
    print(f"  SDs {sds}: type I {k / reps:.4f} CI95 {wilson(k, reps)} ({reps} reps)", flush=True)

(OUT / "gpre_sim_v2_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("saved", OUT / "gpre_sim_v2_results.json")
