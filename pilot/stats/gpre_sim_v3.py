"""G-PRE 통계 시뮬레이션 3차 (G-PRE 초안 T1, 사전 고정: GPRE_simulation_v3_prereg.md).

2차(gpre_sim_v2.py)와 다른 점:
  a. 주 분석 = 시드 블록 짝지은 t (블록 차이 d_b = Y(A2+) - Y(A3_0), H0: E[d] <= -delta), 최종 단측 alpha 0.045.
  b. 맹검 재추정을 짝지은 구조에 맞춘다. 후보 두 가지:
       P (주 후보): 블록 차이 제곱 평균 s_d^2 = sum(d_b^2)/4. 갈래 이름을 바꿔도 d^2는 같아서 맹검이 유지된다.
                    Delta_plan = 0 가정 (E[d^2] = sigma_d^2 + Delta^2).
       W (대조):    2차 방식의 합친 분산 s^2 (8개 값, 이름 무시)을 sigma_d^2 = 2 s^2 로 환산 (rho = 0 가정).
     어느 쪽이든 SD_d를 0.1 단위로 반올림해 사전 고정 표(짝지은 t, alpha 0.045, 검정력 0.8)에서 n을 8~12로 정한다.
  c. 평가 잡음: T3 측정값 1.4%p (주), 1.47 (민감도), 1.68 (성공률 50% 환산).
     자료 모형: a = Delta + sigma z1 + e1, b = sigma z2 + e2, corr(z1, z2) = rho (학습 몫에만), e ~ N(0, eval^2) 독립.
     (2차는 rho를 평가 잡음 포함 전체에 걸었다.)
  d. 민감도: Welch (같은 자료), 블록 안 부호 뒤집기 순열 검정 (delta 이동 뒤, studentized, 2^n 전체 열거).
  e. 난수 시드 20261106 (1·2차와 다름).

사용: python gpre_sim_v3.py <출력 폴더> [반복 수] [순열 반복 수]
"""
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(sys.argv[1])
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
PREPS = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
OUT.mkdir(parents=True, exist_ok=True)
DELTA = 5.0
ALPHA = 0.045
ALPHAS = (0.05, 0.045, 0.04)
N_MIN, N_MAX, N_STAGE1 = 8, 12, 4
SIGMAS = (3.0, 4.0, 5.0, 6.0)
RHOS = (-0.3, 0.0, 0.3, 0.6)
EVALS = (1.4, 1.47, 1.68)
rng = np.random.default_rng(20261106)


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(c - h, 4), round(c + h, 4)


# ---------------------------------------------------------------- 사전 고정 n 표 (짝지은 t)
def required_n_paired(sd_d, delta=DELTA, power=0.8, alpha=ALPHA, n_max=200):
    for n in range(2, n_max + 1):
        df = n - 1
        ncp = delta / (sd_d / math.sqrt(n))
        if 1 - stats.nct.cdf(stats.t.ppf(1 - alpha, df), df, ncp) >= power:
            return n
    return n_max


SD_GRID = np.round(np.arange(0.0, 40.01, 0.1), 1)
N_TABLE = {float(s): int(min(N_MAX, max(N_MIN, required_n_paired(max(s, 0.05))))) for s in SD_GRID}
N_ARR = np.array([N_TABLE[float(s)] for s in SD_GRID])


def n_lookup(sd_d):
    idx = np.clip(np.round(sd_d * 10).astype(int), 0, len(SD_GRID) - 1)
    return N_ARR[idx]


# ---------------------------------------------------------------- 자료와 검정 (벡터화)
def draw(true_diff, sigma, rho, ev, reps):
    z1 = rng.standard_normal((reps, N_MAX))
    z2 = rho * z1 + math.sqrt(1 - rho * rho) * rng.standard_normal((reps, N_MAX))
    a = true_diff + sigma * z1 + ev * rng.standard_normal((reps, N_MAX))
    b = sigma * z2 + ev * rng.standard_normal((reps, N_MAX))
    return a, b


def choose_n(a, b, rule):
    a4, b4 = a[:, :N_STAGE1], b[:, :N_STAGE1]
    if rule == "P":
        sd_d = np.sqrt(np.mean((a4 - b4) ** 2, axis=1))
    elif rule == "W":
        sd_d = math.sqrt(2) * np.std(np.concatenate([a4, b4], axis=1), axis=1, ddof=1)
    else:
        return np.full(a.shape[0], int(rule))
    return n_lookup(sd_d)


def lower_bounds(a, b, ns, alpha):
    """n별로 앞 n개 블록을 써서 짝지은 t와 Welch의 단측 하한."""
    lp = np.empty(a.shape[0])
    lw = np.empty(a.shape[0])
    for n in np.unique(ns):
        m = ns == n
        x, y = a[m, :n], b[m, :n]
        d = x - y
        lp[m] = d.mean(1) - stats.t.ppf(1 - alpha, n - 1) * d.std(1, ddof=1) / math.sqrt(n)
        v1, v2 = x.var(1, ddof=1) / n, y.var(1, ddof=1) / n
        df = (v1 + v2) ** 2 / (v1 ** 2 / (n - 1) + v2 ** 2 / (n - 1))
        lw[m] = (x.mean(1) - y.mean(1)) - stats.t.ppf(1 - alpha, df) * np.sqrt(v1 + v2)
    return lp, lw


def run_cell(sigma, rho, ev, rule, reps):
    row = dict(sigma=sigma, rho=rho, eval_sd=ev, rule=rule, reps=reps)
    for tag, diff in (("t1", -DELTA), ("pow0", 0.0)):
        a, b = draw(diff, sigma, rho, ev, reps)
        ns = choose_n(a, b, rule)
        row[f"mean_n_{tag}"] = round(float(ns.mean()), 3)
        row[f"n_dist_{tag}"] = {int(k): int(v) for k, v in zip(*np.unique(ns, return_counts=True))}
        for al in ALPHAS:
            lp, lw = lower_bounds(a, b, ns, al)
            kp, kw = int((lp > -DELTA).sum()), int((lw > -DELTA).sum())
            row[f"{tag}_paired_a{al}"] = kp / reps
            row[f"{tag}_welch_a{al}"] = kw / reps
            if tag == "t1":
                row[f"{tag}_paired_a{al}_ci95"] = wilson(kp, reps)
    return row


# ---------------------------------------------------------------- 부호 뒤집기 순열 (민감도)
SIGNS = {n: np.array(list(product((1.0, -1.0), repeat=n))) for n in range(N_MIN, N_MAX + 1)}  # 첫 행 = 전부 +1 (관측)


def signflip_reject(d_shift, alpha):
    n = d_shift.shape[0]
    s = SIGNS[n] * d_shift  # (2^n, n)
    t = s.mean(1) / (s.std(1, ddof=1) / math.sqrt(n))
    t = np.nan_to_num(t, nan=-np.inf)
    return float(np.mean(t >= t[0])) <= alpha


def run_perm(sigma, rho, ev, reps):
    a, b = draw(-DELTA, sigma, rho, ev, reps)
    ns = choose_n(a, b, "P")
    k = 0
    for r in range(reps):
        n = int(ns[r])
        k += signflip_reject(a[r, :n] - b[r, :n] + DELTA, ALPHA)
    return dict(sigma=sigma, rho=rho, eval_sd=ev, reps=reps, type1_a0045=k / reps, ci95=wilson(k, reps))


if __name__ == "__main__":
    res = dict(reps=REPS, perm_reps=PREPS, seed=20261106, delta=DELTA, alpha_final=ALPHA,
               n_table={f"{s:.1f}": n for s, n in N_TABLE.items() if 3.0 <= s <= 15.0},
               n_table_breaks={}, procedure=[], fixed=[], permutation=[])
    prev = None
    for s in SD_GRID:
        n = N_TABLE[float(s)]
        if n != prev:
            res["n_table_breaks"][f"{s:.1f}"] = n
            prev = n
    print("n 표 (SD_d 시작값 -> n):", res["n_table_breaks"], flush=True)

    print("== 1. 적응형 절차 (4블록 -> 맹검 SD_d -> n 8~12)", flush=True)
    for ev, sigma, rho, rule in product(EVALS, SIGMAS, RHOS, ("P", "W")):
        if ev != 1.4 and rule == "W":
            continue
        row = run_cell(sigma, rho, ev, rule, REPS)
        res["procedure"].append(row)
        print(f"  eval {ev:.2f} σ {sigma:.0f} ρ {rho:+.1f} {rule} n̄ {row['mean_n_pow0']:5.2f} | T1 paired "
              + " ".join(f"{al}:{row[f't1_paired_a{al}']:.4f}" for al in ALPHAS)
              + f" | T1 Welch .045 {row['t1_welch_a0.045']:.4f} | pow0 paired/Welch .045 "
              f"{row['pow0_paired_a0.045']:.3f}/{row['pow0_welch_a0.045']:.3f}", flush=True)

    print("== 2. 고정 n 대조 (eval 1.4)", flush=True)
    for sigma, rho, nf in product(SIGMAS, (0.0, -0.3), (8, 12)):
        row = run_cell(sigma, rho, 1.4, str(nf), REPS)
        res["fixed"].append(row)
        print(f"  σ {sigma:.0f} ρ {rho:+.1f} n {nf} | T1 paired .045 {row['t1_paired_a0.045']:.4f}"
              f" | pow0 {row['pow0_paired_a0.045']:.3f}", flush=True)

    print("== 3. 부호 뒤집기 순열 (적응형 P, eval 1.4, α 0.045)", flush=True)
    for sigma, rho in product((3.0, 6.0), (0.0, -0.3)):
        row = run_perm(sigma, rho, 1.4, PREPS)
        res["permutation"].append(row)
        print(f"  σ {sigma:.0f} ρ {rho:+.1f} | T1 {row['type1_a0045']:.4f} {row['ci95']}", flush=True)

    (OUT / "gpre_sim_v3_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", OUT / "gpre_sim_v3_results.json")
