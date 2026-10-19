"""G-PRE 통계 시뮬레이션 (연구 명세 v0.9, 4.5·4.6절).

1. 주 비교(비열등성) 절차 전체: 갈래당 시드 4개 → 맹검 합동 분산(Kieser–Friede 보정) → n 결정(8~12) →
   추가 수집 → 최종 Welch 단측 검정(α 0.05 = 90% 양측 구간 하한 > −δ).
   1종 오류율은 Δ = −δ(비열등성 경계)에서, 검정력은 Δ = 0과 Δ = −2에서. 고정 n=8, n=12와 비교.
2. 분석 함수 세 개의 1종 오류율 점검 (합성 자료):
   E2 계보 짝지은 t (8 계보), E3 네 평균 선형 대비 (5·5·8·8, Welch–Satterthwaite), E4-b 독립 두 집단 Welch (8 대 8).
3. MDE(검정력 80%) 계산표 (비중심 t).
4. 비열등성용 순열 검정(δ 이동 + studentized)의 경계 1종 오류율 (민감도 분석용).

사용: python gpre_sim.py <출력 폴더> [반복 수]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(sys.argv[1])
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
OUT.mkdir(parents=True, exist_ok=True)
EVAL_SD = 1.8  # 실행 축 평가 잡음 (%p, 성공률 75% 가정, 4.1절)
rng = np.random.default_rng(20261016)


# ---------------------------------------------------------------- 공통 함수
def welch(x, y):
    """Welch t: (차이, SE, 자유도)."""
    n1, n2 = len(x), len(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return float(np.mean(x) - np.mean(y)), se, df


def ni_reject(x, y, delta, alpha=0.05):
    """비열등성: H0 Δ ≤ −δ 기각 ⇔ 90% 양측 구간 하한 > −δ."""
    d, se, df = welch(x, y)
    return d - stats.t.ppf(1 - alpha, df) * se > -delta


def required_n(sd, delta, power=0.8, alpha=0.05, true_diff=0.0, n_max=200):
    """갈래당 n: 비중심 t로 검정력 ≥ power인 최소 n (등분산 근사, df = 2n−2)."""
    for n in range(2, n_max + 1):
        df = 2 * n - 2
        se = sd * math.sqrt(2 / n)
        ncp = (true_diff + delta) / se
        if 1 - stats.nct.cdf(stats.t.ppf(1 - alpha, df), df, ncp) >= power:
            return n
    return n_max


def blinded_sd(pooled_values, n1, n2, delta_plan):
    """Kieser–Friede: S²_OS − n1·n2/(N(N−1))·Δ_plan²."""
    N = n1 + n2
    s2 = np.var(pooled_values, ddof=1) - n1 * n2 / (N * (N - 1)) * delta_plan ** 2
    return math.sqrt(max(s2, 1e-9))


# ---------------------------------------------------------------- 1. 주 비교 절차
def simulate_procedure(sigma, delta, true_diff, delta_plan, reps, rule="adaptive", n_fixed=8):
    sd = math.sqrt(sigma ** 2 + EVAL_SD ** 2)
    rej, ns = 0, []
    for _ in range(reps):
        if rule == "adaptive":
            a4 = rng.normal(true_diff, sd, 4)
            b4 = rng.normal(0.0, sd, 4)
            s = blinded_sd(np.concatenate([a4, b4]), 4, 4, delta_plan)
            n = int(min(12, max(8, required_n(s, delta))))
            a = np.concatenate([a4, rng.normal(true_diff, sd, n - 4)])
            b = np.concatenate([b4, rng.normal(0.0, sd, n - 4)])
        else:
            n = n_fixed
            a = rng.normal(true_diff, sd, n)
            b = rng.normal(0.0, sd, n)
        rej += ni_reject(a, b, delta)
        ns.append(n)
    return rej / reps, float(np.mean(ns)), float(np.mean(np.array(ns) == 12))


# 필요 n 계산은 반복마다 비중심 t를 부르므로, (sd 반올림, δ)로 캐시한다.
# [알려진 결함, 6차 검토 11.1] 키는 반올림하지만 계산은 최초 입력 sd로 해서, 같은 키의 n이 실행 순서에 따라 달라질 수 있다.
# 1차 결과 보존을 위해 고치지 않고 두며, 수정판은 gpre_sim_v2.py (사전 고정 SD 구간 → n 표).
_req_cache = {}
_orig_required_n = required_n


def required_n(sd, delta, power=0.8, alpha=0.05, true_diff=0.0, n_max=200):  # noqa: F811
    key = (round(sd, 1), delta, power, alpha, true_diff)
    if key not in _req_cache:
        _req_cache[key] = _orig_required_n(sd, delta, power, alpha, true_diff, n_max)
    return _req_cache[key]


results = {"reps": REPS, "eval_sd": EVAL_SD, "procedure": [], "analysis_type1": {}, "mde": [], "permutation": []}
print("== 1. 주 비교 절차 (갈래당 4+4 → 맹검 분산 → n 8~12 → Welch 비열등성)")
print(f"{'sigma':>5} {'delta':>5} {'plan':>5} | {'T1@-d adapt':>11} {'T1 n8':>6} {'T1 n12':>6} | "
      f"{'pow@0 adapt':>11} {'pow n8':>6} {'pow n12':>7} | {'pow@-2':>6} | {'mean n':>6} {'P(n=12)':>7}")
for delta in (5.0, 4.0, 3.0):
    for sigma in (3.0, 4.0, 5.0, 6.0):
        for plan_name, plan in (("0", 0.0), ("-d", -delta)):
            t1, mn, p12 = simulate_procedure(sigma, delta, -delta, plan, REPS)
            pw, mn0, p120 = simulate_procedure(sigma, delta, 0.0, plan, REPS)
            pw2, _, _ = simulate_procedure(sigma, delta, -2.0, plan, REPS)
            if plan_name == "0":
                t1_8, _, _ = simulate_procedure(sigma, delta, -delta, plan, REPS, rule="fixed", n_fixed=8)
                t1_12, _, _ = simulate_procedure(sigma, delta, -delta, plan, REPS, rule="fixed", n_fixed=12)
                pw_8, _, _ = simulate_procedure(sigma, delta, 0.0, plan, REPS, rule="fixed", n_fixed=8)
                pw_12, _, _ = simulate_procedure(sigma, delta, 0.0, plan, REPS, rule="fixed", n_fixed=12)
            else:
                t1_8 = t1_12 = pw_8 = pw_12 = float("nan")
            row = dict(sigma=sigma, delta=delta, delta_plan=plan_name, type1_adaptive=t1, type1_n8=t1_8,
                       type1_n12=t1_12, power0_adaptive=pw, power0_n8=pw_8, power0_n12=pw_12,
                       power_minus2_adaptive=pw2, mean_n_at_0=mn0, p_n12_at_0=p120)
            results["procedure"].append(row)
            print(f"{sigma:5.0f} {delta:5.0f} {plan_name:>5} | {t1:11.4f} {t1_8:6.4f} {t1_12:6.4f} | "
                  f"{pw:11.3f} {pw_8:6.3f} {pw_12:7.3f} | {pw2:6.3f} | {mn0:6.2f} {p120:7.3f}", flush=True)

# ---------------------------------------------------------------- 2. 분석 함수 1종 오류율
print("\n== 2. 분석 함수 1종 오류율 (귀무가설 참, 양측 α = 0.05와 Holm 첫 문턱 0.025)")


def e2_paired(diffs):
    t = np.mean(diffs) / (np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
    return 2 * stats.t.sf(abs(t), len(diffs) - 1)


def e3_contrast(groups):
    """(W2 A2+, W2 A3⁰, W1 A2+, W1 A3⁰) → (μ1−μ2) − (μ3−μ4), Welch–Satterthwaite."""
    c = np.array([1, -1, -1, 1])
    means = np.array([np.mean(g) for g in groups])
    vs = np.array([np.var(g, ddof=1) / len(g) for g in groups])
    est = float(c @ means)
    se = math.sqrt(float((c ** 2) @ vs))
    df = (c ** 2 @ vs) ** 2 / np.sum((c ** 4) * vs ** 2 / (np.array([len(g) for g in groups]) - 1))
    return 2 * stats.t.sf(abs(est / se), df)


def e4b_welch(x, y):
    d, se, df = welch(x, y)
    return 2 * stats.t.sf(abs(d / se), df)


for name, gen in {
    "E2 paired (8 lineages, contrast SD 5)": lambda: e2_paired(rng.normal(0, 5, 8)),
    "E3 4-mean contrast (5,5,8,8; SDs 4,6,5,7 heteroscedastic)": lambda: e3_contrast(
        [rng.normal(0, s, n) for s, n in ((4, 5), (6, 5), (5, 8), (7, 8))]),
    "E4-b Welch (8 vs 8; SDs 4 and 7)": lambda: e4b_welch(rng.normal(0, 4, 8), rng.normal(0, 7, 8)),
}.items():
    ps = np.array([gen() for _ in range(REPS)])
    r05, r025 = float(np.mean(ps < 0.05)), float(np.mean(ps < 0.025))
    se = math.sqrt(0.05 * 0.95 / REPS)
    results["analysis_type1"][name] = dict(alpha05=r05, alpha025=r025, mc_se_at_05=se)
    print(f"  {name:60} α0.05 → {r05:.4f}   α0.025 → {r025:.4f}   (MC SE {se:.4f})", flush=True)

# ---------------------------------------------------------------- 3. MDE 표
print("\n== 3. MDE (검정력 80%) 와 유의성 문턱 대응 관측 차이, 양측 α 0.025 (Holm 첫 문턱), 대비 SD 5")


def mde(se, df, alpha_two=0.025, power=0.8):
    tc = stats.t.ppf(1 - alpha_two / 2, df)
    lo, hi = 0.0, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        pw = 1 - stats.nct.cdf(tc, df, mid / se) + stats.nct.cdf(-tc, df, mid / se)
        lo, hi = (mid, hi) if pw < power else (lo, mid)
    return tc * se, hi


for name, se, df in (("E2 paired, 8 lineages", 5 / math.sqrt(8), 7),
                     ("E2 paired, 12 lineages", 5 / math.sqrt(12), 11),
                     ("E3 contrast 5,5,8,8", math.sqrt(25 / 5 * 2 + 25 / 8 * 2), 17.27),
                     ("E3 contrast 8,8,8,8", math.sqrt(25 / 8 * 4), 28.0),
                     ("E4-b 8 vs 8", math.sqrt(25 / 8 * 2), 14.0),
                     ("E4-b 12 vs 12", math.sqrt(25 / 12 * 2), 22.0)):
    thr, m = mde(se, df)
    results["mde"].append(dict(analysis=name, se=round(se, 3), df=df, threshold=round(thr, 2), mde80=round(m, 2)))
    print(f"  {name:26} SE {se:5.2f}  df {df:6.2f}  threshold {thr:5.2f}  MDE80 {m:5.2f}", flush=True)

# ---------------------------------------------------------------- 4. 비열등성 순열 검정 (민감도)
print("\n== 4. 비열등성 순열 검정 (A를 δ만큼 올려 경계 귀무가설로 만든 뒤 studentized Welch 통계량, 8 대 8)")
from itertools import combinations  # noqa: E402

SPLITS = [np.array(c) for c in combinations(range(16), 8)]  # 12,870개


def perm_ni_p(a, b, delta):
    z = np.concatenate([a + delta, b])
    obs = welch(a + delta, b)
    t_obs = obs[0] / obs[1]
    cnt = 0
    # [알려진 결함, 6차 검토 11.2] 사전식 목록의 10개마다 하나라 균일 표집이 아니다. 수정판은 gpre_sim_v2.py (전체 열거).
    for idx in SPLITS[::10]:  # 1,287개 분할로 근사 (속도)
        mask = np.zeros(16, bool)
        mask[idx] = True
        d, se, _ = welch(z[mask], z[~mask])
        cnt += d / se >= t_obs
    return cnt / len(SPLITS[::10])


for sds in ((4.4, 4.4), (3.0, 6.0)):
    reps = min(REPS, 2000)
    rej = sum(perm_ni_p(rng.normal(-5.0, sds[0], 8), rng.normal(0.0, sds[1], 8), 5.0) < 0.05 for _ in range(reps))
    results["permutation"].append(dict(sds=sds, delta=5.0, reps=reps, type1_at_boundary=rej / reps))
    print(f"  SDs {sds}: boundary type I {rej / reps:.4f} ({reps} reps, 1,287 splits each)", flush=True)

(OUT / "gpre_sim_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("saved", OUT / "gpre_sim_results.json")
