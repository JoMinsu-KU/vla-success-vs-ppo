"""G-PRE KL_max 참고 계산: A3⁰ (S9, 결정적 math 경로, 50라운드)의 탐침 KL로, KL_max 후보마다 라운드 안 몇 번째 스텝에서 멈출지 추정. GPU 사용 없음.

자료: cost_A3.jsonl의 라운드별 kl_probe_after_step1 (스텝 1 뒤)와 kl_probe_round_end (스텝 32 뒤). 둘 다 KL(π_round ‖ π_now)의
      7개 위치 합, 64개 탐침 상태 평균 (명세 3.3절 정의). A3⁰는 비맹검 갈래. A2⁺ 값은 읽지 않는다.
가정 (두 측정점 사이를 잇는 모양, 두 점을 모두 지나게 고정):
  선형:  KL(s) = k1 + (k32 − k1)(s − 1)/31
  √스텝: KL(s) = k1 + (k32 − k1)(√s − 1)/(√32 − 1)
정지 규칙 (명세 3.3절): 각 optimizer 스텝 뒤 KL을 재서 KL_max를 넘으면 그 라운드의 남은 스텝을 건너뛴다.
  → 수행 스텝 수 = KL(s) > KL_max인 첫 s (넘지 않으면 32).
참고 진단: 원점을 지나는 거듭제곱 KL(s) = c·s^b를 두 점에 맞춘 지수 b = ln(k32/k1)/ln 32.
이 계산은 A3⁰(PPO 손실, 32스텝) 궤적의 모양을 빌린 참고 자료다. A2⁺(NLL, 라운드당 스텝 수가 성공 수에 따름)에서 실제로 몇 스텝 뒤 멈출지는 알 수 없다.
사용: python g2_gpre_klmax.py <cost_A3.jsonl> <출력 json>
"""
import json
import math
import sys

import numpy as np

rows = [json.loads(x) for x in open(sys.argv[1])]
rows = [r for r in rows if "kl_probe_round_end" in r]
k1 = np.array([r["kl_probe_after_step1"] for r in rows])
k32 = np.array([r["kl_probe_round_end"] for r in rows])
S = 32
s = np.arange(1, S + 1)


def curve(model, a, b):
    if model == "linear":
        return a + (b - a) * (s - 1) / (S - 1)
    return a + (b - a) * (np.sqrt(s) - 1) / (math.sqrt(S) - 1)


def q(x):
    return dict(min=float(np.min(x)), p10=float(np.percentile(x, 10)), median=float(np.median(x)),
                p90=float(np.percentile(x, 90)), max=float(np.max(x)), mean=float(np.mean(x)))


CANDS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
out = dict(n_rounds=len(rows), kl_step1=q(k1), kl_step32=q(k32), ratio_k32_over_k1=q(k32 / k1),
           power_exponent_b=q(np.log(k32 / k1) / math.log(S)),
           note="b≈1이면 원점에서 거의 선형, b≈0.5면 √스텝 모양 (원점 기준). 두 점만으로 모양을 판정할 수는 없다.",
           candidates={})
for c in CANDS:
    d = {}
    for m in ("linear", "sqrt"):
        steps = np.array([int(np.argmax(curve(m, a, b) > c)) + 1 if np.any(curve(m, a, b) > c) else S for a, b in zip(k1, k32)])
        d[m] = dict(steps=q(steps), frac_rounds_stopped_early=float(np.mean(steps < S)),
                    frac_stop_at_step1=float(np.mean(steps == 1)))
    out["candidates"][str(c)] = d
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"rounds {len(rows)}  KL step1 median {np.median(k1):.4f} (p10 {np.percentile(k1,10):.4f}, p90 {np.percentile(k1,90):.4f}, max {k1.max():.4f})"
      f"  step32 median {np.median(k32):.3f} (p10 {np.percentile(k32,10):.3f}, p90 {np.percentile(k32,90):.3f}, min {k32.min():.3f}, max {k32.max():.3f})")
print(f"exponent b median {np.median(np.log(k32/k1)/math.log(S)):.2f}  range {np.min(np.log(k32/k1)/math.log(S)):.2f}..{np.max(np.log(k32/k1)/math.log(S)):.2f}")
print("KL_max | linear: median steps (p10-p90), stopped% , stop@1% | sqrt: same")
for c in CANDS:
    L, Q = out["candidates"][str(c)]["linear"], out["candidates"][str(c)]["sqrt"]
    print(f"{c:6} | {L['steps']['median']:5.1f} ({L['steps']['p10']:.0f}-{L['steps']['p90']:.0f}) {100*L['frac_rounds_stopped_early']:5.0f}% {100*L['frac_stop_at_step1']:4.0f}%"
          f" | {Q['steps']['median']:5.1f} ({Q['steps']['p10']:.0f}-{Q['steps']['p90']:.0f}) {100*Q['frac_rounds_stopped_early']:5.0f}% {100*Q['frac_stop_at_step1']:4.0f}%")
