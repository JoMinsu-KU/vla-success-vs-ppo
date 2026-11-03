"""G2 S9 A3⁰ 학습 신호 판정 (사전 고정, 태그 g2-prereg). 출력은 "있음/없음"과 사용한 라운드 수 R뿐이다.

정의 (G2 시험 명세 3절):
  라운드 1~R (R ≤ 50; 라운드 1 = W1로 수집한 첫 배치)의
  (a) critic 설명 분산 EV_r (그 라운드 업데이트 전 θ_old의 재계산 value와 GAE returns로 계산):
      mean(EV, 마지막 10라운드) − mean(EV, 처음 10라운드) ≥ 0.05 이고 mean(EV, 마지막 10라운드) > 0
  (b) 라운드별 수집 성공률(Y 평균)의 라운드 번호에 대한 OLS 기울기의 단측 90% 하한 > 0
      (하한 = b − t_{0.90, R−2} · SE(b), 잔차 분산은 자유도 R−2)
  (a)와 (b)를 모두 만족하면 "있음", 아니면 "없음". R < 20이면 판정하지 않는다 ("판정 불가").
사용: python g2_signal.py <blinded/A3_rounds.jsonl>
"""
import json
import math
import sys
from pathlib import Path

from scipy import stats

rows = [json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
rows = sorted(rows, key=lambda r: r["round"])[:50]
R = len(rows)
if R < 20:
    print(f"A3⁰ 학습 신호: 판정 불가 (R={R} < 20)")
    sys.exit(0)
ev = [r["explained_variance"] for r in rows]
ev = [e if e == e else 0.0 for e in ev]  # NaN(분산 0)은 0으로
first, last = sum(ev[:10]) / 10, sum(ev[-10:]) / 10
cond_a = (last - first >= 0.05) and (last > 0)
x = [r["round"] for r in rows]
y = [r["success_rate"] for r in rows]
xm, ym = sum(x) / R, sum(y) / R
sxx = sum((xi - xm) ** 2 for xi in x)
b = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y)) / sxx
resid = [yi - (ym + b * (xi - xm)) for xi, yi in zip(x, y)]
s2 = sum(e * e for e in resid) / (R - 2)
se = math.sqrt(s2 / sxx)
lower = b - stats.t.ppf(0.90, R - 2) * se
cond_b = lower > 0
print(f"A3⁰ 학습 신호: {'있음' if (cond_a and cond_b) else '없음'} (R={R})")
