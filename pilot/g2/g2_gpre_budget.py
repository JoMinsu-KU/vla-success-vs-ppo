"""G-PRE 예산표: G2 실측(S4·S5·S6·S9)으로 본실험 GPU 시간을 시나리오별로 계산한다. GPU 사용 없음.

맹검 (D-G2-4): A2⁺ 비용은 cost_A2p.jsonl의 **실행 전체 합계**만 읽는다. 라운드별 값은 출력하지 않는다.
시나리오:
  A  모든 학습·평가 결정적 (명세 C11 기본안)
  B  결정적 설정은 E2(E2-a, E2-b, 조건부 A3⁰₊)에만, 나머지 학습·평가는 기본 설정 (검토 의견의 정의, A3⁰ 62.5시간)
  B' B + E1 A3⁰ 전 시드 결정적 (명세 3.2 C11: "E2의 계보(A3⁰ 학습 시드)와 모든 분기는 결정적")
갈래 구성·실행 수는 명세 v0.12 7.3절 실행표. n(핵심 갈래 시드)은 GPRE_simulation_v2의 n 대응표 범위 8~12.
사용: python g2_gpre_budget.py <repo>/pilot/artifacts/g2 <출력 json>
"""
import json
import statistics as st
import sys
from pathlib import Path

ART = Path(sys.argv[1])
OUT = Path(sys.argv[2])
R = 390  # 본실험 라운드 수
H = 3600.0

# ---------------------------------------------------------------- 실측
s5 = json.loads((ART / "s5/g2_s5.json").read_text())["round_time"]
a3 = [json.loads(x) for x in (ART / "s9/cost_A3.jsonl").read_text().splitlines()]
a3 = [x for x in a3 if "round_s" in x]
a2 = [json.loads(x) for x in (ART / "s9/cost_A2p.jsonl").read_text().splitlines()]
a2 = [x for x in a2 if "round_s" in x]
tot2 = {k: sum(x[k] for x in a2) for k in ("round_s", "collect_s", "update_s")}  # 합계만
n2 = len(a2)
del a2
s4 = [json.loads(x) for x in (ART / "s4/g2_s4_timing.jsonl").read_text().splitlines()]

A3_det = dict(round=st.mean(x["round_s"] for x in a3), collect=st.mean(x["collect_s"] for x in a3),
              recompute=st.mean(x["recompute_old_s"] for x in a3), update=st.mean(x["update_s"] for x in a3))
A3_def = dict(round=s5["round_total_s"], collect=s5["collect_s"], recompute=s5["recompute_old_c10_s"], update=s5["update_steps_s"])
r_collect = A3_det["collect"] / A3_def["collect"]
r_update = A3_det["update"] / A3_def["update"]
r_round = A3_det["round"] / A3_def["round"]
A2_det_round = tot2["round_s"] / n2
A2_other = (tot2["round_s"] - tot2["collect_s"] - tot2["update_s"]) / n2
A2_def_round_comp = (tot2["collect_s"] / r_collect + tot2["update_s"] / r_update) / n2 + A2_other
A2_def_round_overall = A2_det_round / r_round
# 구조적 상한: 성공 64/64 → 스텝 32 = A3⁰의 업데이트 스텝 수. 스텝당 시간은 A3⁰ 값(value head 포함이라 보수적)
A2_det_round_max = tot2["collect_s"] / n2 + A3_det["update"] + A2_other
eval_batch_def = st.mean(x["total_s"] for x in s4)
eval_batch_det = eval_batch_def * r_collect  # 추정: 평가는 추론만 하므로 수집의 결정적/기본 비율을 쓴다

run_h = {  # 390라운드 1회 GPU시간
    ("A3", "det"): A3_det["round"] * R / H, ("A3", "def"): A3_def["round"] * R / H,
    ("A2p", "det"): A2_det_round * R / H, ("A2p", "def"): A2_def_round_comp * R / H,
}
rnd = {("A3", "det"): A3_det["round"], ("A3", "def"): A3_def["round"],
       ("A2p", "det"): A2_det_round, ("A2p", "def"): A2_def_round_comp}


def h(x):
    return round(x, 1)


def budget(scn, n):
    det_all = scn == "A"
    m = lambda e2=False, a3_lineage=False: "det" if (det_all or e2 or (a3_lineage and scn == "B'")) else "def"  # noqa: E731
    ev = eval_batch_det if det_all else eval_batch_def
    L = []
    L.append(("E1 A2⁺", f"{n}회", n * run_h[("A2p", m())], "보호. 50라운드 평균 외삽"))
    L.append(("E1 A3⁰", f"{n}회", n * run_h[("A3", m(a3_lineage=True))], "보호, E2 계보 겸용"))
    L.append(("E1 A2± (A3⁰ 비용 가정)", "3회", 3 * run_h[("A3", m())], "미측정"))
    L.append(("E1 A3ˢ (A3⁰ 비용 가정)", "3회", 3 * run_h[("A3", m())], "보상만 다른 PPO"))
    tune_rounds_a3, tune_rounds_a2 = 3 * 4 * 150, 4 * 150
    L.append(("튜닝 4갈래 × 후보 4 × 150라운드", "2,400라운드",
              (tune_rounds_a3 * rnd[("A3", m())] + tune_rounds_a2 * rnd[("A2p", m())]) / H, "A2±·A3ˢ는 A3⁰ 비용 가정"))
    col = A3_det["collect"] / H
    e2a_lo = 192 * col + 768 * A3_det["update"] / H
    e2a_hi = 192 * col + 3 * 768 * A3_det["update"] / H
    L.append(("E2-a 기울기 진단 (하한)", "수집 192배치 + 768 (배치×변형)", e2a_lo, "변형당 전체 배치 역전파 1회"))
    L.append(("E2-b 분기", "8×3×4×20 = 1,920라운드", 1920 * A3_det["round"] / H, "결정적 필수"))
    L.append(("E3 W2 A2⁺·A3⁰", "5 + 5회", 5 * run_h[("A2p", m())] + 5 * run_h[("A3", m(a3_lineage=False))], "W2 웜업 학습 별도(미측정)"))
    n_ck_e1 = 2 * n + 3 + 3 + 3 + 1
    ev_lines = [("평가 E1", f"체크포인트 {n_ck_e1} × 16조건 × 192", n_ck_e1 * 48),
                ("평가 E3", "체크포인트 10 × 16 × 192", 10 * 48),
                ("평가 E2-b", "체크포인트 96 × 4조건 × 128", 96 * 4 * 2),
                ("평가 튜닝 ID", "16설정 × 3 × 512", 16 * 3 * 8)]
    for name, desc, nb in ev_lines:
        L.append((name, desc, nb * ev / H, f"{nb}배치 × {ev:.0f}초"))
    base = sum(x[2] for x in L)
    cond = [("조건부 A3⁰₊", "5회", 5 * run_h[("A3", "det")], "E2-b 조건 충족 시, 결정적"),
            ("E5 보조 (상한)", "PPO 6회 환산", 6 * run_h[("A3", m())], "예산이 남을 때")]
    e2a_extra = e2a_hi - e2a_lo
    return dict(lines=[(a, b, h(c), d) for a, b, c, d in L], base_total=h(base),
                e2a_upper_extra=h(e2a_extra), base_total_e2a_upper=h(base + e2a_extra),
                conditional=[(a, b, h(c), d) for a, b, c, d in cond],
                all_total=h(base + sum(x[2] for x in cond)))


res = dict(
    measured=dict(A3_det_round_s=A3_det, A3_default_round_s_S5=A3_def,
                  ratio_det_over_default=dict(collect=round(r_collect, 3), update=round(r_update, 3), round=round(r_round, 3)),
                  A2p_det_round_mean_s_from_total=round(A2_det_round, 1), A2p_rounds=n2,
                  eval_batch64_default_s_S4=round(eval_batch_def, 1), eval_batch64_det_s_est=round(eval_batch_det, 1)),
    per_run_390_gpu_h=dict(A3_det=h(run_h[("A3", "det")]), A3_default=h(run_h[("A3", "def")]),
                           A2p_det=h(run_h[("A2p", "det")]),
                           A2p_default_est_componentwise=h(run_h[("A2p", "def")]),
                           A2p_default_est_overall_ratio=h(A2_def_round_overall * R / H),
                           A2p_det_structural_upper=h(A2_det_round_max * R / H),
                           eval_per_checkpoint_16x192_default=h(48 * eval_batch_def / H),
                           eval_per_checkpoint_16x192_det_est=h(48 * eval_batch_det / H)),
    seed_block_gpu_h={s: h(run_h[("A2p", "det" if s == "A" else "def")] + run_h[("A3", "def" if s == "B" else "det")]
                           + 2 * 48 * (eval_batch_det if s == "A" else eval_batch_def) / H) for s in ("A", "B", "B'")},
    scenarios={s: {str(n): budget(s, n) for n in (8, 10, 12)} for s in ("A", "B", "B'")},
)
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps({k: res[k] for k in ("measured", "per_run_390_gpu_h", "seed_block_gpu_h")}, indent=1, ensure_ascii=False))
for s in ("A", "B", "B'"):
    for n in ("8", "10", "12"):
        b = res["scenarios"][s][n]
        print(s, n, "base", b["base_total"], "base(E2-a upper)", b["base_total_e2a_upper"], "all", b["all_total"])
for ln in res["scenarios"]["A"]["8"]["lines"] + res["scenarios"]["A"]["8"]["conditional"]:
    print("A8", ln)
for ln in res["scenarios"]["B"]["8"]["lines"]:
    print("B8", ln)
