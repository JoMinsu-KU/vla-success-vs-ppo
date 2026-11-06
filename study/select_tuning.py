"""튜닝 설정 선택 (명세 3.5절): 140·145·150라운드 체크포인트 ID 검증 성공률(Y) 평균이 가장 높은 후보.
최고값에서 2%p 안의 후보끼리는 학습률이 작은 쪽 → (A2⁺) U-prop → (A3⁰) value head 학습률이 원 설정(3e-3)인 쪽 → 후보 번호 순.
발산으로 멈춰 선택 라운드 체크포인트가 없는 후보는 제외하고 그 사실을 기록한다.
공개 출력: 갈래별 선택 후보와 설정만 (public/tuning_selection.json). 후보별 성공률은 blinded/tuning_selection_detail.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def choose(arm: str, rows: list, tie_pp: float) -> dict:
    """rows: [{cand, lr, (update|vh_lr), score or None}] → 선택 행."""
    ok = [r for r in rows if r["score"] is not None]
    if not ok:
        raise SystemExit(f"{arm}: 선택 가능한 후보가 없다 (모두 발산·누락). 연구자 결정 필요")
    best = max(r["score"] for r in ok)
    near = [r for r in ok if r["score"] >= best - tie_pp]
    order = {r["cand"]: i for i, r in enumerate(rows)}

    def key(r):
        k = [r["lr"]]
        if arm == "A2p":
            k.append(0 if r["update"] == "U-prop" else 1)
        else:
            k.append(0 if abs(r["vh_lr"] - 3e-3) < 1e-12 else 1)
        k.append(order[r["cand"]])
        return tuple(k)
    return sorted(near, key=key)[0]


def main(work: Path, plan: dict, select_rounds: list):
    detail, public = {}, {}
    for arm in ("A2p", "A3"):
        rows = []
        for job in [j for j in plan["jobs"] if j["kind"] == "tune_train" and j["arm"] == arm]:
            st = C.read_json(work / "public" / "runs" / f"{job['run_id']}.json", {})
            scores = []
            for r in select_rounds:
                ev = C.eval_rows(work / "blinded" / job["run_id"] / f"eval_idval_r{r:03d}.jsonl")
                scores.append(100 * sum(x["Y"] for x in ev) / len(ev) if ev else None)
            score = None if (None in scores or st.get("status") != "done") else sum(scores) / len(scores)
            rows.append(dict(job["hp"], cand=job["cand"], run_id=job["run_id"], status=st.get("status"), scores=scores, score=score))
        pick = choose(arm, rows, C.CFG["tune_tie_pp"])
        detail[arm] = dict(rows=rows, picked=pick["cand"])
        public[arm] = dict(cand=pick["cand"], hp={k: v for k, v in pick.items() if k in ("lr", "vh_lr", "update")},
                           excluded=[r["cand"] for r in rows if r["score"] is None])
    C.write_json(work / "blinded" / "tuning_selection_detail.json", detail)
    C.write_json(work / "public" / "tuning_selection.json", dict(public, decided=C.now()))
    return public
