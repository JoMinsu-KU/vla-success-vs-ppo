"""P5 결과 분석: 6.3절 P5 통과 기준과 기록 항목.

사용: python p5_analyze.py <p5_out_dir>
"""
import json
import math
import sys
from pathlib import Path

d = Path(sys.argv[1])
rows = [json.loads(l) for l in (d / "p5_rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
eps = [json.loads(l) for l in (d / "p5_episodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
log = (d.parent / (d.name + ".log")).read_text(encoding="utf-8", errors="replace") if (d.parent / (d.name + ".log")).exists() else ""

print(f"{'seed':>4} {'rnd':>3} {'succ':>5} {'steps':>6} {'NLL before':>10} {'after':>7} {'grad norm':>14} "
      f"{'KL(k3,subset) end':>17} {'peak GiB':>8} {'collect min':>11} {'train min':>9}")
all_finite, nll_dec, n_steps_tot = True, 0, 0
for r in rows:
    u = r.get("update")
    if not u:
        print(f"{r['seed']:>4} {r['round']:>3} {r['successes']:>3}/64  (no update)")
        continue
    gns = [s["grad_norm"] for s in u["steps"]]
    fin = all(s["finite"] for s in u["steps"])
    all_finite &= fin
    nll_dec += u["subset_nll_after"] < u["subset_nll_before"]
    n_steps_tot += len(u["steps"])
    print(f"{r['seed']:>4} {r['round']:>3} {r['successes']:>3}/64 {len(u['steps']):>3}/{u['n_steps_planned']:<2} "
          f"{u['subset_nll_before']:10.4f} {u['subset_nll_after']:7.4f} {min(gns):6.2f}-{max(gns):<6.2f} "
          f"{u['steps'][-1]['kl']:17.2f} {r['peak_train_cuda_gib']:8.2f} {r['collect_sec'] / 60:11.1f} {r['train_sec'] / 60:9.1f}")

n_upd = sum(1 for r in rows if r.get("update"))
summary = dict(
    rounds=len(rows), episodes_recorded=len(eps), episodes_expected=64 * len(rows),
    successes_total=sum(r["successes"] for r in rows),
    successes_match_episode_records=sum(r["successes"] for r in rows) == sum(e["success"] for e in eps),
    zero_success_rounds=sum(1 for r in rows if r["zero_success_round"]),
    rounds_with_update=n_upd, optimizer_steps_total=n_steps_tot, all_steps_finite=all_finite,
    rounds_subset_nll_decreased=nll_dec,
    max_grad_norm=max(s["grad_norm"] for r in rows if r.get("update") for s in r["update"]["steps"]),
    clip_triggered_steps=sum(1 for r in rows if r.get("update") for s in r["update"]["steps"] if s["grad_norm"] > 10),
    max_peak_train_cuda_gib=max(r.get("peak_train_cuda_gib", 0) for r in rows),
    mean_collect_min=round(sum(r["collect_sec"] for r in rows) / len(rows) / 60, 1),
    mean_train_min=round(sum(r.get("train_sec", 0) for r in rows) / n_upd / 60, 1),
    worker_retries=sum(r.get("worker_retries", 0) for r in rows),
    process_failures_logged=log.count("failed] seed"),
    mean_stored_success_gb_per_round=round(sum(r["successes"] for r in rows) / len(rows) * 80 * 480 * 640 * 3 / 1e9, 2),
)
for s in (0, 1):
    sr = [r["successes"] for r in rows if r["seed"] == s]
    if sr:
        summary[f"seed{s}_success_by_round"] = sr
        summary[f"seed{s}_first_half_vs_second_half"] = (sum(sr[:len(sr) // 2]), sum(sr[len(sr) // 2:]))
print("\nSUMMARY", json.dumps(summary, indent=1))
(d / "p5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
