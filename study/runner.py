"""본실험 1단계 실행기 (H200 1개, 순차): 튜닝(A2⁺·A3⁰) → 설정 선택 → E1 핵심 시드 1~4 (A3⁰·A2⁺) → 평가 → G-mid.

- 처음 실행 때 계획을 만들어 <W>/sealed/plan.json에 저장한다 (실행 ID ↔ 갈래 대응 포함, 분석 담당자는 열지 않는다).
  공개 색인 <W>/public/runs_index.json에는 실행 ID ↔ (종류, 시드)만 둔다.
- 다시 실행하면 끝난 작업은 건너뛰고, 중간에 끊긴 학습은 latest 체크포인트에서 이어 간다.
- 작업이 실패하면 같은 작업을 2번까지 다시 시도(이어 가기)하고, 그래도 실패하면 멈춘다.
- 본 실행은 태그 gpre가 HEAD의 조상이고 study/gpre_config.json이 태그 이후 바뀌지 않았을 때만 시작한다.
  --smoke는 태그 없이 <W>/smoke에서 축소 설정으로 전체 흐름과 재시작·결정성을 확인한다 (약 2시간).

사용: python study/runner.py [--smoke] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PY = sys.executable
STUDY = C.STUDY


def git(*args) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", str(C.REPO), *args], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def check_gpre_tag():
    rc, _ = git("rev-parse", "-q", "--verify", "refs/tags/gpre")
    if rc != 0:
        raise SystemExit("태그 gpre가 없다. G-PRE 고정(연구자 결정 + 명세 v0.13 + 태그 push) 뒤에 본 실행을 시작한다. 시험은 --smoke")
    rc, _ = git("merge-base", "--is-ancestor", "gpre", "HEAD")
    if rc != 0:
        raise SystemExit("태그 gpre가 HEAD의 조상이 아니다")
    rc, _ = git("diff", "--quiet", "gpre", "HEAD", "--", "study/gpre_config.json")
    if rc != 0:
        raise SystemExit("study/gpre_config.json이 태그 gpre 이후 바뀌었다 (이탈 기록 필요)")


# ---------------------------------------------------------------- 계획
def new_id(used: set) -> str:
    while True:
        r = "r" + secrets.token_hex(3)
        if r not in used:
            used.add(r)
            return r


def make_plan(smoke: bool) -> dict:
    cfg = C.CFG
    used = set()
    jobs = []
    tune_rounds = 2 if smoke else cfg["rounds_tune"]
    sel_rounds = [2] if smoke else cfg["tune_select_rounds"]
    for arm in ("A3", "A2p"):
        cands = cfg["tune_candidates"][arm][:1] if smoke else cfg["tune_candidates"][arm]
        for c in cands:
            rid = new_id(used)
            hp = {k: v for k, v in c.items() if k != "cand"}
            jobs.append(dict(kind="tune_train", run_id=rid, arm=arm, cand=c["cand"], hp=hp, seed=cfg["tune_seed"],
                             rounds=tune_rounds, numerics=cfg["numerics_tune"], episode_tag=f"tune-seed{cfg['tune_seed']}",
                             save_rounds=sel_rounds))
            for r in sel_rounds:
                jobs.append(dict(kind="eval", run_id=rid, suite="idval", name=f"r{r:03d}", ckpt=f"r{r:03d}.pt"))
    jobs.append(dict(kind="select", select_rounds=sel_rounds))
    seeds = cfg["e1_seeds_phase1"][:1] if smoke else cfg["e1_seeds_phase1"]
    for s in seeds:
        for arm in ("A3", "A2p"):
            rid = new_id(used)
            save = [1, 2] if smoke else cfg[f"e1_save_rounds_{arm}"]
            jobs.append(dict(kind="e1_train", run_id=rid, arm=arm, seed=s, rounds=2 if smoke else cfg["rounds_e1"],
                             numerics=cfg["numerics_train_e1"], episode_tag=f"e1-seed{s}", save_rounds=save))
            jobs.append(dict(kind="eval", run_id=rid, suite="e1", name="final", ckpt="final.pt"))
    if smoke:
        e1_a3 = next(j for j in jobs if j["kind"] == "e1_train" and j["arm"] == "A3")
        jobs.append(dict(kind="resume_check", run_id=new_id(used), ref_run_id=e1_a3["run_id"], arm="A3", seed=e1_a3["seed"],
                         rounds=2, numerics=cfg["numerics_train_e1"], episode_tag=e1_a3["episode_tag"], save_rounds=[]))
    jobs.append(dict(kind="gmid", seeds=seeds))
    return dict(created=C.now(), smoke=smoke, repo_head=git("rev-parse", "HEAD")[1], jobs=jobs)


def public_index(plan: dict) -> dict:
    idx = {}
    for j in plan["jobs"]:
        if j["kind"] in ("tune_train", "e1_train", "resume_check"):
            idx[j["run_id"]] = dict(kind=j["kind"], seed=j["seed"])
    return idx


# ---------------------------------------------------------------- 작업 실행
def run_cmd(W: Path, name: str, cmd: list, env: dict) -> int:
    log = W / "logs" / f"{name}.log"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {C.now()} {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return p.returncode


def train_cmd(j: dict, hp: dict, smoke: bool, extra: list | None = None) -> list:
    cmd = [PY, str(STUDY / "train.py"), "--run-id", j["run_id"], "--arm", j["arm"], "--seed", str(j["seed"]),
           "--rounds", str(j["rounds"]), "--numerics", j["numerics"], "--episode-tag", j["episode_tag"],
           "--save-rounds", ",".join(str(x) for x in j["save_rounds"])]
    for k, flag in (("lr", "--lr"), ("vh_lr", "--vh-lr"), ("update", "--update")):
        if k in hp:
            cmd += [flag, str(hp[k])]
    if smoke:
        cmd += ["--smoke-steps", "2", "--latest-every", "1"]
    return cmd + (extra or [])


def run_status(W: Path, rid: str) -> str:
    return C.read_json(W / "public" / "runs" / f"{rid}.json", {}).get("status", "new")


def eval_status(W: Path, j: dict) -> str:
    return C.read_json(W / "public" / "evals" / f"{j['run_id']}_{j['suite']}_{j['name']}.json", {}).get("status", "new")


def selected_hp(W: Path, arm: str) -> dict:
    sel = C.read_json(W / "public" / "tuning_selection.json")
    if not sel:
        raise SystemExit("튜닝 선택 결과가 없다")
    return sel[arm]["hp"]


def resume_check(W: Path, j: dict, env: dict, smoke: bool) -> bool:
    """1라운드 뒤 멈췄다가 이어 간 실행이 끊김 없는 실행과 비트 단위로 같은지 (결정적 경로 + 재시작)."""
    import torch
    hp = selected_hp(W, j["arm"])
    if run_status(W, j["run_id"]) not in ("done",):
        if run_status(W, j["run_id"]) == "new":
            if run_cmd(W, j["run_id"], train_cmd(j, hp, smoke, ["--stop-after", "1"]), env) != 0:
                return False
        if run_cmd(W, j["run_id"], train_cmd(j, hp, smoke), env) != 0:
            return False
    a = torch.load(W / "runs" / j["ref_run_id"] / "ckpt" / "final.pt", map_location="cpu")
    b = torch.load(W / "runs" / j["run_id"] / "ckpt" / "final.pt", map_location="cpu")
    same_params = all(torch.equal(a["params"][k], b["params"][k]) for k in a["params"])
    ra = C.read_jsonl(W / "blinded" / j["ref_run_id"] / "rounds.jsonl")
    rb = C.read_jsonl(W / "blinded" / j["run_id"] / "rounds.jsonl")
    keys = ("Y_per_episode", "losses", "kl_probe_round_end", "entropy_probe")
    same_rounds = len(ra) == len(rb) and all(x.get(k) == y.get(k) for x, y in zip(ra, rb) for k in keys)
    ok = bool(same_params and same_rounds)
    C.write_json(W / "public" / "resume_check.json", dict(lora_bitwise_equal=bool(same_params), round_records_equal=bool(same_rounds),
                                                          passed=ok, checked=C.now()))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="계획만 만들고 보여 준다")
    a = ap.parse_args()
    if a.smoke:
        os.environ["VLA_MAIN_WORK"] = str(C.work_dir() / "smoke")
    elif not a.dry_run:
        check_gpre_tag()
    W = C.work_dir()
    C.ensure_dirs(W)
    plan_f = W / "sealed" / "plan.json"
    plan = C.read_json(plan_f)
    if plan is None:
        plan = make_plan(a.smoke)
        C.write_json(plan_f, plan)
        (W / "public" / "plan_sha256.txt").write_text(C.sha256_file(plan_f) + "\n")
        C.write_json(W / "public" / "runs_index.json", public_index(plan))
        C.public_log(W, f"plan created: {len(plan['jobs'])} jobs, smoke={a.smoke}, head={plan['repo_head'][:10]}")
    if a.dry_run:
        kinds = {}
        for j in plan["jobs"]:
            kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
        print(json.dumps(kinds, ensure_ascii=False))
        return
    env = dict(os.environ)
    for i, j in enumerate(plan["jobs"]):
        tag = f"[{i + 1}/{len(plan['jobs'])}] {j['kind']} {j.get('run_id', '')} {j.get('name', '')}".strip()
        if j["kind"] in ("tune_train", "e1_train"):
            if run_status(W, j["run_id"]) in ("done", "diverged_nonfinite", "diverged_kl"):
                continue
            hp = j["hp"] if j["kind"] == "tune_train" else selected_hp(W, j["arm"])
            C.public_log(W, f"{tag} start")
            for attempt in range(3):
                rc = run_cmd(W, j["run_id"], train_cmd(j, hp, a.smoke), env)
                if rc == 0 and run_status(W, j["run_id"]) in ("done", "diverged_nonfinite", "diverged_kl"):
                    break
                C.public_log(W, f"{tag} exit {rc}, retry {attempt + 1}")
                time.sleep(30)
            else:
                C.public_log(W, f"{tag} FAILED - 멈춤. logs/{j['run_id']}.log 확인")
                sys.exit(1)
            C.public_log(W, f"{tag} {run_status(W, j['run_id'])}")
        elif j["kind"] == "eval":
            if eval_status(W, j) == "done":
                continue
            ck = W / "runs" / j["run_id"] / "ckpt" / j["ckpt"]
            if not ck.exists():
                C.public_log(W, f"{tag} skip: checkpoint {j['ckpt']} 없음 (발산 등으로 학습이 일찍 멈춤)")
                continue
            C.public_log(W, f"{tag} start")
            cmd = [PY, str(STUDY / "evaluate.py"), "--run-id", j["run_id"], "--suite", j["suite"], "--ckpt", str(ck), "--name", j["name"]]
            if a.smoke:
                cmd += ["--batches", "1"] + (["--conds", ",".join(["ID"] + C.EXEC_CONDS)] if j["suite"] == "e1" else [])
            for attempt in range(3):
                if run_cmd(W, f"eval_{j['run_id']}_{j['name']}", cmd, env) == 0 and eval_status(W, j) == "done":
                    break
                time.sleep(30)
            else:
                C.public_log(W, f"{tag} FAILED - 멈춤")
                sys.exit(1)
            C.public_log(W, f"{tag} done")
        elif j["kind"] == "select":
            if C.read_json(W / "public" / "tuning_selection.json"):
                continue
            import select_tuning
            pub = select_tuning.main(W, plan, j["select_rounds"])
            C.public_log(W, f"{tag} {json.dumps(pub, ensure_ascii=False)}")
        elif j["kind"] == "resume_check":
            if C.read_json(W / "public" / "resume_check.json", {}).get("passed"):
                continue
            ok = resume_check(W, j, env, a.smoke)
            C.public_log(W, f"{tag} passed={ok}")
            if not ok:
                sys.exit(1)
        elif j["kind"] == "gmid":
            if C.read_json(W / "public" / "gmid.json"):
                continue
            import gmid
            out = gmid.main(W, C.read_json(W / "public" / "runs_index.json"), j["seeds"])
            C.public_log(W, f"{tag} {json.dumps(out, ensure_ascii=False)}")
    C.public_log(W, "PHASE 1 DONE" + (" (smoke)" if a.smoke else ""))


if __name__ == "__main__":
    main()
