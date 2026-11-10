"""본실험 1단계 실행기 (H200 1개, 순차): 튜닝(A2⁺·A3⁰) → 설정 선택 → E1 핵심 시드 1~4 (A3⁰·A2⁺) → 평가 → G-mid.

- 처음 실행 때 계획을 만들어 <W>/sealed/plan.json에 저장한다 (실행 ID ↔ 갈래 대응 포함, 분석 담당자는 열지 않는다).
  공개 색인 <W>/public/runs_index.json에는 실행 ID ↔ (종류, 시드)만 둔다.
- 다시 실행하면 끝난 작업은 건너뛰고, 중간에 끊긴 학습은 latest 체크포인트에서 이어 간다.
- 작업이 실패하면 같은 작업을 2번까지 다시 시도(이어 가기)하고, 그래도 실패하면 멈춘다.
- 본 실행은 태그 gpre가 HEAD의 조상이고 study/gpre_config.json이 태그 이후 바뀌지 않았을 때만 시작한다.
  --smoke는 태그 없이 <W>/smoke에서 축소 설정으로 전체 흐름과 재시작·결정성을 확인한다 (약 2시간).

2단계 (--phase 2, 1단계 G-mid 뒤): 핵심 갈래(A3⁰·A2⁺) 나머지 시드 5~n (n = public/gmid.json) → 보조 갈래(A2±·A3ˢ)
  튜닝 4후보씩 + 선택 → 보조 갈래 E1 시드 1~3. 계획은 <W>/sealed/plan_phase2.json, 설정은 study/phase2_config.json.
  --phase 2 --smoke는 <W>/smoke(1단계 스모크 결과 필요)에서 축소 흐름 + 동등성 검사(새 학습기가 A3⁰·A2⁺를 1단계 스모크와
  비트 단위로 같게 학습하는지)를 한다.

사용: python study/runner.py [--phase 1|2] [--smoke] [--dry-run]
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


P2 = json.loads((STUDY / "phase2_config.json").read_text(encoding="utf-8"))


def with_e4b(jobs: list) -> list:
    """E1·E2-b 최종 평가 뒤에 같은 체크포인트의 E4-b를 붙인다 (study/E4b_design.md)."""
    out = []
    for j in jobs:
        out.append(j)
        if j["kind"] == "eval" and j["suite"] in ("e1", "e2b") and j["name"] == "final":
            out.append(dict(kind="e4b", run_id=j["run_id"], ckpt="final.pt", base=j.get("base", "")))
    return out


def make_plan_phase2(smoke: bool, W: Path) -> dict:
    cfg = C.CFG
    gm = C.read_json(W / "public" / "gmid.json")
    if not gm:
        raise SystemExit("1단계 G-mid 결과(public/gmid.json)가 없다. 2단계는 1단계가 끝난 뒤 시작한다")
    used = set(C.read_json(W / "public" / "runs_index.json", {}))
    jobs = []
    # (0) 1단계 E1 실행(시드 1~4)의 E4-b (1단계 실행기에는 없었다)
    p1 = C.read_json(W / "sealed" / "plan.json") or {"jobs": []}
    for j in p1["jobs"]:
        if j["kind"] == "e1_train":
            jobs.append(dict(kind="e4b", run_id=j["run_id"], ckpt="final.pt", base=""))
    # (a) 핵심 갈래 나머지 시드 (보호 대상, 먼저)
    core_seeds = list(range(P2["core_first_new_seed"], int(gm["n"]) + 1))
    if smoke:
        core_seeds = core_seeds[:1]
    for s in core_seeds:
        for arm in P2["core_arms"]:
            rid = new_id(used)
            jobs.append(dict(kind="e1_train", run_id=rid, arm=arm, seed=s, rounds=2 if smoke else cfg["rounds_e1"],
                             numerics=cfg["numerics_train_e1"], episode_tag=f"e1-seed{s}",
                             save_rounds=[1, 2] if smoke else cfg[f"e1_save_rounds_{arm}"], sel_file="tuning_selection.json"))
            jobs.append(dict(kind="eval", run_id=rid, suite="e1", name="final", ckpt="final.pt"))
    # (b) 보조 갈래 튜닝 (1단계와 같은 튜닝 시드·과제 표본·단축 예산·선택 라운드)
    tune_rounds = 2 if smoke else cfg["rounds_tune"]
    sel_rounds = [2] if smoke else cfg["tune_select_rounds"]
    for arm in P2["aux_arms"]:
        cands = P2["aux_tune_candidates"][arm][:1] if smoke else P2["aux_tune_candidates"][arm]
        for c in cands:
            rid = new_id(used)
            jobs.append(dict(kind="tune_train", run_id=rid, arm=arm, cand=c["cand"], hp={k: v for k, v in c.items() if k != "cand"},
                             seed=cfg["tune_seed"], rounds=tune_rounds, numerics=cfg["numerics_tune"],
                             episode_tag=f"tune-seed{cfg['tune_seed']}", save_rounds=sel_rounds))
            for r in sel_rounds:
                jobs.append(dict(kind="eval", run_id=rid, suite="idval", name=f"r{r:03d}", ckpt=f"r{r:03d}.pt"))
    jobs.append(dict(kind="select", select_rounds=sel_rounds, arms=P2["aux_arms"], out_name="tuning_selection_phase2.json"))
    # (c) 보조 갈래 E1 (시드 1~3, 같은 시드의 핵심 갈래와 같은 과제 표본, 결정적 경로 = 안 A⁻)
    aux_seeds = P2["aux_seeds"][:1] if smoke else P2["aux_seeds"]
    for s in aux_seeds:
        for arm in P2["aux_arms"]:
            rid = new_id(used)
            jobs.append(dict(kind="e1_aux_train", run_id=rid, arm=arm, seed=s, rounds=2 if smoke else cfg["rounds_e1"],
                             numerics=cfg["numerics_train_e1"], episode_tag=f"e1-seed{s}",
                             save_rounds=[1, 2] if smoke else P2[f"e1_save_rounds_{arm}"], sel_file="tuning_selection_phase2.json"))
            jobs.append(dict(kind="eval", run_id=rid, suite="e1", name="final", ckpt="final.pt"))
    # (d) A1 시연 SFT 재학습 (study/A1_design.md), 시드 1~3
    a1_seeds = P2["a1_seeds"][:1] if smoke else P2["a1_seeds"]
    for s in a1_seeds:
        rid = new_id(used)
        jobs.append(dict(kind="a1_train", run_id=rid, seed=s, steps=20 if smoke else P2["a1_steps"]))
        jobs.append(dict(kind="eval", run_id=rid, suite="e1", name="final", ckpt="final.pt"))
        if smoke:
            jobs.append(dict(kind="a1_resume_check", run_id=new_id(used), ref_run_id=rid, seed=s, steps=20))
    if smoke:  # 새 학습기의 A3⁰·A2⁺ 경로가 1단계 스모크 실행과 비트 단위로 같은지
        p1 = C.read_json(W / "sealed" / "plan.json")
        for ref in [j for j in p1["jobs"] if j["kind"] == "e1_train"]:
            jobs.insert(0, dict(kind="equiv_check", run_id=new_id(used), ref_run_id=ref["run_id"], arm=ref["arm"], seed=ref["seed"],
                                rounds=ref["rounds"], numerics=ref["numerics"], episode_tag=ref["episode_tag"],
                                save_rounds=ref["save_rounds"], sel_file="tuning_selection.json"))
    return dict(created=C.now(), phase=2, smoke=smoke, repo_head=git("rev-parse", "HEAD")[1], gmid_n=int(gm["n"]), jobs=with_e4b(jobs))


def make_plan_phase3(smoke: bool, W: Path, a3plus: bool) -> dict:
    """3단계: E2-b 분기 (+ 조건부 A3⁰₊). 계보 = E1 A3⁰ 실행 (봉인 계획에서 찾는다). E1 핵심 갈래가 모두 끝난 뒤에만 만든다."""
    cfg = C.CFG
    lineage_jobs = []
    for f in ("plan.json", "plan_phase2.json"):
        pl = C.read_json(W / "sealed" / f)
        if pl:
            lineage_jobs += [j for j in pl["jobs"] if j["kind"] == "e1_train"]
    if not smoke:
        not_done = [j["run_id"] for j in lineage_jobs if run_status(W, j["run_id"]) not in ("done", "diverged_nonfinite", "diverged_kl")]
        if not_done or not lineage_jobs:
            raise SystemExit(f"E1 핵심 갈래가 끝나지 않았다 ({len(not_done)}개 남음). 3단계는 E1 핵심 갈래가 모두 끝난 뒤 계획한다")
    seeds = [1] if smoke else P2["e2_lineage_seeds"]
    points = [2] if smoke else P2["e2_branch_points"]
    n_br = 1 if smoke else P2["e2_branch_rounds"]
    used = set(C.read_json(W / "public" / "runs_index.json", {}))
    jobs = []
    for s in seeds:
        lin = [j for j in lineage_jobs if j["arm"] == "A3" and j["seed"] == s]
        if len(lin) != 1:
            raise SystemExit(f"계보 시드 {s}의 A3⁰ 실행이 1개가 아니다 ({len(lin)})")
        lin = lin[0]
        for p in points:
            jid = "e" + secrets.token_hex(3)
            jobs.append(dict(kind="e2a", job_id=jid, seed=s, point=p, lineage=lin["run_id"],
                             lineage_ckpt=str(W / "runs" / lin["run_id"] / "ckpt" / f"r{p - 1:03d}.pt"),
                             sizes="1,2" if smoke else "1,2,4,8", ood="Position" if smoke else ",".join(C.EXEC_CONDS)))
            for name, rule in P2["e2_rules"].items():
                rid = new_id(used)
                ck = W / "runs" / lin["run_id"] / "ckpt"
                extra = ["--init-ckpt", str(ck / f"r{p - 1:03d}.pt"), "--adv-rule", rule]
                if name == "C±":
                    extra += ["--check-ckpt", str(ck / f"r{p:03d}.pt")]
                jobs.append(dict(kind="e2b_train", run_id=rid, arm="A3", seed=s, rounds=p - 1 + n_br, numerics=cfg["numerics_train_e1"],
                                 episode_tag=lin["episode_tag"], save_rounds=[], lineage=lin["run_id"], point=p, rule=name,
                                 extra=extra, sel_file="tuning_selection.json"))
                jobs.append(dict(kind="eval", run_id=rid, suite="e2b", name="final", ckpt="final.pt"))
    if a3plus:
        for s in ([1] if smoke else P2["a3plus_seeds"]):
            rid = new_id(used)
            jobs.append(dict(kind="e1_aux_train", run_id=rid, arm="A3", seed=s, rounds=2 if smoke else cfg["rounds_e1"],
                             numerics=cfg["numerics_train_e1"], episode_tag=f"e1-seed{s}", save_rounds=[390] if not smoke else [2],
                             extra=["--adv-rule", "gae_masked"], sel_file="tuning_selection.json", variant="A3plus"))
            jobs.append(dict(kind="eval", run_id=rid, suite="e1", name="final", ckpt="final.pt"))
    return dict(created=C.now(), phase=3, smoke=smoke, repo_head=git("rev-parse", "HEAD")[1], a3plus=a3plus, jobs=with_e4b(jobs))


def public_index(plan: dict) -> dict:
    idx = {}
    for j in plan["jobs"]:
        if j["kind"] in ("tune_train", "e1_train", "e1_aux_train", "resume_check", "equiv_check", "a1_train", "a1_resume_check",
                         "e2b_train"):
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
    for k, flag in (("lr", "--lr"), ("vh_lr", "--vh-lr"), ("update", "--update"), ("iters", "--iters")):
        if k in hp:
            cmd += [flag, str(hp[k])]
    if smoke:
        cmd += ["--smoke-steps", "2", "--latest-every", "1"]
    return cmd + (extra or [])


def a1_data() -> Path:
    d = Path(os.environ.get("VLA_A1_DATA", P2["a1_data_default"])).expanduser()
    if not (d / "meta.json").exists():
        raise SystemExit(f"A1 데이터가 없다: {d} (study/A1_design.md 1절, a1_prepare.py)")
    return d


def a1_cmd(j: dict, smoke: bool, extra: list | None = None) -> list:
    cmd = [PY, str(STUDY / "train_a1.py"), "--run-id", j["run_id"], "--seed", str(j["seed"]), "--data", str(a1_data()),
           "--steps", str(j["steps"]), "--numerics", C.CFG["numerics_train_e1"]]
    if smoke:
        cmd += ["--latest-every", "10", "--log-every", "5"]
    return cmd + (extra or [])


def a1_resume_check(W: Path, j: dict, env: dict, smoke: bool) -> bool:
    """A1: 10스텝에서 멈췄다 이어 간 실행이 끊김 없는 실행(ref)과 LoRA·손실 기록이 비트 단위로 같은지."""
    import torch
    if run_status(W, j["run_id"]) != "done":
        if run_status(W, j["run_id"]) == "new" and run_cmd(W, j["run_id"], a1_cmd(j, smoke, ["--stop-after", "10"]), env) != 0:
            return False
        if run_cmd(W, j["run_id"], a1_cmd(j, smoke), env) != 0:
            return False
    a = torch.load(W / "runs" / j["ref_run_id"] / "ckpt" / "final.pt", map_location="cpu")
    b = torch.load(W / "runs" / j["run_id"] / "ckpt" / "final.pt", map_location="cpu")
    same_params = all(torch.equal(a["params"][k], b["params"][k]) for k in a["params"])
    la = [r["loss"] for r in C.read_jsonl(W / "blinded" / j["ref_run_id"] / "a1_loss.jsonl")]
    lb = [r["loss"] for r in C.read_jsonl(W / "blinded" / j["run_id"] / "a1_loss.jsonl")]
    ok = bool(same_params and la == lb and len(la) > 0)
    C.write_json(W / "public" / "a1_resume_check.json", dict(lora_bitwise_equal=bool(same_params), loss_records_equal=la == lb,
                                                             passed=ok, checked=C.now()))
    return ok


def run_status(W: Path, rid: str) -> str:
    return C.read_json(W / "public" / "runs" / f"{rid}.json", {}).get("status", "new")


def eval_status(W: Path, j: dict) -> str:
    return C.read_json(W / "public" / "evals" / f"{j['run_id']}_{j['suite']}_{j['name']}.json", {}).get("status", "new")


def selected_hp(W: Path, arm: str, sel_file: str = "tuning_selection.json") -> dict:
    sel = C.read_json(W / "public" / sel_file)
    if not sel:
        raise SystemExit("튜닝 선택 결과가 없다")
    return sel[arm]["hp"]


def resume_check(W: Path, j: dict, env: dict, smoke: bool) -> bool:
    """resume_check: 1라운드 뒤 멈췄다가 이어 간 실행이 끊김 없는 실행(ref)과 비트 단위로 같은지 (결정적 경로 + 재시작).
    equiv_check: 현재 코드로 끊김 없이 다시 돈 실행이 ref(1단계 코드의 스모크 실행)와 비트 단위로 같은지."""
    import torch
    hp = selected_hp(W, j["arm"], j.get("sel_file", "tuning_selection.json"))
    if run_status(W, j["run_id"]) not in ("done",):
        if run_status(W, j["run_id"]) == "new" and j["kind"] == "resume_check":
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
    C.write_json(W / "public" / ("resume_check.json" if j["kind"] == "resume_check" else f"equiv_check_{j['ref_run_id']}.json"), dict(lora_bitwise_equal=bool(same_params), round_records_equal=bool(same_rounds),
                                                          passed=ok, checked=C.now()))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="계획만 만들고 보여 준다")
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--a3plus", action="store_true", help="3단계: 조건부 A3⁰₊ (E2-b 판정 뒤에만)")
    a = ap.parse_args()
    if a.smoke:
        os.environ["VLA_MAIN_WORK"] = str(C.work_dir() / "smoke")
    elif not a.dry_run:
        check_gpre_tag()
    W = C.work_dir()
    C.ensure_dirs(W)
    sfx = {1: "", 2: "_phase2", 3: "_phase3"}[a.phase]
    plan_f = W / "sealed" / f"plan{sfx}.json"
    plan = C.read_json(plan_f)
    if plan is None:
        plan = (make_plan(a.smoke) if a.phase == 1 else make_plan_phase2(a.smoke, W) if a.phase == 2
                else make_plan_phase3(a.smoke, W, a.a3plus))
        C.write_json(plan_f, plan)
        (W / "public" / f"plan{sfx}_sha256.txt").write_text(C.sha256_file(plan_f) + "\n")
        idx = C.read_json(W / "public" / "runs_index.json", {}) if a.phase >= 2 else {}
        idx.update(public_index(plan))
        C.write_json(W / "public" / "runs_index.json", idx)
        C.public_log(W, f"plan{sfx} created: {len(plan['jobs'])} jobs, smoke={a.smoke}, head={plan['repo_head'][:10]}")
    if a.dry_run:
        kinds = {}
        for j in plan["jobs"]:
            kinds[j["kind"]] = kinds.get(j["kind"], 0) + 1
        print(json.dumps(kinds, ensure_ascii=False))
        return
    env = dict(os.environ)
    for i, j in enumerate(plan["jobs"]):
        tag = f"[{i + 1}/{len(plan['jobs'])}] {j['kind']} {j.get('run_id', '')} {j.get('name', '')}".strip()
        if j["kind"] in ("tune_train", "e1_train", "e1_aux_train", "e2b_train"):
            if run_status(W, j["run_id"]) in ("done", "diverged_nonfinite", "diverged_kl"):
                continue
            hp = j["hp"] if j["kind"] == "tune_train" else selected_hp(W, j["arm"], j.get("sel_file", "tuning_selection.json"))
            C.public_log(W, f"{tag} start")
            for attempt in range(3):
                rc = run_cmd(W, j["run_id"], train_cmd(j, hp, a.smoke, j.get("extra")), env)
                if rc == 0 and run_status(W, j["run_id"]) in ("done", "diverged_nonfinite", "diverged_kl"):
                    break
                C.public_log(W, f"{tag} exit {rc}, retry {attempt + 1}")
                time.sleep(30)
            else:
                C.public_log(W, f"{tag} FAILED - 멈춤. logs/{j['run_id']}.log 확인")
                sys.exit(1)
            C.public_log(W, f"{tag} {run_status(W, j['run_id'])}")
        elif j["kind"] == "e4b":
            pf = W / "public" / "e4b" / f"{j['run_id']}_final.json"
            if C.read_json(pf, {}).get("status") == "done":
                continue
            ck = W / "runs" / j["run_id"] / "ckpt" / j["ckpt"]
            if not ck.exists():
                C.public_log(W, f"{tag} skip: checkpoint {j['ckpt']} 없음")
                continue
            C.public_log(W, f"{tag} start")
            cmd = [PY, str(STUDY / "e4b.py"), "--run-id", j["run_id"], "--ckpt", str(ck), "--name", "final"]
            cmd += (["--base-ckpt", j["base"]] if j.get("base") else []) + (["--smoke"] if a.smoke else [])
            for attempt in range(3):
                if run_cmd(W, f"e4b_{j['run_id']}", cmd, env) == 0 and C.read_json(pf, {}).get("status") == "done":
                    break
                time.sleep(30)
            else:
                C.public_log(W, f"{tag} FAILED - 멈춤")
                sys.exit(1)
            C.public_log(W, f"{tag} done")
        elif j["kind"] == "e2a":
            pf = W / "public" / "e2a" / f"{j['job_id']}.json"
            if C.read_json(pf, {}).get("status") == "done":
                continue
            C.public_log(W, f"{tag} e2a {j['job_id']} start")
            cmd = [PY, str(STUDY / "e2a.py"), "--job-id", j["job_id"], "--lineage-ckpt", j["lineage_ckpt"], "--seed", str(j["seed"]),
                   "--point", str(j["point"]), "--sizes", j["sizes"], "--ood", j["ood"]] + (["--smoke"] if a.smoke else [])
            for attempt in range(2):
                if run_cmd(W, f"e2a_{j['job_id']}", cmd, env) == 0 and C.read_json(pf, {}).get("status") == "done":
                    break
                if C.read_json(pf, {}).get("status") == "failed":  # 분해 검사 실패는 다시 돌리지 않는다
                    break
                time.sleep(30)
            st = C.read_json(pf, {})
            C.public_log(W, f"{tag} e2a {j['job_id']} {st.get('status')} decomposition_passed={st.get('decomposition_passed')}")
            if st.get("status") != "done":
                sys.exit(1)
        elif j["kind"] == "a1_train":
            if run_status(W, j["run_id"]) in ("done", "diverged_nonfinite"):
                continue
            C.public_log(W, f"{tag} start")
            for attempt in range(3):
                rc = run_cmd(W, j["run_id"], a1_cmd(j, a.smoke), env)
                if rc == 0 and run_status(W, j["run_id"]) in ("done", "diverged_nonfinite"):
                    break
                C.public_log(W, f"{tag} exit {rc}, retry {attempt + 1}")
                time.sleep(30)
            else:
                C.public_log(W, f"{tag} FAILED - 멈춤. logs/{j['run_id']}.log 확인")
                sys.exit(1)
            C.public_log(W, f"{tag} {run_status(W, j['run_id'])}")
        elif j["kind"] == "a1_resume_check":
            if C.read_json(W / "public" / "a1_resume_check.json", {}).get("passed"):
                continue
            ok = a1_resume_check(W, j, env, a.smoke)
            C.public_log(W, f"{tag} passed={ok}")
            if not ok:
                sys.exit(1)
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
                cmd += ["--batches", "1"] + (["--conds", ",".join(["ID"] + C.EXEC_CONDS)] if j["suite"] in ("e1", "e2b") else [])
            for attempt in range(3):
                if run_cmd(W, f"eval_{j['run_id']}_{j['name']}", cmd, env) == 0 and eval_status(W, j) == "done":
                    break
                time.sleep(30)
            else:
                C.public_log(W, f"{tag} FAILED - 멈춤")
                sys.exit(1)
            C.public_log(W, f"{tag} done")
        elif j["kind"] == "select":
            out_name = j.get("out_name", "tuning_selection.json")
            if C.read_json(W / "public" / out_name):
                continue
            import select_tuning
            pub = select_tuning.main(W, plan, j["select_rounds"], arms=tuple(j.get("arms", ("A2p", "A3"))), out_name=out_name)
            C.public_log(W, f"{tag} {json.dumps(pub, ensure_ascii=False)}")
        elif j["kind"] in ("resume_check", "equiv_check"):
            pf = "resume_check.json" if j["kind"] == "resume_check" else f"equiv_check_{j['ref_run_id']}.json"
            if C.read_json(W / "public" / pf, {}).get("passed"):
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
    C.public_log(W, f"PHASE {a.phase} DONE" + (" (smoke)" if a.smoke else ""))


if __name__ == "__main__":
    main()
