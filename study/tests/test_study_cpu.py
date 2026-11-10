"""본실험 실행기 CPU 단위 시험 (GPU·시뮬레이터 불필요). 실행: python -m pytest study/tests -q  또는  python study/tests/test_study_cpu.py"""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import common as C  # noqa: E402


def test_n_table_matches_simulation_v3():
    # pilot/stats/GPRE_simulation_v3.md: α 0.04 n 표 경계 4.9 → 9, 5.3 → 10, 5.6 → 11, 5.9 → 12
    want = {3.0: 8, 4.8: 8, 4.9: 9, 5.2: 9, 5.3: 10, 5.5: 10, 5.6: 11, 5.8: 11, 5.9: 12, 9.0: 12}
    got = {s: C.n_from_blinded_sd(s) for s in want}
    assert got == want, got
    assert C.n_from_blinded_sd(4.94) == 9 and C.n_from_blinded_sd(4.849) == 8  # 0.1 단위 반올림


def test_config_table_text_consistent():
    t = C.CFG["n_table_alpha004"]
    assert t == {"lt_4.9": 8, "4.9-5.2": 9, "5.3-5.5": 10, "5.6-5.8": 11, "ge_5.9": 12}
    assert C.CFG["alpha_one_sided"] == 0.04 and C.CFG["delta_pp"] == 5.0
    assert C.EXEC_CONDS == ["Position", "EEPose", "PositionChangeTo"]
    assert len(C.CFG["eval_conditions"]) == 16
    labels = [c[0] for c in C.CFG["eval_conditions"]]
    assert len(set(labels)) == 16


def test_blinded_sd_is_label_invariant():
    rng = np.random.default_rng(0)
    pairs = [tuple(rng.normal(70, 5, 2)) for _ in range(4)]
    swapped = [(b, a) if i % 2 else (a, b) for i, (a, b) in enumerate(pairs)]
    assert abs(C.blinded_sd_d(pairs) - C.blinded_sd_d(swapped)) < 1e-12
    assert abs(C.blinded_sd_d([(1.0, 4.0), (2.0, 2.0)]) - (9 / 2) ** 0.5) < 1e-12


def test_power_monotone():
    assert C.paired_power(4.0, 12) > C.paired_power(4.0, 8) > C.paired_power(6.0, 8)


def test_select_tuning_rules():
    import select_tuning as S
    rows = [dict(cand="c0", lr=1e-4, update="U-prop", score=70.0), dict(cand="c1", lr=3e-4, update="U-prop", score=71.5),
            dict(cand="c2", lr=1e-4, update="U-fix", score=71.9), dict(cand="c3", lr=3e-4, update="U-fix", score=75.0)]
    assert S.choose("A2p", rows, 2.0)["cand"] == "c3"          # 2%p 밖에서 최고
    rows[3]["score"] = 72.0
    assert S.choose("A2p", rows, 2.0)["cand"] == "c0"          # 70.0은 72.0 − 2 안 → 작은 lr, U-prop
    rows[0]["score"] = None
    assert S.choose("A2p", rows, 2.0)["cand"] == "c2"          # 제외 뒤 작은 lr (U-fix)
    a3 = [dict(cand="c0", lr=5e-5, vh_lr=1e-3, score=60.0), dict(cand="c1", lr=5e-5, vh_lr=3e-3, score=60.5),
          dict(cand="c2", lr=1e-4, vh_lr=1e-3, score=61.0), dict(cand="c3", lr=1e-4, vh_lr=3e-3, score=61.0)]
    assert S.choose("A3", a3, 2.0)["cand"] == "c1"             # 작은 lr → vh 3e-3


def test_a2p_step_plans():
    sys.path.insert(0, str(C.REPO / "pilot" / "learner"))
    import plans
    idx = np.arange(37 * 80)  # 성공 37개 × 80스텝
    p = plans.a2p_step_plan(idx, "U-prop", 5, 8, 20, 32)
    assert sum(len(x) for x in p) == len(idx) and len(p) == int(np.ceil(len(idx) / 160))
    assert sorted(np.concatenate(p).tolist()) == idx.tolist()     # 한 번씩
    assert all(len(x) == 160 for x in p[:-1])
    q = plans.a2p_step_plan(idx, "U-fix", 5, 8, 20, 32)
    assert len(q) == 32 and all(len(x) == 160 for x in q) and set(np.concatenate(q)) <= set(idx)
    p2 = plans.a2p_step_plan(idx, "U-prop", 5, 8, 20, 32)
    assert all((x == y).all() for x, y in zip(p, p2))             # 라운드 시드로 결정적


def test_rollback_restores_params_and_adam_state():
    import torch
    import rollback
    torch.manual_seed(0)
    m = torch.nn.Linear(6, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    params = list(m.parameters())

    def step(seed):
        g = torch.Generator().manual_seed(seed)
        opt.zero_grad()
        (m(torch.randn(4, 6, generator=g)) ** 2).sum().backward()
        opt.step()
    step(1)
    snap = rollback.snapshot(params, opt)
    before = [p.detach().clone() for p in params]
    step(2)
    rollback.restore(params, opt, snap)
    assert all(torch.equal(p, b) for p, b in zip(params, before))
    # 되돌린 뒤의 다음 스텝 = 처음부터 step(1) → step(3)만 한 것
    step(3)
    after = [p.detach().clone() for p in params]
    torch.manual_seed(0)
    m2 = torch.nn.Linear(6, 3)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-2)
    m, opt, params = m2, opt2, list(m2.parameters())
    step(1)
    step(3)
    assert all(torch.equal(p, q) for p, q in zip(params, after))


def test_plan_structure_and_blinding():
    with tempfile.TemporaryDirectory() as d:
        os.environ["VLA_MAIN_WORK"] = d
        import runner
        for smoke in (False, True):
            plan = runner.make_plan(smoke)
            jobs = plan["jobs"]
            rids = [j["run_id"] for j in jobs if j["kind"] in ("tune_train", "e1_train", "resume_check")]
            assert len(rids) == len(set(rids))
            e1 = [j for j in jobs if j["kind"] == "e1_train"]
            for s in {j["seed"] for j in e1}:
                pair = [j for j in e1 if j["seed"] == s]
                assert len(pair) == 2 and {j["arm"] for j in pair} == {"A3", "A2p"}
                assert len({j["episode_tag"] for j in pair}) == 1   # 같은 시드 = 같은 과제 표본
            idx = runner.public_index(plan)
            assert all(set(v) == {"kind", "seed"} for v in idx.values())   # 공개 색인에 갈래 없음
            kinds = [j["kind"] for j in jobs]
            assert kinds.index("select") > max(i for i, k in enumerate(kinds) if k == "tune_train")
            assert kinds[-1] == "gmid"
            if not smoke:
                assert len([j for j in jobs if j["kind"] == "tune_train"]) == 8 and len(e1) == 8
                a3 = [j for j in e1 if j["arm"] == "A3"][0]
                assert a3["numerics"] == "math" and 20 in a3["save_rounds"] and 390 in a3["save_rounds"]
                assert all(j["numerics"] == "default" for j in jobs if j["kind"] == "tune_train")
                assert len([j for j in jobs if j["kind"] == "eval" and j["suite"] == "idval"]) == 24
            else:
                assert "resume_check" in kinds


def test_gmid_endpoint():
    import gmid
    rows = [dict(cond=c, Y=bool(i % 2)) for c in C.EXEC_CONDS for i in range(4)] + [dict(cond="ID", Y=True)]
    assert gmid.primary_endpoint(rows) == 50.0


def test_select_tuning_aux_rules():
    import select_tuning as S
    pm = [dict(cand="c0", lr=1e-4, iters=1, score=60.0), dict(cand="c1", lr=3e-4, iters=1, score=61.0),
          dict(cand="c2", lr=1e-4, iters=2, score=61.5), dict(cand="c3", lr=3e-4, iters=2, score=65.0)]
    assert S.choose("A2pm", pm, 2.0)["cand"] == "c3"
    pm[3]["score"] = 61.9
    assert S.choose("A2pm", pm, 2.0)["cand"] == "c0"           # 작은 lr → 반복 1
    pm[0]["score"] = None
    assert S.choose("A2pm", pm, 2.0)["cand"] == "c2"           # 제외 뒤 작은 lr
    a3s = [dict(cand="c0", lr=5e-5, vh_lr=1e-3, score=50.0), dict(cand="c1", lr=5e-5, vh_lr=3e-3, score=50.0)]
    assert S.choose("A3s", a3s, 2.0)["cand"] == "c1"           # A3⁰와 같은 규칙


def test_a2pm_step_plan_and_weights():
    sys.path.insert(0, str(C.REPO / "pilot" / "learner"))
    import core
    import plans
    for iters in (1, 2):
        p = plans.a2pm_step_plan(5120, iters, 7, 8, 20)
        assert len(p) == 32 * iters and all(len(x) == 160 for x in p)
        for k in range(iters):
            assert np.array_equal(np.sort(np.concatenate(p[32 * k:32 * (k + 1)])), np.arange(5120))
    assert not np.array_equal(plans.a2pm_step_plan(5120, 1, 7, 8, 20)[0], plans.a2pm_step_plan(5120, 1, 8, 8, 20)[0])
    Y = np.array([1, 0, 0, 1] + [0] * 60)
    w = core.a2pm_weights(Y)
    assert np.isclose(w[0], (1 - 1 / 63) / (2 / 64)) and np.isclose(w[1], (0 - 2 / 63) / (2 / 64))
    assert not np.any(core.a2pm_weights(np.zeros(64, int))) and not np.any(core.a2pm_weights(np.ones(64, int)))


def test_phase2_plan_structure_and_blinding():
    with tempfile.TemporaryDirectory() as d:
        os.environ["VLA_MAIN_WORK"] = d
        import runner
        W = C.work_dir()
        C.ensure_dirs(W)
        p1 = runner.make_plan(False)
        C.write_json(W / "sealed" / "plan.json", p1)
        C.write_json(W / "public" / "runs_index.json", runner.public_index(p1))
        try:
            runner.make_plan_phase2(False, W)
            raise AssertionError("gmid 없이 2단계 계획이 만들어졌다")
        except SystemExit:
            pass
        C.write_json(W / "public" / "gmid.json", dict(n=10))
        plan = runner.make_plan_phase2(False, W)
        jobs = plan["jobs"]
        e1 = [j for j in jobs if j["kind"] == "e1_train"]
        assert sorted({j["seed"] for j in e1}) == [5, 6, 7, 8, 9, 10] and len(e1) == 12
        assert all(j["sel_file"] == "tuning_selection.json" and j["numerics"] == "math" for j in e1)
        assert all(j["save_rounds"] == C.CFG[f"e1_save_rounds_{j['arm']}"] for j in e1)
        tune = [j for j in jobs if j["kind"] == "tune_train"]
        assert len(tune) == 8 and {j["arm"] for j in tune} == {"A2pm", "A3s"}
        assert all(j["seed"] == C.CFG["tune_seed"] and j["numerics"] == "default" for j in tune)
        assert {j["hp"].get("iters") for j in tune if j["arm"] == "A2pm"} == {1, 2}
        aux = [j for j in jobs if j["kind"] == "e1_aux_train"]
        assert len(aux) == 6 and sorted({j["seed"] for j in aux}) == [1, 2, 3]
        assert all(j["episode_tag"] == f"e1-seed{j['seed']}" and j["numerics"] == "math" for j in aux)
        kinds = [j["kind"] for j in jobs]
        assert kinds.index("select") > max(i for i, k in enumerate(kinds) if k == "tune_train")
        assert kinds.index("select") < min(i for i, k in enumerate(kinds) if k == "e1_aux_train")
        assert max(i for i, k in enumerate(kinds) if k == "e1_train") < min(i for i, k in enumerate(kinds) if k == "tune_train")
        rids = [j["run_id"] for j in jobs if "run_id" in j and j["kind"] != "eval"]
        assert len(rids) == len(set(rids)) and not set(rids) & set(runner.public_index(p1))
        idx = runner.public_index(plan)
        assert all(set(v) == {"kind", "seed"} for v in idx.values())
        # 스모크: 동등성 검사가 1단계 E1 실행 둘을 참조
        p1s = runner.make_plan(True)
        C.write_json(W / "sealed" / "plan.json", p1s)
        ps = runner.make_plan_phase2(True, W)
        eq = [j for j in ps["jobs"] if j["kind"] == "equiv_check"]
        refs = {j["run_id"]: j for j in p1s["jobs"] if j["kind"] == "e1_train"}
        assert len(eq) == 2 and all(j["ref_run_id"] in refs and j["arm"] == refs[j["ref_run_id"]]["arm"] for j in eq)
        assert all(j[k] == refs[j["ref_run_id"]][k] for j in eq for k in ("seed", "rounds", "numerics", "episode_tag", "save_rounds"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} passed")
