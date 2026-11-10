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
        e4 = [j for j in jobs if j["kind"] == "e4b"]
        p1_e1 = [j["run_id"] for j in p1["jobs"] if j["kind"] == "e1_train"]
        assert [j["run_id"] for j in e4[:len(p1_e1)]] == p1_e1 and kinds[:len(p1_e1)] == ["e4b"] * len(p1_e1)
        finals = [j["run_id"] for j in jobs if j["kind"] == "eval" and j["suite"] == "e1"]
        assert {j["run_id"] for j in e4} == set(p1_e1) | set(finals)
        for i, j in enumerate(jobs):
            if j["kind"] == "eval" and j["suite"] == "e1":
                assert jobs[i + 1]["kind"] == "e4b" and jobs[i + 1]["run_id"] == j["run_id"]
        a1 = [j for j in jobs if j["kind"] == "a1_train"]
        assert sorted(j["seed"] for j in a1) == [1, 2, 3] and all(j["steps"] == 60000 for j in a1)
        assert min(i for i, k in enumerate(kinds) if k == "a1_train") > max(i for i, k in enumerate(kinds) if k == "e1_aux_train")
        assert all(any(e["kind"] == "eval" and e["run_id"] == j["run_id"] and e["suite"] == "e1" for e in jobs) for j in a1)
        rids = [j["run_id"] for j in jobs if "run_id" in j and j["kind"] not in ("eval", "e4b")]
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
        a1s = [j for j in ps["jobs"] if j["kind"] == "a1_train"]
        rc = [j for j in ps["jobs"] if j["kind"] == "a1_resume_check"]
        assert len(a1s) == 1 and a1s[0]["steps"] == 20 and len(rc) == 1 and rc[0]["ref_run_id"] == a1s[0]["run_id"]


def test_a1_prepare_rules():
    import a1_prepare as P
    files = [f"f{k:05d}.npz" for k in range(16400)]
    sp = P.split_files(files)
    assert sp["train"] == files[:16384] and sp["val"] == files[16384:16400]
    stats = dict(q01=[-0.1] * 6 + [0.0], q99=[0.1] * 6 + [1.0], mask=[True] * 6 + [False])
    acts = np.array([[0.0, 0.05, -0.05, 0.1, -0.1, 0.3, 1.0], [0.01, -0.02, 0.0, 0.0, 0.0, -0.3, -1.0]])
    tok = P.tokenize(acts, stats)
    assert tok.min() >= 31744 and tok.max() <= 31999
    back = P.detokenize(tok, stats)
    assert np.all(np.abs(back[:, :5] - acts[:, :5]) <= 0.2 / 255 / 2 + 1e-9)          # 범위 안: 반 bin 이내
    assert np.allclose(back[:, 5], [0.1 * (1 - 1 / 255), -0.1 * (1 - 1 / 255)], atol=1e-3)  # 범위 밖은 끝 bin으로
    assert list((back[:, 6] > 0.5)) == [True, False]                                     # 그리퍼 열림/닫힘 보존
    m = P.filter_small_actions(np.array([[0, 0, 0, 0, 0, 0, 1], [0.02, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, -1],
                                         [0, 0, 0, 0, 0, 0.1, -1], [0.001, 0, 0, 0, 0, 0, -1]], float))
    assert list(m) == [False, True, True, True, False]


def test_phase3_e2b_plan():
    with tempfile.TemporaryDirectory() as d:
        os.environ["VLA_MAIN_WORK"] = d
        import runner
        W = C.work_dir()
        C.ensure_dirs(W)
        p1 = runner.make_plan(False)
        C.write_json(W / "sealed" / "plan.json", p1)
        C.write_json(W / "public" / "runs_index.json", runner.public_index(p1))
        C.write_json(W / "public" / "gmid.json", dict(n=8))
        p2 = runner.make_plan_phase2(False, W)
        C.write_json(W / "sealed" / "plan_phase2.json", p2)
        try:
            runner.make_plan_phase3(False, W, False)
            raise AssertionError("E1이 끝나기 전에 3단계 계획이 만들어졌다")
        except SystemExit:
            pass
        core = [j for pl in (p1, p2) for j in pl["jobs"] if j["kind"] == "e1_train"]
        for j in core:
            C.write_json(W / "public" / "runs" / f"{j['run_id']}.json", dict(status="done"))
        p3 = runner.make_plan_phase3(False, W, False)
        br = [j for j in p3["jobs"] if j["kind"] == "e2b_train"]
        assert len(br) == 8 * 3 * 4
        a3 = {j["seed"]: j for j in core if j["arm"] == "A3"}
        for j in br:
            lin = a3[j["seed"]]
            assert j["lineage"] == lin["run_id"] and j["episode_tag"] == lin["episode_tag"] and j["numerics"] == "math"
            assert j["rounds"] == j["point"] - 1 + 20 and f"r{j['point'] - 1:03d}.pt" in j["extra"][1]
            assert ("--check-ckpt" in j["extra"]) == (j["rule"] == "C±")
            assert j["point"] - 1 in C.CFG["e1_save_rounds_A3"] and j["point"] in C.CFG["e1_save_rounds_A3"]
        assert len([j for j in p3["jobs"] if j["kind"] == "eval" and j["suite"] == "e2b"]) == len(br)
        e2a_jobs = [j for j in p3["jobs"] if j["kind"] == "e2a"]
        assert len(e2a_jobs) == 24 and all(j["sizes"] == "1,2,4,8" and j["lineage_ckpt"].endswith(f"r{j['point'] - 1:03d}.pt") for j in e2a_jobs)
        kinds = [j["kind"] for j in p3["jobs"]]
        assert kinds[0] == "e2a"
        p3b = runner.make_plan_phase3(False, W, True)
        plus = [j for j in p3b["jobs"] if j.get("variant") == "A3plus"]
        assert len(plus) == 5 and all(j["extra"] == ["--adv-rule", "gae_masked"] and j["rounds"] == 390 for j in plus)


def test_e2a_affine_decomposition_premise():
    sys.path.insert(0, str(C.REPO / "pilot" / "learner"))
    import ppo
    rng = np.random.default_rng(3)
    T, m = 80, 3
    recs = []
    for b in range(m):
        succ = np.zeros((T, 64), bool)
        succ[-1] = rng.random(64) < 0.4
        recs.append(dict(actions=np.zeros((T, 64, 7), np.int64), lp_gen=np.zeros((T, 64), np.float32), v_gen=np.zeros((T, 64), np.float32),
                         success=succ, grasp_latch=succ.copy(), consec_latch=succ.copy(), v=rng.normal(0, 0.3, (T, 64)).astype(np.float32)))
    cat = lambda k: np.concatenate([r[k] for r in recs], axis=1)  # noqa: E731
    rb = ppo.RoundBatch(actions=cat("actions"), logprob_gen=cat("lp_gen"), value_gen=cat("v_gen"), success=cat("success"),
                        grasp_latch=cat("grasp_latch"), consec_latch=cat("consec_latch"))
    v = cat("v")
    raw = np.concatenate([ppo.advantages_and_returns(ppo.RoundBatch(actions=r["actions"], logprob_gen=r["lp_gen"], value_gen=r["v_gen"],
                                                                     success=r["success"], grasp_latch=r["grasp_latch"],
                                                                     consec_latch=r["consec_latch"]), r["v"], "terminal", "gae")[1][:, :, 0]
                          - r["v"] for r in recs], axis=1)
    Y = np.broadcast_to(rb.Y().astype(np.float64)[None, :], (T, 64 * m))

    def fit(adv, u):
        A = np.stack([u.reshape(-1), np.ones(u.size)], 1)
        c, *_ = np.linalg.lstsq(A, adv.reshape(-1).astype(np.float64), rcond=None)
        return float(np.abs(A @ c - adv.reshape(-1)).max())
    get = lambda rule: ppo.advantages_and_returns(rb, v, "terminal", rule)[0][:, :, 0]  # noqa: E731
    assert fit(get("gae"), raw) < 1e-4                 # 배치별 GAE(정규화 전)를 합친 버퍼의 공식 C± = 1차식
    assert fit(get("outcome"), Y) < 1e-4
    assert fit(get("outcome_masked"), Y) < 1e-4       # O⁺ = 성공에만 양의 상수
    assert fit(get("gae_masked"), raw) > 1e-2          # C⁺는 1차식이 아니다 → 직접 계산


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} passed")
