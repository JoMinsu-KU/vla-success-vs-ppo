"""P4 단위 시험 — 통제 PPO 학습기(ppo.py)와 E2 분기(e2.py), CPU·작은 모델.

실행 (WSL rlvla_env):  python -m unittest -v pilot/p4/test_ppo_cpu.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "learner"))
import core  # noqa: E402
import e2  # noqa: E402
import ppo  # noqa: E402


class TinyVLA(torch.nn.Module):
    def __init__(self, d=16, h=32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.backbone = torch.nn.Linear(d, h)
        self.policy = torch.nn.Linear(h, 7 * 256)
        self.value_head = torch.nn.Linear(h, 1)
        for p in self.parameters():
            with torch.no_grad():
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)

    def forward(self, x, actions):
        hdn = torch.tanh(self.backbone(x))
        logits = self.policy(hdn).view(-1, 7, 256)
        lp = torch.log_softmax(logits, -1).gather(2, actions.unsqueeze(-1)).squeeze(-1)
        return lp.sum(1, keepdim=True), self.value_head(hdn)


class TinyAdapter:
    """ppo.py의 어댑터 인터페이스를 작은 모델로 구현."""

    def __init__(self, batch, seed=0, d=16):
        self.model = TinyVLA(d=d, seed=seed)
        g = torch.Generator().manual_seed(seed + 100)
        n = batch.T * batch.N
        self.x = torch.randn(n, d, generator=g)
        self.a = torch.as_tensor(batch.actions.reshape(n, 7), dtype=torch.long)
        self.params_policy = [p for n_, p in self.model.named_parameters() if "value_head" not in n_]
        self.params_value = [p for n_, p in self.model.named_parameters() if "value_head" in n_]
        self.opt_policy = torch.optim.AdamW(self.params_policy, lr=1e-3, betas=(0.9, 0.999))
        self.opt_value = torch.optim.AdamW(self.params_value, lr=3e-3, betas=(0.9, 0.999))

    def evaluate(self, idx):
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        lp, v = self.model(self.x[idx], self.a[idx])
        return lp, v

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()


def make_batch(T=8, N=20, seed=0, success_rate=0.4):
    g = np.random.default_rng(seed)
    actions = g.integers(0, 256, size=(T, N, 7))
    Ylast = g.random(N) < success_rate
    success = np.zeros((T, N), bool)
    success[-1] = Ylast
    success[-3:-1] = g.random((2, N)) < 0.3
    grasp = np.zeros((T, N), bool)
    for i in range(N):
        t0 = g.integers(1, T - 1) if (Ylast[i] or g.random() < 0.5) else T
        grasp[t0:, i] = True
    grasp[-1] |= Ylast
    consec = np.zeros((T, N), bool)
    consec[-2:] = grasp[-2:]
    return ppo.RoundBatch(actions=actions, logprob_gen=np.zeros((T, N), np.float32),
                          value_gen=np.zeros((T, N), np.float32), success=success, grasp_latch=grasp,
                          consec_latch=consec)


def fill_generation_values(adapter, batch, noise=0.0, seed=0):
    """생성 때 저장값을 흉내: 현재 정책의 값 + (선택) 잡음 (생성 경로와 teacher forcing의 차이, C10)."""
    n = batch.T * batch.N
    with torch.no_grad():
        lp, v = adapter.evaluate(np.arange(n))
    g = np.random.default_rng(seed)
    batch.logprob_gen = (lp[:, 0].numpy() + noise * g.standard_normal(n)).reshape(batch.T, batch.N).astype(np.float32)
    batch.value_gen = v[:, 0].numpy().reshape(batch.T, batch.N).astype(np.float32)


class TestAdvantageRules(unittest.TestCase):
    def test_rules_and_shared_returns(self):
        b = make_batch()
        vals = np.random.default_rng(1).normal(size=(b.T, b.N)).astype(np.float32)
        out = {r: ppo.advantages_and_returns(b, vals, "terminal", r) for r in e2.RULES.values()}
        for r in e2.RULES.values():
            np.testing.assert_array_equal(out[r][1], out["gae"][1])  # returns(→ value 손실) 네 칸 동일
        o = out["outcome"][0]
        self.assertTrue(np.all(o == o[0:1]))  # 결과 advantage는 궤적 안에서 상수
        self.assertTrue(np.all(out["outcome_masked"][0] >= 0) and np.all(out["gae_masked"][0] >= 0))
        np.testing.assert_array_equal(out["gae_masked"][0], np.maximum(out["gae"][0], 0))

    def test_reward_kinds(self):
        b = make_batch()
        rt = ppo.rewards_of(b, "terminal")
        self.assertTrue(np.all(rt[:-1] == 0))
        np.testing.assert_array_equal(rt[-1, :, 0], b.Y())
        rp = ppo.rewards_of(b, "potential")
        self.assertAlmostEqual(float(rp.sum()), float((0.1 * b.grasp_latch[-1] + 0.1 * b.consec_latch[-1]
                                                       + (b.success[-1] & b.grasp_latch[-1])).sum()), places=4)


class TestPPOUpdate(unittest.TestCase):
    def test_order_reproducible(self):
        np.testing.assert_array_equal(ppo.minibatch_order(160, 3), ppo.minibatch_order(160, 3))
        self.assertFalse(np.array_equal(ppo.minibatch_order(160, 3), ppo.minibatch_order(160, 4)))

    def test_c9_original_equals_controlled_full_round(self):
        """n = 160 (스텝 하나, 부분 묶음 없음), 재계산 없음: 원 코드형과 통제형의 업데이트 결과가 fp32 수치 오차 안에서 같다."""
        b = make_batch(T=8, N=20)
        res = {}
        for mode in ("original", "controlled"):
            ad = TinyAdapter(b, seed=0)
            fill_generation_values(ad, b)
            info = ppo.ppo_update(ad, b, ppo.PPOConfig(mode=mode, recompute_old=False, order_seed=5, chunk=3))
            res[mode] = (torch.cat([p.detach().reshape(-1) for p in ad.model.parameters()]), info)
        self.assertEqual(res["original"][1]["n_steps"], 1)
        torch.testing.assert_close(res["original"][0], res["controlled"][0], rtol=0, atol=1e-4)
        self.assertAlmostEqual(res["original"][1]["steps"][0]["loss"], res["controlled"][1]["steps"][0]["loss"], places=4)

    def test_c10_recompute_gives_ratio_one(self):
        """생성 경로 log-prob에 잡음(C10 불일치 흉내)이 있으면 첫 ratio ≠ 1, θ_old 재계산이면 정확히 1."""
        b = make_batch(T=8, N=20)
        ad = TinyAdapter(b, seed=1)
        fill_generation_values(ad, b, noise=0.2, seed=3)
        snap = ppo.snapshot(ad)
        no = ppo.ppo_update(ad, b, ppo.PPOConfig(recompute_old=False, order_seed=1))
        ppo.restore(ad, snap)
        yes = ppo.ppo_update(ad, b, ppo.PPOConfig(recompute_old=True, order_seed=1))
        self.assertFalse(no["first_minibatch"]["ratio_equal_one"])
        self.assertGreater(no["first_minibatch"]["max_abs_log_ratio"], 0.01)
        self.assertTrue(yes["first_minibatch"]["ratio_equal_one"])
        self.assertEqual(yes["first_minibatch"]["max_abs_log_ratio"], 0.0)

    def test_partial_chunk_and_steps(self):
        b = make_batch(T=8, N=25)  # 200 전이 → 25 미니배치 → 스텝 2 (20, 5)
        ad = TinyAdapter(b)
        fill_generation_values(ad, b)
        info = ppo.ppo_update(ad, b, ppo.PPOConfig(mode="controlled", order_seed=0))
        self.assertEqual([s["divisor"] for s in info["steps"]], [20, 5])
        ad2 = TinyAdapter(b)
        info2 = ppo.ppo_update(ad2, b, ppo.PPOConfig(mode="original", order_seed=0))
        self.assertEqual([s["divisor"] for s in info2["steps"]], [20, 20])
        self.assertTrue(all(s["finite"] for s in info["steps"] + info2["steps"]))

    def test_joint_grad_clip(self):
        b = make_batch(T=8, N=20)
        ad = TinyAdapter(b)
        fill_generation_values(ad, b)
        info = ppo.ppo_update(ad, b, ppo.PPOConfig(order_seed=0, grad_clip=1e-3))
        self.assertGreater(info["steps"][0]["grad_norm"], 1e-3)  # 클리핑 전 노름 기록
        self.assertTrue(info["steps"][0]["finite"])


class TestE2Branch(unittest.TestCase):
    def test_c_pm_branch_reproduces_lineage(self):
        """계보의 실제 업데이트(C± = gae)와, 분기 체크포인트에서 같은 순서 시드로 다시 한 C± 첫 업데이트가 비트 단위로 같다."""
        b = make_batch(T=8, N=40)
        ad = TinyAdapter(b, seed=2)
        fill_generation_values(ad, b)
        cfg = ppo.PPOConfig(order_seed=11)
        snap0 = ppo.snapshot(ad)
        ppo.ppo_update(ad, b, cfg)                 # 계보: 라운드 업데이트
        lineage_next = ppo.snapshot(ad)
        e2.branch_first_update(ad, b, snap0, "C±", cfg)   # 분기: 원 체크포인트에서 재시작
        d = ppo.param_diff(ad, lineage_next)
        self.assertTrue(d["bitwise_equal"], d)

    def test_other_cells_differ_and_masked_smaller(self):
        b = make_batch(T=8, N=40)
        ad = TinyAdapter(b, seed=2)
        fill_generation_values(ad, b)
        cfg = ppo.PPOConfig(order_seed=11)
        snap0 = ppo.snapshot(ad)
        outs = {}
        for name in e2.RULES:
            e2.branch_first_update(ad, b, snap0, name, cfg)
            outs[name] = ppo.snapshot(ad)
        ppo.restore(ad, outs["C±"])
        self.assertFalse(ppo.param_diff(ad, outs["C⁺"])["bitwise_equal"])
        ppo.restore(ad, outs["O±"])
        self.assertFalse(ppo.param_diff(ad, outs["O⁺"])["bitwise_equal"])

    def test_e2a_diagnostics(self):
        b = make_batch(T=8, N=40)
        ad = TinyAdapter(b, seed=4)
        snap = ppo.snapshot(ad)
        out = e2.e2a_diagnostics(ad, b, snap, ppo.PPOConfig(chunk=8))
        for row in ("O", "C"):
            for k in ("policy_grad", "total_grad", "delta_theta"):
                self.assertIsNotNone(out[row][k]["cos"])
                self.assertTrue(-1.0 <= out[row][k]["cos"] <= 1.0 + 1e-6)
        # 1.4절 (5): θ = θ_old(ratio 1)에서 O⁺ 정책 기울기 ∝ 성공 BC 기울기. O± 대 O⁺ 코사인은 1이 아니어도 됨 (기준선 항)
        self.assertLess(out["O"]["policy_grad"]["norm_ratio"], 1.5)
        self.assertEqual(ppo.param_diff(ad, snap)["bitwise_equal"], True)  # 진단 뒤 원래 상태로 복원

    def test_e2a_degenerate_batch_undefined(self):
        """전부 실패 배치: 결과 advantage가 0이라 O행 정책 기울기의 코사인은 '정의 불가'."""
        b = make_batch(T=8, N=20, success_rate=0.0)
        self.assertEqual(int(b.Y().sum()), 0)
        ad = TinyAdapter(b, seed=5)
        out = e2.e2a_diagnostics(ad, b, ppo.snapshot(ad), ppo.PPOConfig(chunk=8))
        self.assertIsNone(out["O"]["policy_grad"]["cos"])
        self.assertTrue(out["batch"]["degenerate"])
        self.assertIsNotNone(out["C"]["policy_grad"]["cos"])  # GAE 행은 그런 배치에서도 0이 아님


if __name__ == "__main__":
    unittest.main(verbosity=2)
