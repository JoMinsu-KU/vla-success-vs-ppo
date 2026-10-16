"""P4 단위 시험 (GPU·모델 불필요 부분). 연구 명세 v0.9, 6.3절 P4 표.

실행 (WSL rlvla_env):  python -m unittest -v pilot/p4/test_core_cpu.py
모델이 필요한 시험(C9의 손실 경로)은 OpenVLA 대신 같은 구조(공유 backbone + 7토큰 정책 + value head)의
작은 모델로 확인한다. 실제 모델·전체 규모 확인은 GPU 시험(별도)과 G1 전 H200 제한 사용에서 한다.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "learner"))
import core  # noqa: E402

T = 80


class TinyVLA(torch.nn.Module):
    """공유 backbone → (7토큰 × 256 행동 로짓, value). OpenVLA의 손실 경로 구조만 흉내 낸다."""

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
        lp = torch.log_softmax(logits.float(), -1).gather(2, actions.unsqueeze(-1)).squeeze(-1)  # [B,7]
        return lp.sum(1, keepdim=True), self.value_head(hdn)  # joint log-prob [B,1], value [B,1]


def make_batch(B=8, d=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, d, generator=g)
    actions = torch.randint(0, 256, (B, 7), generator=g)
    adv = torch.randn(B, 1, generator=g)
    returns = torch.randn(B, 1, generator=g)
    value_preds = torch.randn(B, 1, generator=g) * 0.1
    return x, actions, adv, returns, value_preds


class TestRewardsAndLabels(unittest.TestCase):
    """C5: terminal 보상, 원 잠재함수 보상, 절단 마스크."""

    def _traj(self, success_steps):
        s = np.zeros((T, 1), bool)
        for t in success_steps:
            s[t - 1] = True
        return s

    def test_terminal_same_Y_same_return(self):
        Y = np.array([1, 1, 0, 0])
        R = core.discounted_return(core.terminal_rewards(Y, T))
        self.assertAlmostEqual(R[0], R[1])
        self.assertAlmostEqual(R[0], 0.99 ** (T - 1))
        self.assertEqual(R[2], 0.0)

    def test_potential_reward_counterexample(self):
        """3.2절 반례: 10~79스텝 성공 유지 후 80스텝에 잃은 궤적(Y=0)이 80스텝 첫 성공(Y=1)보다 수익이 크다."""
        g = np.ones((T, 1), bool)  # 집기·연속 집기 래치가 처음부터 켜졌다고 둔다 (두 궤적에 같은 상수)
        a = self._traj(range(10, 80))  # Y = 0
        b = self._traj([80])           # Y = 1
        Ra = core.discounted_return(core.potential_rewards(g, g, a))[0]
        Rb = core.discounted_return(core.potential_rewards(g, g, b))[0]
        self.assertAlmostEqual(Ra - Rb, (0.99 ** 9 - 0.99 ** 79) - 0.99 ** 79, places=6)
        self.assertGreater(Ra, Rb)
        Ta = core.discounted_return(core.terminal_rewards(a[-1].astype(int), T))[0]
        Tb = core.discounted_return(core.terminal_rewards(b[-1].astype(int), T))[0]
        self.assertLess(Ta, Tb)

    def test_truncation_mask_ignores_V_T(self):
        """80스텝 절단은 마스크 0으로 종결: V(s_80)를 바꿔도 반환값·advantage가 같다."""
        rng = np.random.default_rng(0)
        r = core.terminal_rewards(np.array([1, 0, 1]), T)[..., None]
        v = rng.normal(size=(T + 1, 3, 1)).astype(np.float32)
        m = np.ones((T + 1, 3, 1), np.float32)
        m[T] = 0
        r1, a1 = core.original_gae(r, v, m)
        v2 = v.copy()
        v2[T] += 123.0
        r2, a2 = core.original_gae(r, v2, m)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(a1, a2)


class TestOutcomeAdvantage(unittest.TestCase):
    """3.6절 E2 결과 advantage (P4 표 'E2 결과 advantage' (i)~(iv))."""

    def test_constant_within_trajectory_and_masked_constant(self):
        Y = np.array([1, 0, 0, 1, 0, 0, 0, 1])
        A = core.outcome_advantage(Y)
        self.assertTrue(np.all(A == A[0:1]))                     # (i) 궤적 안에서 상수
        Ap = core.mask_negative(A)
        succ = Ap[0, Y == 1, 0]
        self.assertTrue(np.allclose(succ, succ[0]) and succ[0] > 0)  # (ii) 성공 궤적에 같은 양의 상수
        self.assertTrue(np.all(Ap[0, Y == 0, 0] == 0))
        m = Y.mean()
        self.assertAlmostEqual(succ[0], (1 - m) / (math.sqrt(m * (1 - m)) + 1e-8), places=5)

    def test_all_success_or_failure_is_zero(self):
        for Y in (np.ones(8, int), np.zeros(8, int)):
            self.assertTrue(np.all(core.outcome_advantage(Y) == 0))  # (iii)

    def test_population_std_matches_batch_standardization(self):
        """원 코드식 버퍼 정규화(모집단 std)를 결과 advantage에 적용하면 Ã^O와 같다 (ε 차이만)."""
        Y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 0])
        raw = np.broadcast_to(Y[None, :, None].astype(np.float32), (T, len(Y), 1))
        np.testing.assert_allclose(core.normalize_adv(raw), core.outcome_advantage(Y), atol=1e-4)

    def test_pooled_normalization_differs_from_per_batch(self):
        """(iv) 진단 크기별로 '합친 뒤 한 번 정규화'해야 한다: 배치별 정규화 평균과 다르다."""
        Y1, Y2 = np.array([1, 0, 0, 0]), np.array([1, 1, 1, 0])
        pooled = core.outcome_advantage(np.concatenate([Y1, Y2]))[0, :, 0]
        separate = np.concatenate([core.outcome_advantage(Y1)[0, :, 0], core.outcome_advantage(Y2)[0, :, 0]])
        self.assertFalse(np.allclose(pooled, separate))

    def test_identity_max_Y_minus_b(self):
        """1.4절 (5): 공통 상수 b에서 max(Y − b, 0) = (1 − b)Y, 모든 표본에서."""
        Y = np.array([1, 0, 1, 1, 0])
        for b in (0.1, 0.37, 0.9):
            np.testing.assert_allclose(np.maximum(Y - b, 0), (1 - b) * Y)

    def test_mask_after_normalization_order(self):
        """정규화 → 마스킹 (재정규화 없음). 마스킹 뒤 재정규화하면 값이 달라진다."""
        rng = np.random.default_rng(1)
        a = rng.normal(size=(T, 8, 1)).astype(np.float32)
        ours = core.mask_negative(core.normalize_adv(a))
        renorm = core.normalize_adv(core.mask_negative(core.normalize_adv(a)))
        self.assertTrue(np.all(ours >= 0))
        self.assertFalse(np.allclose(ours, renorm))


class TestLikelihoodWeights(unittest.TestCase):
    def test_a2pm_loo_proportional(self):
        """Y_i − b_{−i} = N/(N−1)·(Y_i − Ȳ)."""
        Y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
        N, m = len(Y), Y.mean()
        np.testing.assert_allclose(core.a2pm_weights(Y) * m, N / (N - 1) * (Y - m))

    def test_degenerate_batches_skip(self):
        for Y in (np.ones(6, int), np.zeros(6, int)):
            self.assertTrue(np.all(core.a2pm_weights(Y) == 0))
        self.assertTrue(np.all(core.a2p_weights(np.zeros(6, int)) == 0))


class TestStepSchedule(unittest.TestCase):
    """C7·3.3절: 라운드당 스텝 수, 부분 묶음 스케일, U-fix."""

    def test_steps(self):
        self.assertEqual(core.steps_per_round(5120), 32)
        self.assertEqual(core.steps_per_round(320), 2)   # 성공 궤적 4개
        self.assertEqual(core.steps_per_round(240), 2)   # 성공 3개: ⌈1.5⌉

    def test_partial_chunk_divisors(self):
        self.assertEqual(core.chunk_loss_divisors(240, controlled=False), [20, 20])  # 원: 마지막 묶음 과소 스케일
        self.assertEqual(core.chunk_loss_divisors(240, controlled=True), [20, 10])
        self.assertEqual(core.chunk_loss_divisors(5120, controlled=True), [20] * 32)

    def test_u_fix(self):
        idx = core.u_fix_indices(240, rng=np.random.default_rng(0))
        self.assertEqual(len(idx), 32)
        self.assertTrue(all(len(i) == 160 and i.max() < 240 for i in idx))
        self.assertEqual(core.u_fix_indices(0), [])


class TestStabilityGuards(unittest.TestCase):
    def test_kl_estimate(self):
        lp = torch.randn(100)
        self.assertAlmostEqual(core.kl_estimate(lp, lp), 0.0, places=7)
        self.assertGreater(core.kl_estimate(lp, lp - 0.5), 0.0)
        self.assertTrue(core.should_stop_round(0.06, 0.05))

    def test_divergence_rule(self):
        self.assertTrue(core.is_diverged(True, 0, 1.0, 1.0))
        self.assertTrue(core.is_diverged(False, 10, 2.1, 1.0))
        self.assertFalse(core.is_diverged(False, 9, 3.0, 1.0))
        self.assertFalse(core.is_diverged(False, 12, 1.9, 1.0))


class TestC9LossScale(unittest.TestCase):
    """C9 동일성 시험: 원 코드형 계산(joint log-prob 7칸 복제) 대 통제 학습기 계산(joint 1개 × 정책 계수 7)."""

    def _losses(self, dtype):
        torch.manual_seed(0)
        out = {}
        for mode in ("original", "controlled", "v06_value_over_7"):
            model = TinyVLA()
            x, actions, adv, returns, vpred = make_batch()
            with torch.no_grad():
                old_lp, _ = model(x, actions)
                noise = torch.randn(old_lp.shape, generator=torch.Generator().manual_seed(9))  # 모드마다 같은 잡음
                old_lp = old_lp - 0.05 * noise  # ratio ≠ 1이 되도록
            lp, v = model(x, actions)
            lp, adv_t, old_t = lp.to(dtype), adv.to(dtype), old_lp.to(dtype)
            vl = core.value_loss_original(v.float(), vpred, returns)
            if mode == "original":
                loss = core.policy_loss_original(lp, old_t.repeat(1, 7), adv_t).float() + vl
            elif mode == "controlled":
                loss = core.POLICY_COEF * core.policy_loss_controlled(lp, old_t, adv_t).float() + core.VALUE_COEF * vl
            else:
                loss = core.policy_loss_controlled(lp, old_t, adv_t).float() + vl / 7
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            gc = torch.cat([p.grad.flatten() for p in model.parameters()]).clone()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
            opt.step()
            params = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
            out[mode] = dict(loss=loss.item(), grad=g, grad_norm=float(gn), grad_clipped=gc, params=params)
        return out

    def test_identity_fp32(self):
        o = self._losses(torch.float32)
        a, b = o["original"], o["controlled"]
        self.assertAlmostEqual(a["loss"], b["loss"], places=5)
        torch.testing.assert_close(a["grad"], b["grad"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(a["grad_clipped"], b["grad_clipped"], rtol=1e-5, atol=1e-6)
        # AdamW 1회 업데이트: 기울기가 ε(1e-8) 근처인 원소에서는 Adam의 g/(|g|+ε)가 7칸 합과 ×7의 1e-7 수준
        # 반올림 차이를 키운다 (첫 실행: 59,713개 중 3개, 최대 2e-5). 허용 오차는 lr(1e-2)의 1%.
        torch.testing.assert_close(a["params"], b["params"], rtol=0, atol=1e-4)
        frac = (a["params"] - b["params"]).abs().gt(1e-6).float().mean().item()
        self.assertLess(frac, 1e-3)

    def test_identity_bf16_policy_path(self):
        """원 코드는 ratio·advantage를 bf16으로 계산한다. bf16에서는 7칸 합과 ×7의 반올림이 달라 근사 일치만 기대."""
        o = self._losses(torch.bfloat16)
        a, b = o["original"], o["controlled"]
        self.assertLess(abs(a["loss"] - b["loss"]) / max(1e-6, abs(a["loss"])), 2e-2)
        cos = torch.nn.functional.cosine_similarity(a["grad"], b["grad"], dim=0).item()
        self.assertGreater(cos, 0.999)
        self.bf16_report = dict(loss_rel_diff=abs(a["loss"] - b["loss"]) / abs(a["loss"]), grad_cos=cos)

    def test_v06_alternative_differs(self):
        """v0.6의 'value 계수 1/7'은 비율은 같지만 총 기울기가 1/7이라 같은 결과가 아니다 (v0.7에서 대체한 이유)."""
        o = self._losses(torch.float32)
        cos = torch.nn.functional.cosine_similarity(o["original"]["grad"], o["v06_value_over_7"]["grad"], dim=0).item()
        self.assertGreater(cos, 1 - 1e-6)  # 클리핑 전에는 방향이 같다 (총 손실이 정확히 1/7배)
        self.assertAlmostEqual(o["v06_value_over_7"]["grad_norm"] * 7, o["original"]["grad_norm"], places=3)
        self.assertFalse(torch.allclose(o["original"]["params"], o["v06_value_over_7"]["params"]))


class TestOPlusProportionalToBC(unittest.TestCase):
    """1.4절 (5)·C8 (iii): ratio = 1(θ = θ_old)에서 O⁺ 정책 손실 기울기 ∝ 성공 궤적 NLL(BC) 기울기."""

    def test_cosine_one(self):
        model = TinyVLA(seed=3)
        B = 12
        x, actions, *_ = make_batch(B=B, seed=4)
        Y = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0])
        A = torch.tensor(core.mask_negative(core.outcome_advantage(Y, T=1))[0])  # [B,1]
        lp, _ = model(x, actions)
        core.policy_loss_controlled(lp, lp.detach(), A).backward()
        g_ppo = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        model.zero_grad()
        lp, _ = model(x, actions)
        (-(lp[torch.tensor(Y == 1)]).mean()).backward()
        g_bc = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        cos = torch.nn.functional.cosine_similarity(g_ppo, g_bc, dim=0).item()
        self.assertGreater(cos, 1 - 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
