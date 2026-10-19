"""P4 저장 정확성 (C6) — CPU, 합성 자료. 실행: python -m unittest -v pilot/p4/test_storage_cpu.py"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "learner"))
import storage  # noqa: E402


def fake_episode(i, T=80, rgb=True, H=48, W=64):
    g = np.random.default_rng(i)
    succ = g.random(T) < 0.1
    succ[-1] = i % 2 == 0
    grasp = np.zeros(T, bool)
    grasp[20:] = True
    return storage.EpisodeRecord(
        episode_id=int(g.integers(1 << 62)), instruction=f"put object {i} on plate",
        sim_state=g.standard_normal((T + 1, 200)).astype(np.float32),
        controller_state=g.standard_normal((T + 1, 7)).astype(np.float32),
        task_state=g.integers(0, 80, (T + 1, 5)).astype(np.int64),
        tokens=g.integers(31744, 32000, (T, 7)).astype(np.int64),
        action=g.standard_normal((T, 7)).astype(np.float32),
        logprob_gen=g.standard_normal(T).astype(np.float32), value_gen=g.standard_normal(T).astype(np.float32),
        success=succ, grasp_latch=grasp, consec_latch=grasp, grasp_now=grasp,
        reward_potential=g.standard_normal(T).astype(np.float32),
        rgb=g.integers(0, 256, (T + 1, H, W, 3), dtype=np.uint8) if rgb else None)


class TestStorage(unittest.TestCase):
    def test_roundtrip_exact_and_policy(self):
        with tempfile.TemporaryDirectory() as td:
            st = storage.RoundStore(td, "A3_0", 0, storage.StoragePolicy(full_rgb_rounds=(20,), keep_envs=2,
                                                                          video_every=0))
            eps = [fake_episode(i) for i in range(5)]
            for rnd in (3, 20):
                m = st.write_round(rnd, eps)
                man, t1, instr, rgb = st.read_round(rnd)
                for k in storage.TIER1_KEYS:
                    a = np.stack([getattr(e, k) for e in eps])
                    self.assertEqual(t1[k].dtype, a.dtype, k)
                    self.assertTrue(np.array_equal(t1[k], a), k)
                self.assertEqual(instr, [e.instruction for e in eps])
                self.assertEqual(t1["Y"].tolist(), [e.Y for e in eps])
                expected_kept = list(range(5)) if rnd == 20 else [0, 1]
                self.assertEqual(sorted(rgb), expected_kept)
                for i in expected_kept:
                    self.assertTrue(np.array_equal(rgb[i], eps[i].rgb))
                    self.assertEqual(rgb[i].tobytes(), eps[i].rgb.tobytes())

    def test_hash_detects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            st = storage.RoundStore(td, "A2p", 1, storage.StoragePolicy(video_every=0))
            st.write_round(5, [fake_episode(0)])
            p = st.round_dir(5) / "tier1.npz"
            b = bytearray(p.read_bytes())
            b[len(b) // 2] ^= 0xFF
            p.write_bytes(bytes(b))
            with self.assertRaises(IOError):
                st.read_round(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
