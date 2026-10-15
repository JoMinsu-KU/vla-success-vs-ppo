"""P2 평가 루프의 모델 이외 부분 점검 (체크포인트 없이).

OpenVLA 대신 무작위 행동 토큰(32000-256 ~ 31999)을 내는 대역 정책으로 run_cell을 돌려
환경 생성·episode_id 전달·원 래퍼의 행동 변환·80스텝 종료·기록 형식을 확인한다.
수치(성공률)는 의미 없다.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_eval  # noqa: E402
import torch  # noqa: E402


class RandomTokenPolicy:
    def get_action(self, x, deterministic):
        b = x["image"].shape[0]
        action = torch.randint(32000 - 256, 32000, (b, 7))
        return torch.zeros(b, 1), action, torch.zeros(b, 1)


# bridge_orig 행동 통계와 같은 형식의 대역 (q01·q99, 그리퍼 마스크 없음)
unnorm_state = {
    "q01": [-0.03, -0.04, -0.03, -0.1, -0.1, -0.2, 0.0],
    "q99": [0.03, 0.04, 0.03, 0.1, 0.1, 0.2, 1.0],
    "mask": [True, True, True, True, True, True, False],
}
out = Path(sys.argv[1])
a = SimpleNamespace(ckpt="RANDOM_TOKEN_POLICY", unnorm_key="stub", n_episodes=2, n_videos=1,
                    temperature_eval=0.6, video_dir=str(out.parent / "videos"))
for env_id, obj_set in p2_eval.P2_TASKS:
    p2_eval.run_cell(RandomTokenPolicy(), unnorm_state, a, env_id, obj_set, seed=0, out=out)
