"""Windows 네이티브 평가용 호환 처리 (원 RL4VLA 코드는 고치지 않는다).

`prismatic/__init__.py`는 추론에 쓰지 않는 학습 데이터 파이프라인
(`prismatic.vla.datasets` → RLDS → dlimp → tensorflow_datasets)까지 즉시 import한다.
tensorflow_datasets 4.9.3은 Unix 전용 `resource` 모듈을 import해서 Windows에서 실패한다.

추론 경로(`prismatic.extern.hf`의 모델·프로세서)는 이 데이터 파이프라인을 쓰지 않으므로,
`prismatic.vla.datasets`만 빈 대역(stub)으로 등록한다. 대역의 클래스를 실제로 쓰면 즉시 오류가 난다.
"""
import sys
import types


class _TrainingDataUnavailable:
    """학습 데이터 파이프라인 대역. 생성하면 오류를 낸다."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "prismatic.vla.datasets is stubbed on Windows (evaluation only). "
            "Training data pipelines need Linux/WSL with tensorflow_datasets."
        )


def install_prismatic_datasets_stub() -> None:
    name = "prismatic.vla.datasets"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    stub.__file__ = __file__
    stub.__path__ = []  # 패키지처럼 보이게 함 (하위 모듈 import는 실패)
    for cls_name in ("DummyDataset", "EpisodicRLDSDataset", "RLDSBatchTransform", "RLDSDataset"):
        setattr(stub, cls_name, type(cls_name, (_TrainingDataUnavailable,), {}))
    sys.modules[name] = stub
