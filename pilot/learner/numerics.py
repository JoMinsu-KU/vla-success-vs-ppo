"""C11 결정적 수치 설정과 실제 실행 경로 기록 (연구 명세 v0.12, 3.2절 C11).

결정적 설정은 난수 시드 고정이 아니라 attention 커널·연산 경로를 바꾸는 것이다. 결과는 이 구현 조건에 대한 것이며
다른 커널과의 성능 동일성을 가정하지 않는다 (6차 검토 4번).
"""
import os

import torch


def enable_deterministic(strict: bool = True):
    """strict=True: 결정적 구현이 없는 연산을 만나면 오류 (G2 검증 모드). False: 경고만 (파일럿 P4)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def numerics_report(model=None) -> dict:
    """실제로 선택된 수치 경로. 설정 이름만으로 FlashAttention 실행을 판단하지 않기 위해 기록한다."""
    r = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
             device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
             deterministic=torch.are_deterministic_algorithms_enabled(),
             deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
             sdp_flash=torch.backends.cuda.flash_sdp_enabled(), sdp_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
             sdp_math=torch.backends.cuda.math_sdp_enabled(), cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
             tf32_matmul=torch.backends.cuda.matmul.allow_tf32)
    if model is not None:
        impls = set()
        for m in model.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                impls.add(str(cfg._attn_implementation))
        r["attn_implementation"] = sorted(impls)
    return r
