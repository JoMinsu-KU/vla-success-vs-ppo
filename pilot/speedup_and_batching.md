# 정책 추론 속도 개선과 배치 실행기 (Windows 네이티브)

- 일시: 2026-10-16 ~ 17
- 목적: P2 기준 에피소드당 30.7초(스텝당 정책 약 0.34초, 시간의 89%)를 줄여 P3·P5를 이 워크스테이션에서 돌릴 수 있게 한다.
- 원칙: 원 코드 경로와 결과가 같은지(또는 얼마나 다른지) 먼저 확인하고 쓴다.

## 1. 병목 진단 ([eval/speed_profile.py](eval/speed_profile.py))

| 측정 | 값 |
|---|---|
| 전처리 (processor) | 0.9 ms |
| `predict_action_batch` (7토큰 생성) | 364 ms |
| 연산자 호출 | 스텝당 `aten::linear` 약 5,300회 |
| GPU 사용률 (P2 실행 중) | 평균 약 26% |

생성 1회(prefill + 6회 증분 디코딩)가 작은 행렬 연산 수천 개로 이뤄져 **CPU의 커널 디스패치가 병목**이다. 원 코드는 공개 체크포인트 위에 새 LoRA(0 B)를 모든 선형층에 붙이므로 선형 연산 수가 약 3배로 늘어 있다.

## 2. 0-B LoRA 병합 ([eval/speed_equivalence_test.py](eval/speed_equivalence_test.py))

원 코드가 붙인 LoRA는 B가 모두 0이라 병합 증분 B·A·scale이 정확히 0이다. 병합 전에 B가 전부 0인지 확인하고, 하나라도 0이 아니면 병합하지 않는다 (`p2_eval.merge_zero_lora`).

| 항목 | 결과 |
|---|---|
| 병합한 LoRA 모듈 | 439개 |
| 스텝당 시간 | 357 ms → 257 ms (**1.39배**) |
| 폐루프 2 에피소드 × 80스텝, 같은 RNG 시드 | 관측·행동 토큰 **완전 일치**, log-prob·value 차이 **0**, 성공 여부 같음 |

결과: [artifacts/speedup/equivalence_rl.json](artifacts/speedup/equivalence_rl.json). 학습 중인 LoRA(B ≠ 0)에는 쓰지 않는다 (P5는 원 경로).

## 3. 환경 병렬 + 배치 추론 ([eval/batched_eval.py](eval/batched_eval.py))

Windows에는 GPU 병렬 시뮬레이션이 없으므로 CPU 시뮬레이션 환경을 별도 프로세스 N개로 돌리고, 같은 스텝의 관측을 모아 한 번에 추론한다. 정책은 원 `_preprocess_obs`·`predict_action_batch`, 환경은 원 `SimlerWrapper` 메서드를 그대로 쓴다.

| 항목 | 결과 |
|---|---|
| 환경 하나의 GPU 메모리 | 약 1.1 GB (Vulkan 렌더 컨텍스트). 작업자의 CUDA는 숨겨 CUDA 컨텍스트는 만들지 않음 |
| N = 8, 4 | GPU 메모리 초과(24 GB 중 화면 출력 약 4 GB 포함) → Windows가 시스템 메모리로 넘겨 극도로 느려짐 |
| **N = 3** | 최대 22.4 GB, **에피소드당 약 9.1초** (P2의 30.7초 대비 **3.4배**) |
| 배치 대 배치 1 수치 차이 | 첫 행동 토큰 분포 KL 평균 0.0025, 최대 로짓 차이 0.25 (배치 1 반복은 차이 0). bf16 배치 연산의 누산 순서 차이 |
| **P2와의 일치 검증** | P2와 같은 초기 상태 64개(RL, 학습 조건, seed 0): P2 57/64, 배치 실행기 55/64, 짝지은 McNemar p = 0.75 |

검증 결과: [artifacts/speedup/validate_rl.summary.json](artifacts/speedup/validate_rl.summary.json), [artifacts/speedup/batch3.json](artifacts/speedup/batch3.json).

**해석:** 배치 실행기는 비트 단위로 같지는 않지만 성공률 수준에서 원 경로와 구분되지 않는다. P3처럼 새로 재는 측정에 쓰고, 그 사실을 이탈 기록(D-003)에 남긴다. greedy 토큰 일치율(약 75~82%)은 지표로 부적절하다. 행동 분포에 거의 같은 확률의 bin이 많아 작은 수치 차이로 argmax가 바뀌고, 자기회귀로 이후 토큰이 모두 달라지기 때문이다.

## 4. 남은 속도 개선 여지 (하지 않음)

- CUDA graph·`torch.compile`: Windows torch 2.2에는 triton이 없어 보류.
- 정적 KV 캐시: prismatic 모델 수정이 필요해 보류.
- 환경 렌더 메모리: 환경 수를 3보다 늘리려면 렌더러를 CPU(lavapipe, WSL)로 옮겨야 하는데, 그러면 렌더러가 P2와 달라진다.
