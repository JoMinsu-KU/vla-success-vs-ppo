# C11 엄격 결정성 사전 확인과 C10 잔여 불일치 감사 (4090)

- 일시: 2026-10-19
- 근거: 6차 외부 검토 4·5번(엄격 모드, Adam 모멘트 상태의 A/A, 실제 attention 경로), 1번(잔여 생성–학습 불일치). 연구 명세 v0.12, 3.2절 C10·C11.
- 코드: [learner/numerics.py](learner/numerics.py) (결정적 설정·수치 경로 기록), [p4/test_strict_determinism.py](p4/test_strict_determinism.py). 결과: [p4/test_strict_determinism.json](p4/test_strict_determinism.json).
- 설정: W1 + 새 LoRA(시드 0), 에피소드 3개(240전이, Y = 0·1·1), K0, 통제 PPO(C9·C10), 미니배치 순서 시드 7. 본실험 기기(H200·flash-attn 경로) 결과가 아니다.

## 1. 실제 수치 경로

torch 2.2.0+cu121, cuDNN 8801, RTX 4090. `use_deterministic_algorithms(True, warn_only=False)`, flash·메모리 효율 SDPA 꺼짐, math SDPA만, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 꺼짐. 모델 config의 attention 구현은 `sdpa` (파일럿 어댑터가 `flash_attention_2` 요청을 바꿈).

## 2. 엄격 모드 + A/A 검사 — 통과

| 시작점 | 결정적 구현이 없는 연산 | LoRA 비트 단위 동일 | value head 비트 단위 동일 | 첫 ratio | 시간 (두 실행) |
|---|---|---|---|---|---|
| 새 LoRA·optimizer (W1 스냅샷) | 없음 (오류 없이 완료) | 예 (최대 차이 0) | 예 | 정확히 1 | 78초, 111초 |
| 한 라운드 업데이트 뒤 (Adam 모멘트가 쌓인 상태) | 없음 | 예 | 예 | 정확히 1 | 131초, 162초 |

피크 GPU 메모리 20.8 GiB. 이 경로(OpenVLA 학습 forward·backward, AdamW)에 결정적 구현이 없는 연산은 없었다. 본실험의 attention 경로(flash-attn 결정적 역전파 또는 math)에서는 G2에서 다시 확인한다.

## 3. C10 잔여 불일치 감사 (새 LoRA 시작점, 240전이)

log w = log p₀(teacher forcing) − log q₀(생성 경로), 뽑힌 행동의 7토큰 joint log-prob 차이. KL·TV가 아니다.

| 부분집합 | n | 평균 | 평균 절대값 | 95% 절대값 | 최대 | exp(−log w)가 ±0.2 밖 |
|---|---|---|---|---|---|---|
| 전체 | 240 | −0.034 | 0.187 | 0.52 | 0.73 | 35% |
| advantage > 0 | 70 | −0.088 | 0.198 | 0.53 | 0.65 | 41% |
| advantage ≤ 0 | 170 | −0.012 | 0.183 | 0.51 | 0.73 | 33% |
| Y = 1 | 160 | −0.056 | 0.176 | 0.47 | 0.73 | 34% |
| Y = 0 | 80 | +0.010 | 0.211 | 0.53 | 0.62 | 39% |

value 차이 평균 0.0013. 크기는 P4 2차(평균 0.20, 95% 0.50, 최대 0.95)와 같다. 부호별 평균이 조금 다르지만 에피소드 3개라 판단할 수 없다 — G2에서 위치별 TV·KL과 함께 충분한 표본으로 잰다.

**"한 라운드 뒤" 시작점의 감사 값(평균 절대값 1.72)은 C10 측정이 아니다.** 같은 배치를 재사용해 생성 log-prob는 W1에서, 재계산은 업데이트된 정책에서 나왔으므로 정책 변화가 섞였다. 이 시험에서는 A/A 검사에만 쓴다.
