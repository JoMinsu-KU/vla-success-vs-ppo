# P2 준비: OpenVLA 설치와 환경 1개용 평가 루프 (Windows 네이티브)

- 일시: 2026-10-15 (KST)
- 근거: 연구 명세 v0.9, 6.3절 P2 (공개 체크포인트 3종 × 핵심 과제 4개 × 평가 시드 3 × 64 에피소드)
- 상태: **평가 루프 준비 완료, 체크포인트 다운로드에서 막힘.** HuggingFace CDN까지의 전송 속도가 약 0.29 MB/s여서 체크포인트 3종(약 45 GB)을 받는 데 40시간 이상 걸린다. P2 본 실행은 아직 시작하지 않았다.

## OpenVLA 설치 (`rlvla_win`)

| 패키지 | 버전 | 비고 |
|---|---|---|
| openvla (RL4VLA 포크) | 0.0.3, editable, `--no-deps` | `third_party/RL4VLA/openvla` |
| transformers / tokenizers | 4.40.1 / 0.19.1 | 포크 `pyproject.toml` 고정값 |
| timm / peft | 0.9.10 / 0.11.1 | 같음 |
| accelerate | 1.15.0 | |
| huggingface_hub | 0.36.2 | transformers 4.40.1 요구(<1.0) |
| flash-attn | **설치 안 함** | Windows 미지원. 추론은 `sdpa` attention |
| tensorflow / tensorflow_datasets / dlimp | **설치 후 제거** | 학습 데이터 파이프라인 전용. tfds 4.9.3이 Unix 전용 `resource` 모듈을 import해서 Windows에서 동작하지 않음 |

제약 파일: [scripts/constraints_openvla.txt](scripts/constraints_openvla.txt). 전체 목록: [env/rlvla_win_pip_freeze.txt](env/rlvla_win_pip_freeze.txt).

## 평가 루프: [eval/p2_eval.py](eval/p2_eval.py)

원 RL4VLA 코드를 고치지 않고 가져다 쓴다.

| 부분 | 쓰는 원 코드 |
|---|---|
| 정책 로드·행동 생성 | `OpenVLAPolicy` (`openvla_train.py`): 공개 체크포인트 + 새 LoRA(가우시안 A, 0 B → 출력은 체크포인트와 같음), `predict_action_batch`, 평가 온도 0.6 샘플링. 생성 설정은 원 코드 그대로(top_k 기본 50). P2는 원 레시피 재현이므로 v0.9의 C4 통제를 적용하지 않는다 |
| 행동 변환 | `SimlerWrapper._process_action` (토큰 → bin 중심 → `q01`·`q99` 역정규화 → 그리퍼 이진화) |
| 스텝·보상·종료 정보 | `SimlerWrapper.step` |
| 성공 판정 | 원 평가와 같이 80스텝 절단 시 `info["episode"]["success"]` (마지막 스텝) |

Windows 때문에 바꾼 것:

| # | 바꾼 것 | 이유 |
|---|---|---|
| 1 | `sim_backend` gpu → cpu, `render_backend` cpu, `num_envs` 64 → 1 | Windows는 GPU 병렬 시뮬레이션 미지원, CUDA 직결 렌더링은 충돌 (P1 기록) |
| 2 | `attn_implementation` flash_attention_2 → sdpa | flash-attn 미설치. 수치는 아주 조금 다를 수 있음 |
| 3 | `prismatic.vla.datasets` 대역 ([eval/win_compat.py](eval/win_compat.py)) | `prismatic/__init__.py`가 추론과 무관한 학습 데이터 파이프라인(TensorFlow)을 즉시 import함. 대역을 실제로 쓰면 즉시 오류 |
| 4 | 초기 상태를 `episode_id`로 직접 지정 | 원 코드는 `torch.randint`(GPU 시뮬레이션에서는 CUDA RNG)로 64개 `episode_id`를 뽑으므로 정확히 같은 초기 상태를 재현할 수 없음. 대신 (seed, 에피소드 번호)의 SHA-256에서 63비트 정수를 만들어 넘기고, 환경이 `episode_id % ltt`로 같은 균등 분포에서 고른다 |
| 5 | 행동 샘플링 시드 | 에피소드마다 `torch.manual_seed(seed·100000 + k)` |
| 6 | `USE_TF=0` | transformers가 TensorFlow를 찾지 않게 함 |

같은 (seed, 에피소드 번호)는 과제·체크포인트가 달라도 같은 `episode_id`를 쓴다. 그래서 체크포인트 간 비교가 초기 상태에 대해 짝지어진다 (원 평가에서는 보장되지 않던 성질).

기록 항목 (에피소드마다 JSONL 한 줄): 체크포인트, 과제, split, seed, 에피소드 번호, `episode_id`, 선택된 물체·접시·배경·위치·자세 번호, 지시문, `success`(원 기준), 마지막 스텝의 집기·연속 집기, `success ∧ grasp`(v0.9 Y 후보), 한 번이라도 성공했는지, 소요 시간(정책 몫 포함). 같은 출력 파일로 다시 실행하면 끝난 에피소드는 건너뛴다.

## 모델 없이 한 점검: [eval/p2_smoke_random_policy.py](eval/p2_smoke_random_policy.py)

OpenVLA 대신 무작위 행동 토큰을 내는 대역 정책으로 핵심 4과제 × 2 에피소드를 돌렸다.

- 4과제 모두 80스텝에서 절단되고 기록·영상이 저장됨.
- split 범위 확인: Carrot train의 물체 번호 3·5(학습 물체 0~15), Carrot test는 23·16(새 물체 16~24).
- 환경 몫 시간: 에피소드당 약 6.3~6.8초 (정책 몫 제외).

## 다운로드 문제

| 측정 | 값 |
|---|---|
| HF 체크포인트 파일 직접 받기 (curl, 25초) | 약 0.29 MB/s (CDN 54.169.134.11) |
| `snapshot_download` (Xet 사용 / 끔) | 2분 안에 약 256 MB 받은 뒤 사실상 멈춤 |
| 비교: PyTorch 휠(download.pytorch.org) | 수 MB/s |

다운로드 스크립트: [scripts/p2_download_ckpts.py](scripts/p2_download_ckpts.py) (캐시 `D:\VLA_ReinforceLearning\hf_cache`, 커밋 리비전과 파일별 LFS SHA-256을 `env/p2_checkpoints.json`에 기록). 이어받기가 되므로 다시 실행하면 받은 부분부터 계속한다.

## P2 본 실행 예상 시간 (체크포인트가 준비된 뒤)

정책 추론 시간은 아직 재지 못했다. 환경 몫(에피소드당 약 6.5초)에 OpenVLA-7B 추론 80회(배치 1)가 더해진다. 추론이 스텝당 0.15초라면 에피소드당 약 18.5초이고, P2 전체(3 × 4 × 3 × 64 = 2,304 에피소드)는 약 12시간이다.

```powershell
$py = "C:\Users\choiLee\anaconda3\envs\rlvla_win\python.exe"
& $py -u D:\VLA_ReinforceLearning\pilot\eval\p2_eval.py --ckpt gen-robot/openvla-7b-rlvla-rl --unnorm-key bridge_orig --seeds 0 1 2 --tasks p2 --n-episodes 64 --out D:\VLA_ReinforceLearning\pilot\artifacts\p2\p2_results.jsonl --video-dir D:\VLA_ReinforceLearning\pilot\artifacts\p2\videos --n-videos 2
```

SFT-16k는 `--unnorm-key sft`, 웜업은 `--unnorm-key bridge_orig`.
