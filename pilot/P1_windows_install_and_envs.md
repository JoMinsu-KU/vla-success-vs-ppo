# P1 RL4VLA 설치와 평가 환경 16개 생성·렌더링 (Windows 네이티브)

- 일시: 2026-10-15 (KST)
- 근거: 연구 명세 v0.9, 6.3절 P1 절차 1·3. ManiSkill 파일럿 경로는 Windows 네이티브로 정했다 ([P1b_windows_native_rendering.md](P1b_windows_native_rendering.md)).
- 판정: **통과 — 16/16 조건 생성·리셋·렌더링·스텝 성공** (6.3절 P1 통과 기준 "16개 조건이 모두 생성·렌더링된다").
- 아직 하지 않은 것: OpenVLA 설치(`openvla` 패키지, transformers 4.40.1), 생성 설정 기록(P1 절차 4), 공개 체크포인트 평가(P2). flash-attn은 Windows에서 쓰지 않을 예정이다.

## RL4VLA

| 항목 | 값 |
|---|---|
| 저장소 | `gen-robot/RL4VLA`, `third_party/RL4VLA`에 클론 (`.gitignore`로 저장소 밖) |
| 커밋 | `de46e0174e7ac310255a2836718fcefab28b8963` (2025-08-25, "fix: prep_training vla.train()") — 명세의 `de46e01`과 일치 |
| 설치 | `ManiSkill`(포크), `SimplerEnv`를 `pip install -e`로 설치. PyPI판 mani_skill 3.0.1·sapien 3.0.3은 제거 |

## `rlvla_win` 최종 구성

| 패키지 | 버전 | 출처·비고 |
|---|---|---|
| Python | 3.10.22 | Anaconda 기본 채널 (서명됨) |
| torch / torchvision | 2.2.0+cu121 / 0.17.0+cu121 | download.pytorch.org |
| numpy | 1.26.4 | **PyPI 휠** (아래 OpenMP 문제 때문에 conda-forge판을 PyPI판으로 덮어씀) |
| sapien | 3.0.0b1 | 포크의 `setup.py` 고정값 |
| gymnasium | 0.29.1 | 포크의 `setup.py` 고정값 |
| mani_skill | 포크 (editable) | `third_party/RL4VLA/ManiSkill` |
| simpler_env | 0.0.1 (editable) | `third_party/RL4VLA/SimplerEnv` |
| pinocchio | 3.2.0 | **conda-forge** (Windows용 PyPI 휠 없음). CPU 시뮬레이션의 EE 포즈 제어기가 필요로 함 |
| setuptools | 69.5.1 | 포크 `setup.py`가 `pkg_resources`를 써서 70 미만으로 고정 |

전체 목록: [env/rlvla_win_pip_freeze.txt](env/rlvla_win_pip_freeze.txt), [env/rlvla_win_conda_explicit.txt](env/rlvla_win_conda_explicit.txt)

## 자산

| 자산 | 위치 | 비고 |
|---|---|---|
| `bridge_v2_real2sim` (78 MB) | `C:\Users\choiLee\.maniskill\data\tasks\bridge_v2_real2sim_dataset` | ManiSkill 공식 HF 데이터셋. custom 모델 51개 |
| `widowx250s` v0.2.0 | `C:\Users\choiLee\.maniskill\data\robots\widowx` | 다운로드 도구가 Windows에서 이름 바꾸기에 실패(빈 `widowx` 폴더가 먼저 생김) → 빈 폴더를 지우고 수동으로 이름 바꿈 |

## 해결한 문제 (재현 시 참고)

1. **포크 editable 설치 실패** (`ModuleNotFoundError: pkg_resources`): 빌드 격리 환경의 최신 setuptools에 `pkg_resources`가 없음 → `setuptools<70` 설치 후 `--no-build-isolation`.
2. **모든 환경이 리셋에서 실패** (`'NoneType' object is not callable`): CPU 시뮬레이션에서는 `pd_ee_pose` 제어기의 기구학을 pinocchio로 계산한다 (`kinematics.py` `_setup_cpu`). GPU 시뮬레이션은 pytorch_kinematics를 쓰므로 원 Linux 환경에서는 드러나지 않는 의존성이다. SAPIEN 내장 pinocchio 모듈은 Windows판에 없음 → conda-forge `pinocchio` 설치 (`--freeze-installed`, `numpy=1.26`, 사전 dry-run으로 python 교체 없음 확인).
3. **OpenMP 런타임 충돌로 프로세스 중단** (`OMP: Error #15 ... libomp.dll ... libiomp5md.dll`): conda-forge numpy(LLVM OpenMP)와 torch(Intel OpenMP)가 한 프로세스에 올라감. 공식적으로 안전하지 않은 `KMP_DUPLICATE_LIB_OK=TRUE` 대신 numpy를 PyPI 휠로 강제 재설치(`--force-reinstall --no-deps numpy==1.26.4`)해 해결.

## 시험 설정

스크립트: [scripts/p1_env_check.py](scripts/p1_env_check.py). 원 `SimplerEnv/simpler_env/env/simpler_wrapper.py`와 같은 설정을 쓰고, Windows에서 되는 경로로만 바꿨다.

| 설정 | 원 래퍼 | 이 시험 |
|---|---|---|
| `obs_mode` | `rgb+segmentation` | 같음 |
| `control_mode` | `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` | 같음 |
| `sim_config` | sim 500 Hz, control 5 Hz | 같음 |
| `sensor_configs` | `shader_pack="default"` | 같음 |
| `max_episode_steps` | 80 | 같음 |
| `sim_backend` | `gpu` | **`cpu`** (Windows는 GPU 병렬 시뮬레이션 미지원) |
| `render_backend` | 기본(`gpu`) | **`cpu`** (RTX 4090 Vulkan 렌더링, CPU로 읽음) |
| `num_envs` | 64 | **1** |

각 조건: `gym.make` → `reset(seed=[0])` → `reset(options={"obj_set": split})` → `3rd_view_camera` RGB 저장 → 0 행동 5스텝.

## 결과

| # | 환경 ID | split | 축 | 생성 (s) | 스텝 FPS | 지시문 (seed 0) |
|---|---|---|---|---|---|---|
| 1 | Carrot-v1 | train | ID | 2.65 | 14.1 | put watering can on plate |
| 2 | VisionImage-v1 | test | 시각 | 1.50 | 14.1 | put plant on plate |
| 3 | VisionTexture03-v1 | test | 시각 | 1.49 | 14.1 | put toy bear on plate |
| 4 | VisionTexture05-v1 | test | 시각 | 1.55 | 12.9 | put watering can on plate |
| 5 | VisionWhole03-v1 | test | 시각 | 1.73 | 13.9 | put ketchup bottle on plate |
| 6 | VisionWhole05-v1 | test | 시각 | 1.77 | 14.0 | put banana on plate |
| 7 | Carrot-v1 | test | 의미 | 1.70 | 14.6 | put pepper on plate |
| 8 | Plate-v1 | test | 의미 | 1.96 | 13.0 | put toy bear on notepad |
| 9 | Instruct-v1 | test | 의미 | 1.70 | 13.3 | Can you put watering can on plate? |
| 10 | MultiCarrot-v1 | test | 의미 | 1.88 | 14.0 | put cup on plate |
| 11 | MultiPlate-v1 | test | 의미 | 2.05 | 13.7 | put golf ball on flat bowl |
| 12 | Position-v1 | test | 실행 | 1.79 | 13.4 | put watering can on plate |
| 13 | EEPose-v1 | test | 실행 | 1.58 | 16.6 | put ketchup bottle on plate |
| 14 | PositionChangeTo-v1 | test | 실행 | 1.86 | 14.3 | put BBQ sauce on plate |
| 15 | MultiCarrot-v1 | train | 학습 분포 변형 | 1.81 | 14.1 | put zuchinni on plate |
| 16 | MultiPlate-v1 | train | 학습 분포 변형 | 1.85 | 12.8 | put pipe on plate |

모든 프레임은 640×480×3 uint8이다. 원자료: [artifacts/p1_envs/p1_env_check.json](artifacts/p1_envs/p1_env_check.json). 16개 프레임을 4×4로 모은 그림 (위 표 순서, 왼쪽 위부터): [artifacts/p1_envs/p1_grid_4x4.jpg](artifacts/p1_envs/p1_grid_4x4.jpg). 시각 조건 3·4는 물체·로봇 질감 덧씌움, 5·6은 화면 전체 노이즈가 보이고, 의미·실행 조건은 각자 다른 물체·접시·배치가 보인다.

## 해석과 다음 단계

1. **처리량의 바닥이 정해졌다.** CPU 시뮬레이션은 제어 스텝 하나에 물리 서브스텝 100개(500/5 Hz)를 돌려 약 14 FPS다. 렌더링(약 200 FPS)보다 물리가 병목이다. 환경만으로 에피소드(80스텝)당 약 5.7초이고, P3(에피소드 2,048개)는 환경 부분만 약 3.2시간이다. 여기에 OpenVLA-7B 추론 시간이 더해진다. 실제 값은 P2에서 잰다.
2. **Windows 경로에서 원 Linux 경로와 다른 점**은 위 표의 세 설정(`sim_backend`, `render_backend`, `num_envs`)과, CPU 시뮬레이션 때문에 생긴 기구학 계산 경로(pinocchio, 원 GPU 경로는 pytorch_kinematics)다. 같은 행동에 대한 제어 결과가 수치적으로 조금 다를 수 있으므로, P2 재현 대조로 그 영향을 함께 본다.
3. **다음 (P1 나머지):** `openvla` 패키지와 transformers 4.40.1 설치, flash-attn 없이 OpenVLA 로드(표준 attention), 생성 설정(`top_k`, 온도) 출력·기록(P1 절차 4). 그다음 P2: 공개 체크포인트 3종 × 핵심 과제 4개 평가. 원 평가 코드(`train_ms3_ppo.py --only_render`)는 환경 64개와 GPU 시뮬레이션을 전제하므로, Windows용 환경 1개 평가 루프를 따로 만들되 행동 처리·정규화·성공 판정은 원 코드를 그대로 가져다 쓴다.
