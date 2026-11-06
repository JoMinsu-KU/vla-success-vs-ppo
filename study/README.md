# 본실험 1단계 실행 안내 (H200 1개)

1단계 = 튜닝(A2⁺·A3⁰ 각 4후보) → 설정 선택 → E1 핵심 시드 1~4 (A3⁰·A2⁺, 390라운드) → 평가(16조건) → G-mid 맹검 분산 재추정.
근거: [G-PRE 초안](../docs/GPRE_draft.md) 7.1절, 설정 [gpre_config.json](gpre_config.json).

| 단계 | GPU시간 (추정) | 24시간 가동 시 |
|---|---|---|
| 스모크 시험 (`--smoke`) | 약 2 | 당일 |
| 튜닝 8후보 + ID 검증 24회 | 약 170 | 약 1주 |
| E1 시드 1~4 × 2갈래 + 평가 8회 | 약 560~650 | 약 3.5주 |
| G-mid | 0 | 자동 |

## 0. 준비 (G2 인스턴스를 그대로 쓰는 경우)

G2 때 만든 `~/vla_g2` (conda `rlvla_env`, `g2env.sh`, `RL4VLA`, `checkpoints/openvla-7b-rlvla-warmup`)를 그대로 쓴다.
새 인스턴스라면 먼저 `bash ~/vla_g2/vla-success-vs-ppo/pilot/g2/g2_install.sh ~/vla_g2`와 G2 S0~S2(체크포인트 내려받기)를 한다.

```bash
cd ~/vla_g2/vla-success-vs-ppo && git pull
```

작업 폴더는 `~/vla_main` (다른 곳이면 `export VLA_MAIN_WORK=...`). 여유 디스크 200 GiB 이상이 필요하다.

## 1. 스모크 시험 (태그 없이 지금 가능, 약 2시간)

```bash
tmux new -s main
bash ~/vla_g2/vla-success-vs-ppo/study/run_phase1.sh --smoke
```

축소 설정(후보 1개씩, 2라운드, 평가 1배치)으로 전체 흐름을 한 번 돈다. `~/vla_main/smoke`에 쓴다. 확인하는 것:
- preflight (GPU, 디스크, 체크포인트, 패키지, 저장소 상태)
- 튜닝 → 선택 → E1(결정적 경로) → 평가 → G-mid가 끝까지 도는지
- **재시작 검사:** 1라운드 뒤 멈췄다가 이어 간 실행이 끊김 없는 실행과 비트 단위로 같은지 (`public/resume_check.json`의 `passed: true`)

끝나면 아래만 보내 주면 된다 (공개 정보만 들어 있다):

```bash
cd ~/vla_main/smoke && tail -n 40 public/progress.log && cat public/resume_check.json && cat public/gmid.json
```

## 2. 본 실행 (G-PRE 고정 뒤)

본 실행은 태그 `gpre`가 있어야 시작된다 (연구자 결정 → 명세 v0.13 → `gpre_config.json` 확정 → 태그 push).
태그 이후 `study/gpre_config.json`이 바뀌면 실행기가 거부한다.

```bash
cd ~/vla_g2/vla-success-vs-ppo && git pull && git fetch --tags
tmux new -s main
bash ~/vla_g2/vla-success-vs-ppo/study/run_phase1.sh
```

- **끊겼을 때:** 같은 명령을 다시 실행한다. 끝난 작업은 건너뛰고, 학습은 마지막 latest 체크포인트(5라운드마다)부터 이어 간다. 실패한 작업은 자동으로 2번까지 다시 시도한다.
- **진행 확인:** `bash ~/vla_g2/vla-success-vs-ppo/study/status.sh` (공개 정보만 보여 준다).
- **로그:** `~/vla_main/logs/<실행 ID>.log`. 오류가 나면 이 파일 끝부분을 보내 준다.

## 3. 맹검 규칙 (꼭 지킬 것)

| 폴더 | 내용 | 열어도 되나 |
|---|---|---|
| `public/` | 진행 라운드 수, 상태, 실행 전체 GPU시간, 튜닝 선택 결과, G-mid 결과 | 예 |
| `logs/` | 프로그램 출력 (성공 수는 찍지 않는다) | 오류 확인용으로만 |
| `blinded/` | 성공·손실·KL·스텝 수·라운드 시간, 평가 에피소드 결과 | **E1 완료 전에는 열지 않는다** |
| `sealed/plan.json` | 실행 ID ↔ 갈래 대응 | **E1 완료 전에는 열지 않는다** |
| `runs/` | 체크포인트 | 지우지 않는다 (E2가 A3⁰ 라운드 19·20·149·150·299·300을 쓴다) |

- 성공 수와 함께 움직이는 값(라운드 시간, optimizer 스텝 수)도 맹검이다 (D-G2-4). 진행 중에 라운드별 시간을 따로 재서 공유하지 않는다.
- G-mid 결과(`public/gmid.json`)는 SD_d, n, 예상 검정력, 판정(`plan` 또는 `reduce`)만 담는다.

## 4. 파일

| 파일 | 역할 |
|---|---|
| `run_phase1.sh` | 환경 설정 → preflight → runner |
| `runner.py` | 계획 생성·순차 실행·재시작·재시도, 태그 확인 |
| `train.py` | 학습 1회 (A3⁰ PPO, A2⁺ U-prop/U-fix + KL 되돌림, 결정적/기본 경로) |
| `evaluate.py` | ID 검증(512) 또는 16조건 × 192 평가 |
| `select_tuning.py` | 140·145·150라운드 ID 검증 평균으로 후보 선택 |
| `gmid.py` | 맹검 SD_d → n, 예상 검정력 |
| `preflight.py`, `status.sh` | 점검, 진행 상황 |
| `tests/test_study_cpu.py` | CPU 단위 시험 (`python study/tests/test_study_cpu.py`) |
