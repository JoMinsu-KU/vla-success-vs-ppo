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

**고정 복제본에서 실행한다 (2026-11-10부터).** 실행기는 작업마다 저장소의 `study/train.py` 등을 새로 실행하므로, 개발용 저장소에서 실행하면 `git pull`이나 코드 수정이 진행 중인 실험에 섞인다. 그래서 실행은 태그에 고정한 별도 복제본에서 하고, 개발은 `~/vla_g2/vla-success-vs-ppo`에서 한다.

```bash
git clone ~/vla_g2/vla-success-vs-ppo ~/vla_main_code && git -C ~/vla_main_code checkout gpre   # 처음 한 번
nohup setsid bash ~/vla_main_code/study/run_phase1.sh >> ~/vla_main/main_console.log 2>&1 < /dev/null &
```

- **1단계는 2026-11-10 12:04부터 `~/vla_main_code`(태그 `gpre`, 7b5d2dc)에서 돌고 있다.** 이 폴더에서는 `git pull`·`checkout`을 하지 않는다.
- tmux가 있으면 tmux 안에서 실행해도 된다 (G2 인스턴스에는 없어서 `nohup setsid`를 쓴다).

- **끊겼을 때:** 같은 명령을 다시 실행한다. 끝난 작업은 건너뛰고, 학습은 마지막 latest 체크포인트(5라운드마다)부터 이어 간다. 실패한 작업은 자동으로 2번까지 다시 시도한다.
- **진행 확인:** `bash ~/vla_g2/vla-success-vs-ppo/study/status.sh` (공개 정보만 보여 준다).
- **로그:** `~/vla_main/logs/<실행 ID>.log`. 오류가 나면 이 파일 끝부분을 보내 준다.

## 2b. 2단계 (1단계 G-mid 뒤)

2단계 = 핵심 갈래(A3⁰·A2⁺) 나머지 시드 5~n (n은 `public/gmid.json`) → 보조 갈래(A2±·A3ˢ) 튜닝 4후보씩(1단계와 같은 튜닝 시드·과제 표본·150라운드·선택 규칙) → 보조 갈래 E1 시드 1~3 (결정적 경로, 같은 시드의 핵심 갈래와 같은 과제 표본). 설정은 [phase2_config.json](phase2_config.json) (명세 3.3·3.5절 값 그대로). A1·E2·E3·E4-b는 이후에 추가한다.

1단계가 끝나면 (`PHASE 1 DONE`):

1. 2단계 코드가 들어간 커밋으로 새 고정 복제본을 만든다 (1단계 폴더는 그대로 둔다). 예: `git clone ~/vla_g2/vla-success-vs-ppo ~/vla_main_code2 && git -C ~/vla_main_code2 checkout <2단계 커밋>`.
2. **스모크 먼저 (약 2.5시간):** `bash ~/vla_main_code2/study/run_phase2.sh --smoke`. 1단계 스모크 폴더(`~/vla_main/smoke`)에서 축소 흐름을 돌리고, **동등성 검사**로 새 학습기가 A3⁰·A2⁺를 1단계 코드와 비트 단위로 같게 학습하는지 확인한다 (`public/equiv_check_*.json`의 `passed: true`). 통과하지 않으면 2단계를 시작하지 않는다 (핵심 갈래 시드 5~n이 시드 1~4와 다른 코드로 학습되면 안 된다).
3. 본 실행: `nohup setsid bash ~/vla_main_code2/study/run_phase2.sh >> ~/vla_main/main_console.log 2>&1 < /dev/null &`

1단계가 도는 동안에는 GPU 스모크를 돌리지 않는다 (H200 하나를 같이 쓰면 메모리 부족으로 본 실행이 죽을 수 있다). CPU 시험은 언제든 된다.

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
| `run_phase1.sh`, `run_phase2.sh` | 환경 설정 → preflight → runner (1단계 / 2단계) |
| `runner.py` | 계획 생성(1단계 `--phase 1`, 2단계 `--phase 2`)·순차 실행·재시작·재시도, 태그 확인 |
| `train.py` | 학습 1회 (A3⁰·A3ˢ PPO, A2⁺ U-prop/U-fix, A2± 가중 NLL, 우도 갈래 KL 되돌림, 결정적/기본 경로) |
| `evaluate.py` | ID 검증(512) 또는 16조건 × 192 평가 |
| `select_tuning.py` | 140·145·150라운드 ID 검증 평균으로 후보 선택 |
| `gmid.py` | 맹검 SD_d → n, 예상 검정력 |
| `preflight.py`, `status.sh` | 점검, 진행 상황 |
| `tests/test_study_cpu.py` | CPU 단위 시험 (`python study/tests/test_study_cpu.py`) |
