# P1-b′ Windows 네이티브 렌더링 시험 기록

- 일시: 2026-10-15 (KST)
- 근거: 연구 명세 v0.9, 6.0절 "또 다른 대안: Windows 네이티브 Python". WSL 시험(P1-b)이 CPU 렌더러(lavapipe)로만 통과해 추가로 시험했다.
- 판정: **GPU 렌더링 통과 (조건부)**. Windows 네이티브에서 ManiSkill 카메라 프레임이 **RTX 4090 Vulkan으로** 렌더링된다. 단, Vulkan–CUDA 직결 경로(`render_backend="gpu"`)는 충돌하므로 `render_backend="cpu"`(4090으로 렌더링하고 결과만 CPU 메모리로 읽음)를 쓴다. GPU 병렬 시뮬레이션은 Windows에서 지원되지 않으므로 환경은 1개다.

## 요약

| 경로 | 렌더링 장치 | 결과 | 640×480 속도 (PickCube, 환경 1개) |
|---|---|---|---|
| WSL + lavapipe (P1-b) | llvmpipe (CPU) | 통과 | 약 46 FPS |
| Windows, `render_backend="gpu"` (`sapien_cuda`) | RTX 4090 | **충돌**: `get_picture_cuda`에서 access violation (0xC0000005) | — |
| Windows, `render_backend="pci:01:00.0"` | — | ManiSkill이 `pci:` 문자열 파싱 실패 (`ValueError: too many values to unpack`) | — |
| **Windows, `render_backend="cpu"` (`sapien_cpu`)** | **RTX 4090** (nvidia-smi에 `python.exe`가 C+G로 표시, 사용률 상승) | **통과** | **약 198 FPS** (128×128은 약 232 FPS) |

`render_backend="cpu"`라는 이름과 달리, SAPIEN은 기본 Vulkan 장치(첫 번째 장치 = RTX 4090)로 렌더링하고, CUDA 직결 대신 CPU 메모리로 결과를 읽는다. Vulkan 장치 목록에는 RTX 4090(`cudaId 0`)과 Intel UHD Graphics 770(`cudaId -1`)이 있다. 렌더링 중 4090에 프로세스가 그래픽 컨텍스트로 잡힌 것으로 4090 사용을 확인했다.

## 만든 환경: `rlvla_win` (Windows, Anaconda)

| 패키지 | 버전 | 비고 |
|---|---|---|
| Python | 3.10.22 | **Anaconda 기본 채널의 서명된 Python**. conda-forge의 서명 없는 `python.exe`는 이 PC에서 실행이 차단됨("Access is denied"), 기존 Anaconda 환경의 서명된 Python은 실행됨. 보안 설정은 바꾸지 않았다 |
| torch / torchvision | 2.2.0+cu121 / 0.17.0+cu121 | `torch.cuda.is_available() = True` |
| numpy | 1.26.4 | |
| mani_skill / sapien | 3.0.1 / 3.0.3 | `pilot/scripts/constraints_torch220.txt` 제약으로 설치 |

```powershell
conda create -y -n rlvla_win python=3.10 pip
C:\Users\choiLee\anaconda3\envs\rlvla_win\python.exe -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
C:\Users\choiLee\anaconda3\envs\rlvla_win\python.exe -m pip install mani_skill "numpy<2" -c D:\VLA_ReinforceLearning\pilot\scripts\constraints_torch220.txt
```

## 시험과 출력

스크립트: [scripts/p1b_render_test.py](scripts/p1b_render_test.py) (`P1B_RENDER_BACKEND`, `P1B_RES` 환경 변수), [scripts/p1b_which_device.py](scripts/p1b_which_device.py)

`render_backend="gpu"`:

```
Windows fatal exception: access violation
  File "...\mani_skill\utils\structs\render_camera.py", line 173 in <listcomp>   # get_picture_cuda
  File "...\mani_skill\sensors\camera.py", line 222 in get_obs
  ...
exit=-1073741819
```

`render_backend="cpu"`:

```
[render_device] cpu backend sapien_cpu
[make+reset] 0.74s
[frame] cam=base_camera shape=(128, 128, 3) dtype=uint8 mean=123.51 std=48.06 nonzero=1.000
[steps] 50 steps with rgb obs: 0.22s -> 231.7 FPS
render ok
```

640×480, 1,500스텝 동안 nvidia-smi:

```
GPU 사용률 4% → 17%, 메모리 약 4.17 GB (화면 출력분 포함)
|    0   N/A  N/A   61060   C+G   ...da3\envs\rlvla_win\python.exe   N/A |
[steps 640x480] 1500 steps: 7.57s -> 198.2 FPS
```

저장 프레임: [artifacts/p1b_win/W_cpu_base_camera.png](artifacts/p1b_win/W_cpu_base_camera.png) (로봇 팔, 나무 테이블, 빨간 큐브 정상).

## 해석

1. **4090에서 ManiSkill을 돌릴 경로는 Windows 네이티브가 WSL 저하 모드보다 낫다.** 렌더러가 원 논문과 같은 계열(NVIDIA Vulkan)이고, 단순 장면 기준 약 4.3배 빠르다. 드라이버와 OS는 원 논문(Linux)과 다르므로, P2 재현 대조로 그 영향을 점검한다는 6.0절 원칙은 그대로다.
2. **환경은 1개다.** Windows에는 GPU 병렬 시뮬레이션이 없다. RL4VLA의 평가·수집 코드는 환경 64개를 전제하므로 환경 수를 1로 두고 돌리도록 고쳐야 한다. 이 수정은 파일럿 한정이고 기록한다.
3. **병목은 렌더링이 아니라 OpenVLA-7B 추론일 것이다.** 198 FPS는 렌더링·물리만의 속도다. RL4VLA 장면(질감·Objaverse 물체)은 더 무겁고, 실제 처리량은 7B 추론(배치 1)이 정한다. P2 핵심 과제 1개로 실측한다.
4. **남은 작업 (P1 본 설치):** RL4VLA를 클론해 Windows에서 설치한다. flash-attn은 Windows 설치가 어려우므로 추론은 표준 attention(`attn_implementation="eager"` 또는 `sdpa`)으로 시도하고, 원 설정과 다르다는 점을 기록한다. 학습(P4 일부, A1 등)은 WSL `rlvla_env`에서 한다.

## 6.0절 경로표에 대한 제안

| 작업 | 위치 |
|---|---|
| ManiSkill이 필요한 파일럿 (P2 재현, P3 공급 측정, P5 미니 실행의 수집) | **Windows 네이티브 `rlvla_win`** (GPU Vulkan 렌더링, 환경 1개) |
| 렌더링이 필요 없는 시험·학습·분석 (P4, 통계 시뮬레이션) | WSL `rlvla_env` |
| 본실험 | Linux H200·H100 (변경 없음) |

P3를 Windows에서 재면, 렌더러 계열은 같지만 OS·드라이버·환경 수가 본실험과 다르다. 그래서 3.7절대로 잠정치로 두고 G2에서 다시 잰다.
