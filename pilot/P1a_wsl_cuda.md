# P1-a WSL CUDA 인식 시험 기록

- 일시: 2026-10-15 (KST)
- 근거: 연구 명세 v0.9, 6.0절 "WSL 시험 (P1 앞부분, 1일로 제한)"
- 판정: **통과** — WSL2 Ubuntu에서 RTX 4090과 PyTorch CUDA가 정상 인식됨. 다음은 P1-b(렌더링).

## 환경

| 항목 | 값 |
|---|---|
| WSL | 2.3.26.0, 커널 5.15.167.4-microsoft-standard-WSL2, WSLg 1.0.65 |
| 배포판 | Ubuntu 22.04.5 LTS (`wsl -d Ubuntu`로 지정. **기본 배포판은 `docker-desktop`**이므로 이름을 반드시 붙인다) |
| GPU | NVIDIA GeForce RTX 4090, compute capability 8.9, 24,564 MiB |
| 드라이버 | Windows 591.86 / WSL nvidia-smi 590.57, CUDA Version 13.1 |
| WSL 메모리 | 31 GiB (호스트 64 GB의 절반, WSL 기본 상한) |
| 디스크 | WSL `/` 782 GB 여유, `/mnt/d` 5.8 TB 여유 |
| conda | 기존 Miniforge `~/miniforge3` (conda 26.1.1, 채널 conda-forge). 새로 설치하지 않음 |

## 만든 환경: `rlvla_env`

| 패키지 | 버전 | 비고 |
|---|---|---|
| Python | 3.10.22 | RL4VLA README 기준 3.10 |
| torch | 2.2.0+cu121 | README 기준 PyTorch 2.2.0 + CUDA 12.1 |
| torchvision | 0.17.0+cu121 | torch 2.2.0 짝 |
| cuDNN | 8.9.2 (8902) | |
| numpy | 1.26.4 | pip가 2.2.6을 받았으나 torch 2.2.0은 numpy 1.x 빌드라 `numpy<2`로 고정 |

flash-attn 2.7.4, transformers 4.40.1, ManiSkill·SimplerEnv는 아직 설치하지 않았다 (P1 본 설치에서 README대로).

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -y -n rlvla_env python=3.10 pip
conda activate rlvla_env
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
```

## 시험과 출력

```bash
nvidia-smi
```

```
NVIDIA-SMI 590.57   Driver Version: 591.86   CUDA Version: 13.1
GPU 0: NVIDIA GeForce RTX 4090   3838MiB / 24564MiB 사용 중 (P8, 21W)
```

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# + 메모리 조회, bf16 8192×8192 행렬곱 20회
```

```
torch 2.2.0+cu121 cuda build 12.1 cudnn 8902
available True count 1
device NVIDIA GeForce RTX 4090 cap (8, 9)
mem free/total GB 22.45 23.99
bf16 matmul TFLOPS 118.4
bf16 supported True
```

## 파일럿에 영향을 주는 관찰

1. **WSL 메모리가 31 GiB다.** 문서의 "RAM 64GB"는 호스트 기준이다. 16k 시연 RLDS 변환이나 7B 모델 로드(CPU 쪽 bf16 가중치 약 15 GB)에서 부족할 수 있다. 필요하면 Windows의 `%UserProfile%\.wslconfig`에 `[wsl2] memory=48GB`처럼 상한을 올린다 (WSL 재시작 필요, docker-desktop 배포판에도 영향). 바꿀지는 연구자가 정한다.
2. **GPU 메모리 약 3.8 GB를 Windows 화면 출력이 이미 쓰고 있다.** 실제 여유는 약 22.5 GB다. P4 모델 로드 시험과 P5의 `buffer_inferbatch`는 이 값을 기준으로 잡는다.
3. 기본 WSL 배포판이 `docker-desktop`이다. 스크립트와 명령은 항상 `wsl -d Ubuntu`로 실행한다.
