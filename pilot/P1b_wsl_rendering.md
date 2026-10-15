# P1-b WSL 렌더링 시험 기록

- 일시: 2026-10-15 (KST)
- 근거: 연구 명세 v0.9, 6.0절 "WSL 시험" 판정 흐름
- 판정: **CPU 렌더러로만 통과** → 6.0절 표의 **저하 모드** (CPU 시뮬레이션, 환경 1개). GPU Vulkan 경로는 없다. 이 경로의 결과는 실행 가능성 확인용이고 논문 근거로 쓰지 않는다. P3를 여기서 재면 잠정치이며 G2에서 다시 잰다 (3.7절).

## 요약

| 경로 | Vulkan 장치 | 결과 |
|---|---|---|
| (A) 기본 (NVIDIA ICD) | 없음 | **실패**: `vk::createInstanceUnique: ErrorIncompatibleDriver`. WSL2 NVIDIA 드라이버는 CUDA만 제공하고 Vulkan을 제공하지 않음 (R20 그대로) |
| (B) Mesa lavapipe (CPU Vulkan), `render_backend="cpu"` | `llvmpipe (LLVM 23.1.3, 256 bits)` | **통과**: PickCube-v1 카메라 프레임 정상. 128×128 약 50 FPS, 640×480 약 46 FPS (환경 1개) |
| Dozen (D3D12 경유 GPU Vulkan) | 시험 못 함 | conda-forge Mesa에 dzn 드라이버가 없음. 시스템 설치는 `sudo`(비밀번호)가 필요해 진행하지 않음 |

## 시스템 상태 (시험 전)

- Vulkan 로더·도구·ICD가 WSL Ubuntu에 **하나도 설치되어 있지 않았다** (`vulkan-tools` 없음, `/usr/share/vulkan/icd.d` 없음, Dozen `.so` 없음).
- `/usr/lib/wsl/lib`에는 `libcuda`, `libd3d12`, `libdxcore`, `libnvoptix` 등이 있고 NVIDIA Vulkan ICD는 없다.
- `sudo`는 비밀번호가 필요하다. 그래서 apt 설치 대신 conda-forge 패키지로만 진행했다.

## 설치한 것 (sudo 없이)

| 위치 | 패키지 | 버전 |
|---|---|---|
| `rlvla_env` | libvulkan-loader, vulkan-tools | conda-forge 1.4.357.0 |
| `rlvla_env` | mani_skill / sapien | 3.0.1 / 3.0.3 (pip, `torch==2.2.0`·`numpy<2` 제약으로 torch·numpy 변경 없음) |
| `p1b_mesa_probe` (임시 환경) | mesalib | conda-forge 26.2.x. Vulkan 드라이버는 `libvulkan_lvp.so`(lavapipe)뿐 |

`rlvla_env`의 ManiSkill은 PyPI판이다. P1 본 설치에서 RL4VLA README가 지정하는 ManiSkill로 바꿀 수 있다.

## 시험 명령과 출력

스크립트: [scripts/p1b_vulkaninfo.sh](scripts/p1b_vulkaninfo.sh), [scripts/p1b_install_maniskill.sh](scripts/p1b_install_maniskill.sh), [scripts/p1b_render_test.py](scripts/p1b_render_test.py), [scripts/p1b_run_render.sh](scripts/p1b_run_render.sh)

```bash
wsl -d Ubuntu -- bash /mnt/d/VLA_ReinforceLearning/pilot/scripts/p1b_vulkaninfo.sh
```

```
== vulkaninfo --summary (rlvla_env, 시스템 ICD 없음)
ERROR: [Loader Message] Code 0 : vkCreateInstance: Found no drivers!
Cannot create Vulkan instance.
== Mesa 임시 환경
-- Vulkan 드라이버 .so
/home/jms/miniforge3/envs/p1b_mesa_probe/lib/libvulkan_lvp.so
-- ICD json
/home/jms/miniforge3/envs/p1b_mesa_probe/share/vulkan/icd.d/lvp_icd.x86_64.json
```

```bash
wsl -d Ubuntu -- bash /mnt/d/VLA_ReinforceLearning/pilot/scripts/p1b_run_render.sh
```

(A) 기본 경로:

```
[svulkan2] [error] Your GPU driver does not support Vulkan. You may not use the renderer for rendering.
RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver
```

(B) lavapipe, 첫 시도 (`render_backend` 기본값 `gpu`):

```
[device] GPU: llvmpipe (LLVM 23.1.3, 256 bits)  Supported: 1  cudaId: -1
RuntimeError: Failed to find a supported physical device "cuda:0"
```

(B) lavapipe, `render_backend="cpu"`:

```
[resolution] default
[frame] cam=base_camera shape=(128, 128, 3) dtype=uint8 mean=111.81 std=51.62 nonzero=0.999
[steps] 50 steps with rgb obs: 1.01s -> 49.7 FPS
render ok
[resolution] 640x480
[frame] cam=base_camera shape=(480, 640, 3) dtype=uint8 mean=108.50 std=50.05 nonzero=0.999
[steps] 50 steps with rgb obs: 1.09s -> 45.9 FPS
render ok
```

저장 프레임: [artifacts/p1b/B_lavapipe_base_camera.png](artifacts/p1b/B_lavapipe_base_camera.png), [artifacts/p1b/B2_lavapipe_640x480_base_camera.png](artifacts/p1b/B2_lavapipe_640x480_base_camera.png). 로봇 팔, 나무 테이블, 빨간 큐브가 정상적으로 보인다.

## 해석과 다음 단계

1. **6.0절 흐름상 "CPU 렌더러로만 통과" 분기다.** P1-c(GPU 시뮬레이션 + 렌더링)는 이 흐름에서 해당 없음이다. 다만 PhysX GPU 시뮬레이션 자체는 CUDA만 쓰므로, "GPU 물리 + CPU 렌더"가 되는지는 별도 확인 대상으로 남긴다 (결정은 연구자).
2. **속도는 단순 장면(PickCube) 기준이다.** RL4VLA 과제는 테이블 질감·Objaverse 물체가 있어 더 느릴 것이다. 저하 모드에서 P3(에피소드 2,048개 × 80스텝 ≈ 16.4만 스텝)는 렌더링만 보면 약 1시간이지만, 실제 병목은 OpenVLA-7B 추론(환경 1개, 배치 1)이다. P2 핵심 과제 1개로 실측해 판단한다.
3. **렌더러가 원 논문(Linux NVIDIA Vulkan)과 다르다.** llvmpipe는 같은 장면도 픽셀이 다를 수 있다. 그래서 이 경로의 P2 재현 수치는 렌더러 차이의 영향을 함께 담는다.
4. **남은 선택지 (연구자 결정):**
   - (가) 저하 모드로 P2·P3를 4090에서 진행 (잠정치, G2 재측정).
   - (나) 연구자가 직접 `sudo`로 kisak-mesa PPA의 Dozen 드라이버를 설치해 GPU Vulkan을 한 번 시도. 6.0절의 1일 상한 안에서만. 지원 확장이 적다는 보고가 있어 SAPIEN이 받아들일지는 불확실하다.
   - (다) 경로 분리: ManiSkill 작업은 Linux 네이티브 GPU 기기, WSL은 학습·분석.
   - 어느 쪽이든 렌더링이 필요 없는 P4 단위 시험과 통계 시뮬레이션은 지금 진행할 수 있다.

## 정리할 것

- 임시 환경 `p1b_mesa_probe`는 lavapipe ICD를 제공하므로 저하 모드를 쓰는 동안 유지한다. 저하 모드를 쓰지 않기로 하면 `conda env remove -n p1b_mesa_probe`로 지운다.
