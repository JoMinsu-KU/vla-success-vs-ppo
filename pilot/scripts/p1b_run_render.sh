#!/usr/bin/env bash
# P1-b: (A) 기본 경로(SAPIEN이 NVIDIA ICD 시도), (B) lavapipe CPU 렌더러로 카메라 프레임 시험
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rlvla_env
D=/mnt/d/VLA_ReinforceLearning/pilot
OUT=$D/artifacts/p1b
mkdir -p "$OUT"
if [ "${1:-}" != "--skip-a" ]; then
echo "===== (A) default (no explicit ICD)"
timeout 300 python $D/scripts/p1b_render_test.py A_default "$OUT" 2>&1 | grep -v UserWarning | tail -25
fi
echo "===== (B) lavapipe (Mesa CPU Vulkan)"
P1B_RENDER_BACKEND=cpu VK_ICD_FILENAMES=~/miniforge3/envs/p1b_mesa_probe/share/vulkan/icd.d/lvp_icd.x86_64.json \
  timeout 600 python $D/scripts/p1b_render_test.py B_lavapipe "$OUT" 2>&1 | grep -v UserWarning | tail -25
echo "===== (B2) lavapipe, 640x480"
P1B_RES=640x480 P1B_RENDER_BACKEND=cpu VK_ICD_FILENAMES=~/miniforge3/envs/p1b_mesa_probe/share/vulkan/icd.d/lvp_icd.x86_64.json \
  timeout 600 python $D/scripts/p1b_render_test.py B2_lavapipe_640x480 "$OUT" 2>&1 | grep -E '^\[|render ok|Error'
