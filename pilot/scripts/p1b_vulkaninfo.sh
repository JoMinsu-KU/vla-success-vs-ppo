#!/usr/bin/env bash
# P1-b: rlvla_env에 Vulkan 로더·도구를 넣고 vulkaninfo로 장치를 확인한다.
# Mesa 드라이버는 임시 환경(p1b_mesa_probe)에서 Vulkan ICD 포함 여부만 본다.
set -u
source ~/miniforge3/etc/profile.d/conda.sh
conda install -y -q -n rlvla_env -c conda-forge libvulkan-loader vulkan-tools > /tmp/p1b_vk_install.log 2>&1
tail -2 /tmp/p1b_vk_install.log
conda activate rlvla_env
echo "== vulkaninfo --summary (rlvla_env, 시스템 ICD 없음)"
vulkaninfo --summary 2>&1 | head -40
conda deactivate
echo "== Mesa 임시 환경"
conda create -y -q -n p1b_mesa_probe -c conda-forge mesalib > /tmp/p1b_mesa.log 2>&1
tail -1 /tmp/p1b_mesa.log
P=~/miniforge3/envs/p1b_mesa_probe
echo "-- Vulkan 드라이버 .so"; find "$P" -name 'libvulkan_*.so*' 2>/dev/null
echo "-- ICD json"; find "$P" -path '*vulkan/icd.d*' -name '*.json' 2>/dev/null
