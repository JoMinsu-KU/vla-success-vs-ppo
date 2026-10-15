#!/usr/bin/env bash
# P1-b: sudo 없이 conda-forge에서 Vulkan 로더·도구·Mesa 드라이버를 찾는다
source ~/miniforge3/etc/profile.d/conda.sh
for p in vulkan-tools libvulkan-loader mesalib mesa-libgl-cos7-x86_64; do
  echo "== $p"
  conda search -q -c conda-forge "$p" 2>&1 | tail -2
done
