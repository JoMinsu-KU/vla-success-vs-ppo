#!/usr/bin/env bash
# G2 S0: H200 인스턴스 기본 정보와 Vulkan(렌더링) 준비 상태 확인. GPU 계산 없음.
# 사용: bash g2_00_sysinfo.sh > g2_s0_sysinfo.txt 2>&1
set -u
sec() { echo; echo "===== $1"; }

sec "OS / kernel";        cat /etc/os-release 2>/dev/null | head -4; uname -r
sec "CPU / RAM / disk";   nproc; free -g | head -2; df -h "$HOME" /tmp 2>/dev/null
sec "container?";         [ -f /.dockerenv ] && echo "docker: yes" || echo "docker: no"; echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
sec "nvidia-smi";         nvidia-smi
sec "GPU list";           nvidia-smi --query-gpu=index,name,driver_version,memory.total,mig.mode.current,display_mode,compute_mode --format=csv
sec "CUDA toolkit";       (nvcc --version 2>/dev/null | tail -2) || echo "nvcc not on PATH"
sec "conda";              (conda --version && conda env list) 2>/dev/null || echo "conda not on PATH"

sec "Vulkan ICD files (NVIDIA ICD가 있어야 GPU 렌더링 가능)"
for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d /usr/local/share/vulkan/icd.d; do
  [ -d "$d" ] && { echo "$d:"; ls -l "$d"; for f in "$d"/*.json; do [ -f "$f" ] && { echo "--- $f"; cat "$f"; }; done; }
done
sec "NVIDIA GL/Vulkan user-space libraries"
ldconfig -p 2>/dev/null | grep -E "libGLX_nvidia|libEGL_nvidia|libnvidia-gl|libnvidia-vulkan|libvulkan" || echo "none found by ldconfig"
sec "vulkaninfo (없으면 생략)"
if command -v vulkaninfo >/dev/null; then vulkaninfo --summary 2>&1 | head -60; else echo "vulkaninfo not installed (apt: vulkan-tools). S2의 SAPIEN 시험으로 대신 확인"; fi

sec "network: HuggingFace / GitHub"
curl -sS -o /dev/null -w "huggingface.co %{http_code} %{time_total}s\n" https://huggingface.co || true
curl -sS -o /dev/null -w "github.com %{http_code} %{time_total}s\n" https://github.com || true
echo; echo "S0 done"
