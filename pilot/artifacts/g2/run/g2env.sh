# G2 실행 환경: conda rlvla_env + 홈 폴더 NVIDIA 그래픽 라이브러리
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate rlvla_env
export PATH=$HOME/anaconda3/envs/rlvla_env/bin:$PATH
source $HOME/vla_g2/nvidia_gfx/env.sh
export TOKENIZERS_PARALLELISM=false
# 컨테이너 CPU 할당이 4개(cgroup cpu.max 400000/100000)인데 nproc은 128을 보고한다.
# 라이브러리 기본 스레드 수(torch 64/128)가 할당을 크게 넘어 cgroup 스로틀이 걸리므로 4로 맞춘다.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
