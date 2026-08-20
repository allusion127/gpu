# Welcome to the HYUSENM Wiki

RASBERY (Reactor Analyzer for Statics, Burnup Evaluation and tRansient analYsis) is an advanced computational framework designed for efficient and accurate simulation of neutron diffusion problems in nuclear reactor physics. 
This project integrates modern numerical techniques, including the Source Expansion Nodal Method (SENM), Flux Expansion Nodal Method (FENM), and Coarse Mesh Finite Difference (CMFD), enhanced by the Biconjugate Gradient Stabilized (BICGStab) solver.


# Dependencies

sudo apt update
sudo apt full-upgrade -y

sudo apt install -y \
  build-essential \
  gcc-13 g++-13 \
  clang lld \
  cmake ninja-build pkg-config git \
  libhdf5-dev hdf5-tools \
  libomp-dev libgomp1 \
  python3 python3-venv python3-pip python3-tk \
  libfreetype6-dev libpng-dev libjpeg-dev zlib1g-dev \
  fontconfig \
  fonts-dejavu fonts-dejavu-core fonts-dejavu-extra \
  fonts-liberation fonts-freefont-ttf fonts-stix \
  fonts-noto-core fonts-noto-cjk

fc-cache -fv

cd ./Rasbery

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r Viewer/requirements.txt

## How to test
use ./test/Tests.txt

## CUDA CMFD backend

Configure with `-DRASBERY_ENABLE_CUDA=ON` and set `RASBERY_GPU=1` at
runtime. A requested but unavailable CUDA backend fails closed.

The CMFD BiCGSTAB vectors remain device-resident during the inner iteration.
Each iteration reads one 64-byte `DeviceSolveStatus`; the full flux is copied
at the CPU Wielandt/nodal observation boundary. At shutdown, the executable
prints one machine-readable `[RASBERY][CUDA][BACKEND_COUNTERS]` JSON record.

Tuning controls:

- `RASBERY_GPU_BLOCK_SIZE`: `64`, `128`, `192`, or `256` (default `256`)
- `RASBERY_GPU_RB_SWEEPS`: non-negative red/black block sweep count
