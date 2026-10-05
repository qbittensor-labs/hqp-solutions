# Copyright (C) 2026 qBitTensor Labs.
# Original author: an anonymous competition participant (Enigma / Hardening Quantum Proof competition).
# IP in custom components assigned to qBitTensor Labs under the Enigma rules.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details. You should have received a copy of the license with this program;
# if not, see <https://www.gnu.org/licenses/>.

# Hardening Quantum Proof solver — GPU MPS peak finder.
# linux/amd64 + NVIDIA GPU (validator: RTX PRO 6000, 96 GB). CUDA 12 runtime libs
# come from the NVIDIA pip wheels below; only the host driver (via --gpus) is
# required at runtime.
FROM python:3.12-slim

# Pinned, self-contained deps. qiskit 1.4.x parses OpenQASM 2/3; cupy does the
# GPU linear algebra (SVD/QR/matmul) for the MPS engine. No network at runtime.
# cupy-cuda12x v13 does NOT bundle the CUDA libraries, so install the NVIDIA
# CUDA 12 pip wheels (nvrtc/cudart/cublas/cusolver + deps); cupy auto-discovers
# them under site-packages/nvidia/*/lib. Only the host driver (via --gpus) is
# needed at runtime.
RUN pip install --no-cache-dir \
    "numpy>=1.26,<3" \
    "qiskit==1.4.2" \
    "qiskit-qasm3-import>=0.5" \
    "cupy-cuda12x==13.*" \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-curand-cu12 \
    nvidia-cufft-cu12

# Register the pip-wheel CUDA libraries with the dynamic loader so cupy's lazy
# dlopen (e.g. libnvrtc.so.12 for JIT) resolves them at runtime under the
# read-only root filesystem. ldconfig builds the cache into the image layer.
RUN find /usr/local/lib/python3.12/site-packages/nvidia -type d -name lib \
        > /etc/ld.so.conf.d/nvidia-wheels.conf && ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib

# cupy JIT kernel cache must be writable; the validator mounts a read-only root
# filesystem with a writable /tmp (tmpfs). Point the cache there.
ENV CUPY_CACHE_DIR=/tmp/.cupy_kernel_cache \
    CUDA_CACHE_PATH=/tmp/.nv_cache \
    HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root runtime user required by the validator hardening.
RUN useradd -r -s /sbin/nologin -d /app -U miner

WORKDIR /app
# enigma_challenges is vendored into the build context by the validator/workbench.
COPY --chown=miner:miner enigma_challenges /app/enigma_challenges/
COPY --chown=miner:miner solve_hqp.py qload.py mps_gpu.py /app/

USER miner
ENTRYPOINT ["python", "/app/solve_hqp.py"]
