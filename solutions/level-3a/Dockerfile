# Copyright (C) 2026 qBitTensor Labs.
# Original author: Charlie (Enigma / Hardening Quantum Proof competition).
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

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    TMPDIR=/tmp \
    NUMBA_CACHE_DIR=/tmp \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp

# python:3.12-slim does not set these (unlike the nvidia/cuda base). The validator
# runs bare `--gpus all`, so it relies on the image to request the CUDA "compute"
# capability. Without this the container toolkit may inject only "utility"
# (nvidia-smi, no CUDA) -> torch silently falls back to CPU. Request it explicitly.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the exact build the solver was validated on. Unpinned, the validator builds fresh
# and silently gets whatever is newest on the index -- a different torch than the one tested.
RUN pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0

RUN pip install \
    "numpy==2.4.4" \
    "qiskit==2.4.0" \
    "qiskit-quimb==0.0.10" \
    "quimb==1.13.0" \
    "cotengra==0.7.5" \
    "autoray==0.8.10" \
    "scipy==1.17.1" \
    "qiskit-qasm3-import"

# Optional quality deps (env-gated features; default-off path never imports them, so a failure
# here can't break the validated baseline). #11 ZX (pyzx) + cotengra path optimizers (kahypar/optuna).
RUN pip install pyzx kahypar optuna cmaes || true

RUN useradd -r -s /sbin/nologin -d /app -U miner

ENV WALL_TIME=14400
# Submission config: the validator runs with no custom env, so bake the WORKING unswap config in.
# #5 anti-livelock (or the bare unswap LIVELOCKS on hard circuits), #6 spectral ordering,
# early-abandon (drop a thrashing ordering -> try the next), #13 faithful extraction, direct
# (faithful + memory-safe + ~10x faster) absorption, and best-effort (always report the top
# candidate -- no penalty for wrong answers, unlimited resubmissions). Tune HQP_US_MAXBOND up
# on a bigger GPU (e.g. 2048-4096 on a 48-96GB PRO 6000; default 1024 is safe on 24GB).
ENV HQP_US_ANTILIVELOCK=1 \
    HQP_US_SMART_ORDER=1 \
    HQP_US_EARLY_ABANDON=1 \
    HQP_FAITHFUL=1 \
    HQP_COMPRESS_METHOD=direct \
    HQP_BEST_EFFORT=1

WORKDIR /app
COPY enigma_challenges /app/enigma_challenges/
COPY *.py /app/

USER miner
ENTRYPOINT ["python", "/app/hardening_quantum_proof.py"]
