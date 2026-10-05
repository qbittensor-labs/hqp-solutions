# Copyright (C) 2026 qBitTensor Labs.
# Original author: Alexey (Enigma / Hardening Quantum Proof competition).
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

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    TMPDIR=/tmp \
    NUMBA_CACHE_DIR=/tmp \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp \
    CUPY_CACHE_DIR=/tmp \
    CUDA_CACHE_PATH=/tmp \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN umask 022 && curl -LsSf https://astral.sh/uv/0.11.26/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.12.13 \
    && uv venv /opt/venv --python 3.12.13

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN umask 022 && uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0" \
    && SP=/opt/venv/lib/python3.12/site-packages \
    && rm -rf $SP/triton $SP/torch/include \
    && rm -f $SP/nvidia/cudnn/lib/libcudnn_engines_precompiled.so* $SP/nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so* \
              $SP/nvidia/cudnn/lib/libcudnn_adv.so* $SP/nvidia/cudnn/lib/libcudnn_cnn.so* $SP/nvidia/cudnn/lib/libcudnn_ops.so* \
    && find $SP/nvidia -type d -name include -prune -exec rm -rf {} + \
    && find $SP/nvidia $SP/torch -name "*.a" -delete

COPY requirements.txt /opt/requirements.txt
RUN umask 022 && uv pip install -r /opt/requirements.txt

RUN useradd -r -s /usr/sbin/nologin -d /app -U miner

ENV WALL_TIME=14400 \
    HQP_WALL_BUDGET=13400 \
    HQP_RESERVE=500 \
    HQP_WORKDIR=/tmp/hqp \
    HQP_DEVICE=cuda:0 \
    HQP_DTYPE=complex64 \
    PYTHONPATH=/app \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app
COPY enigma_challenges /app/enigma_challenges/
COPY enigma_peaked /app/enigma_peaked/
COPY src /app/src/
COPY scripts /app/scripts/
COPY schemas /app/schemas/
COPY evidence /app/evidence/
COPY *.py /app/

USER miner
ENTRYPOINT ["python", "/app/hardening_quantum_proof.py"]
