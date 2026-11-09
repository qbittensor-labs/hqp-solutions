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

# Level-4 HQP solver, CPU only (2026-10-13). Entry wrapper on Python 3.12 (stdlib only), solver on Python 3.10. Python build and package pins mirror the laptop environments that solved
# the difficulty-4 samples: peakpilot venv (undo / census / excision / route, torch CPU) and the p9solver venv (p9solve).
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
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN umask 022 && curl -LsSf https://astral.sh/uv/0.11.26/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.10.20 3.12.13 \
    && ln -s "$(uv python find 3.12.13)" /usr/local/bin/hqp-py312 \
    && uv venv /opt/pp --python 3.10.20 \
    && uv venv /opt/p9 --python 3.10.20

COPY requirements-pp.txt requirements-p9.txt /opt/
RUN umask 022 && uv pip install --python /opt/pp/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0" \
    && uv pip install --python /opt/pp/bin/python -r /opt/requirements-pp.txt \
    && uv pip install --python /opt/p9/bin/python -r /opt/requirements-p9.txt

COPY p9solver /opt/p9solver-src
RUN umask 022 && uv pip install --python /opt/p9/bin/python --no-deps /opt/p9solver-src \
    && /opt/p9/bin/p9solve --help >/dev/null \
    && /opt/pp/bin/python -c "import torch, numpy, scipy, qiskit, quimb; print('pp', torch.__version__, numpy.__version__, qiskit.__version__, quimb.__version__)"

RUN useradd -r -s /usr/sbin/nologin -d /app -U miner

ENV HQP4_PY=/opt/pp/bin/python \
    HQP4_P9SOLVE=/opt/p9/bin/p9solve \
    HQP_WORKDIR=/tmp/hqp4 \
    HQP_WATCHDOG_AT=13500 \
    HQP_DEADLINE=13200

WORKDIR /app
COPY enigma_challenges /app/enigma_challenges/
COPY hqp4 /app/hqp4/
COPY hardening_quantum_proof.py /app/

USER miner
# the wrapper needs Python >= 3.12 (enigma_challenges uses PEP 695 generics) and only the stdlib; the solver runs
# in /opt/pp (3.10) as subprocesses
ENTRYPOINT ["/usr/local/bin/hqp-py312", "/app/hardening_quantum_proof.py"]
