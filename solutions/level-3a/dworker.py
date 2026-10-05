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

"""Measure D for ONE block in its own process (own CUDA context) and leave the result as a pickle.

Why a separate process: a D-measurement is latency-bound (MEASURED: one Python core at 100 %, tensors
under 1M elements, GPU far from saturated), so the blocks of one circuit can be measured CONCURRENTLY on
the single validator GPU at close to no cost -- wall time of the D stage = the slowest block, not the sum.

usage: dworker.py <job.json>      job: qasm, centre, tau, window, tol, max_dist, cutoff, seed, budget, out
writes <out> = {"ok": bool, "s", "e", "tau", "corr", "info"}  (atomically)
"""
import json
import os
import pickle
import sys
import time

# Thread caps BEFORE numpy/torch load their BLAS: the validator gives the container 24 CPUs and sets no OMP_*
# variables, so every worker would start 24 BLAS threads on tensors of a few hundred elements and 4-8 workers
# would fight over the same cores. A worker is a latency-bound, essentially single-threaded job.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("HQP_DEV_CPU_THREADS", "4"))

job = json.load(open(sys.argv[1]))
os.environ.setdefault("EXC_MIN_EVIDENCE", str(job.get("min_ev", 3)))
os.environ.setdefault("EXC_PEEL_MODE", job.get("peel", "close"))
os.environ["EXC_EXTEND"] = str(int(job.get("extend", 1)))       # the parent decides which cut option this worker measures
t0 = time.time()
TAG = f"D@{job['centre']}"


def log(m):
    print(f"[{TAG} +{time.time()-t0:7.1f}s] {m}", flush=True)


import torch                                      # noqa: E402
import fp32_patch                                 # noqa: E402,F401
import excise as X                                # noqa: E402
import excision_solver as ES                      # noqa: E402
import ref_align                                  # noqa: E402,F401  (reference-path residual correction: fail LOUDLY if missing)
from qiskit import qasm2                          # noqa: E402
from qiskit.circuit.library import UGate          # noqa: E402

X.EXC_MAX_DIST = int(job["max_dist"])
X.EXC_TWIN_TOL = float(job["tol"])
custom = [qasm2.CustomInstruction('u', 3, 1, lambda t, p, l: UGate(t, p, l), builtin=True)]
circ = qasm2.load(job["qasm"], custom_instructions=custom)       # FILE order (DAG order breaks discovery)
n = circ.num_qubits
gates = ES._circ_to_gates(circ)


def _tb_cuda(x):
    return torch.tensor(x, dtype=torch.complex64, device="cuda")


tb = _tb_cuda
if os.environ.get("HQP_DEV_ADAPT", "0") == "1":
    _u = None
    for _name in ("unswap_ref", "unswap"):                        # production: unswap_ref is the reference absorber
        try:
            _m = __import__(_name)
            if hasattr(_m, "AdaptiveBackend"):
                _u = _m
                break
        except Exception:                                         # noqa: BLE001
            pass
    if _u is None:
        raise RuntimeError("HQP_DEV_ADAPT=1 but no absorber module provides AdaptiveBackend")
    tb = _u.AdaptiveBackend(dtype=torch.complex64)
    torch.set_num_threads(int(os.environ.get("HQP_DEV_CPU_THREADS", "4")))
    log(f"device-adaptive absorber: cpu while bond <= {tb.cpu_bond}, gpu from {tb.gpu_bond}")


res = {"ok": False, "info": {}}
try:
    cut, corr, info = ES.measure_D(gates, n, int(job["centre"]), list(job["tau"]), int(job["window"]), tb, log,
                                   budget=float(job["budget"]), cutoff=float(job["cutoff"]), seed=int(job["seed"]))
    res["info"] = {k: v for k, v in info.items() if k != "audit"}
    res["audit"] = info.get("audit")
    if cut is not None:
        res.update({"ok": True, "s": list(cut.s), "e": list(cut.e), "tau": list(job["tau"]), "corr": corr})
except Exception as e:                                            # noqa: BLE001
    import traceback
    traceback.print_exc()
    res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
tmp = job["out"] + ".tmp"
pickle.dump(res, open(tmp, "wb"))
os.replace(tmp, job["out"])
log(f"done ok={res['ok']} model_err={res['info'].get('model_err')} in {time.time()-t0:.0f}s")
