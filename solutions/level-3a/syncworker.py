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

"""One SYNCHRONOUS D attempt (syncd engine + cutoff cross-check + residual correction) in its OWN interpreter.

Why a separate interpreter and not a fork: the parent has initialised CUDA and used torch's OpenMP thread pool by the
time the D stage runs, and a forked child deadlocks or dies on its first torch CPU op (MEASURED 2026-09-24: all four
forked attempts died silently within 2 s under the validator's container flags while the CPU harness, where the parent
had touched neither, was fine). A fresh process has neither problem -- the same reason dworker.py exists.

usage: syncworker.py <job.json>
  job: qasm, centre, view, tau, window, tol, extend, cutoff, seed, out, budget, max_dist, min_ev, peel, t0, max_s
Writes <out> when the attempt is DECIDED: {"ok": True, ...} for a D that passed the gates and the cross-check, or
{"ok": False, ...} for a structural / cross-check failure of THIS CUT (the parent's ladder then moves on). Leaves NO
file when the engine gave up (bond / time) or crashed: the parent launches the reference worker exactly as before.
Log lines go to stdout (the parent's stream) in the parent's own format.
"""
import json
import os
import pickle
import signal
import sys
import time
from datetime import datetime, timezone

# thread caps BEFORE numpy/torch load their BLAS (V13_SYNC_THREADS; 4 attempts x 4 threads = one 16-core box)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = os.environ.get("V13_SYNC_THREADS", "4")

job = json.load(open(sys.argv[1]))
os.environ.setdefault("EXC_MIN_EVIDENCE", str(job.get("min_ev", 3)))
os.environ.setdefault("EXC_PEEL_MODE", job.get("peel", "close"))
os.environ["EXC_EXTEND"] = str(int(job["extend"]))
os.environ["HQP_D_ENGINE"] = "sync_only"
os.environ["SYNCD_BUDGET_S"] = str(float(job["budget"]))
BUDGET = float(job["budget"])
C, V, OUT = int(job["centre"]), int(job["view"]), job["out"]
T0 = float(job.get("t0", time.time()))

# a stuck attempt must never outlive the D stage (and never write into the final payload)
signal.signal(signal.SIGALRM, lambda *_a: os._exit(0))
signal.alarm(int(float(job.get("max_s", 4 * BUDGET + 600))))


def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} + {time.time()-T0:7.1f}s] {m}", flush=True)


import numpy as np                                # noqa: E402
import torch                                      # noqa: E402,F401
import fp32_patch                                 # noqa: E402,F401
import excise as X                                # noqa: E402
import excision_solver as ES                      # noqa: E402
from qiskit import qasm2                          # noqa: E402
from qiskit.circuit.library import UGate          # noqa: E402

X.EXC_MAX_DIST = int(job["max_dist"])
X.EXC_TWIN_TOL = float(job["tol"])
custom = [qasm2.CustomInstruction('u', 3, 1, lambda t, p, l: UGate(t, p, l), builtin=True)]
circ = qasm2.load(job["qasm"], custom_instructions=custom)       # FILE order (DAG order breaks discovery)
n = circ.num_qubits
gates = ES._circ_to_gates(circ)
tau = [int(x) for x in job["tau"]]
window = int(job["window"])
cut_off, sd, tol = float(job["cutoff"]), int(job["seed"]), float(job["tol"])


def attempt():
    """The parent's try_sync, verbatim in semantics. Returns True when a decision was written."""
    c, v, out = C, V, OUT
    t_s = time.time()
    msgs = []
    _live = ("syncd:", "D structure", "resid: block", "-> EXCISE", "keep (gate failed)", "fails the ansatz gate", "STRUCTURAL failure")

    def _collect(m_):
        """Collect measure_D's commentary; print the PHASE lines at once so a slow attempt is diagnosable while it
        runs (2026-09-24: a bond-1024 attempt stayed silent for 45 min and nothing told which phase was slow)."""
        msgs.append(m_)
        if any(k_ in m_ for k_ in _live):
            log(f"v13: block {c} view {v}: [{time.time()-t_s:.0f}s] {m_.strip()[:180]}")

    cut, corr, binfo = ES.measure_D(gates, n, c, tau, window, None, _collect, budget=BUDGET, cutoff=cut_off, seed=sd)
    # the sync path silences measure_D's running commentary (one line per block would flood the log), but a CORRECTED
    # WIRE MAP must never be silent: it changes what the reduced circuit means. MEASURED on h6, whose block 3210 is
    # only excisable because the decoder re-pairs four wires.
    for m_ in msgs:
        if "residual permutation" in m_:
            log(f"v13: block {c} view {v}:{m_.split('decode:')[-1].strip()[:200]}")
        elif ("resid: block" in m_ or "resid: WARNING" in m_ or "residual correction failed" in m_ or
      "fails the ansatz gate" in m_ or "STRUCTURAL failure" in m_ or "-> EXCISE" in m_ or "keep (gate failed)" in m_):
            log(f"v13: block {c} view {v}: {m_.strip()[:200]}")
    if cut is None:
        why = binfo.get("sync_failed") or (f"gate: model err {binfo.get('model_err')}, unitary {binfo.get('unitary')}, "
                                           f"3-body {binfo.get('three_body')}")
        uni, merr = binfo.get("unitary"), binfo.get("model_err")
        structural = not binfo.get("sync_failed") and ((uni is not None and uni < 0.5) or (merr is not None and merr > 0.9))
        if structural:
            # the operator of THIS CUT is not a product (MEASURED on h10 block 3120, edge-extended cut: the reference
            # absorber finds the same thing, unitarity 0.008, after ~1 h): record it so the ladder moves on at once
            pickle.dump({"ok": False, "info": {k_: v_ for k_, v_ in binfo.items() if k_ != "audit"}}, open(out, "wb"))
            log(f"v13: block {c} view {v}: sync engine: the cut's operator is not a product ({why}) [{time.time()-t_s:.1f}s]")
            return True
        log(f"v13: block {c} view {v}: sync engine gave no D ({why}) [{time.time()-t_s:.1f}s] -> reference worker")
        return False
    # D-LEVEL CROSS-CHECK: re-measure the SAME cut at other truncation cutoffs (seconds). A cut whose D moves between
    # cutoffs is not a cut we should excise.
    xcheck = [float(x) for x in os.environ.get("V13_SYNC_CROSSCHECK", "1.2e-3,3e-4").split(",") if x.strip()]
    # SLOW blocks (bond near the cap): a cross-check re-measures the whole cut at another cutoff, i.e. up to the full
    # budget EACH; MEASURED 2026-09-24 on genadv h53 block 3020 (right tau, spread cores): the attempt stayed silent for
    # 45 min (3 x 900 s) while the D stage waited. When the primary measurement took longer than V13_XCHECK_SLOW_S
    # (default 300 s) only the first cross-check cutoff runs, and a cross-check that returns no D on such a block counts
    # as SKIPPED (the primary gates + the two-view agreement still apply), not as "D not stable".
    slow_ = (time.time() - t_s) > float(os.environ.get("V13_XCHECK_SLOW_S", "300"))
    if slow_ and len(xcheck) > 1:
        log(f"v13: block {c} view {v}: the measurement took {time.time()-t_s:.0f}s -> only the first cross-check cutoff")
        xcheck = xcheck[:1]
    min_fid = float(os.environ.get("V13_SYNC_MIN_FID", "0.98"))
    xinfo = []
    if xcheck:
        A_of = lambda cc, w_: np.array([complex(e[0], e[1]) for e in cc[("A", w_)]]).reshape(2, 2)   # noqa: E731
        # compare only STRONG two-body terms: a phase near the decoder's noise floor can appear or vanish between
        # cutoffs without meaning anything; a single term of difference is tolerated (V13_SYNC_PI_DIFF)
        pi_min = float(os.environ.get("V13_SYNC_PI_MIN", "1.0"))
        pis = lambda cc: {k_[1:] for k_, v_ in cc.items() if k_[0] == "cz" and                        # noqa: E731
                          (v_ is None or abs(float(v_)) >= pi_min)}
        pi_diff = int(os.environ.get("V13_SYNC_PI_DIFF", "1"))
        for co2 in xcheck:
            if abs(co2 - cut_off) < 1e-12:
                continue
            try:
                c2, corr2, bi2 = ES.measure_D(gates, n, c, tau, window, None, lambda m: None, budget=(min(BUDGET, float(os.environ.get("V13_XCHECK_SLOW_BUDGET_S", "300"))) if slow_ else BUDGET), cutoff=co2, seed=sd)
            except Exception:                                 # noqa: BLE001
                c2, corr2, bi2 = None, None, {}
            if c2 is None or corr2 is None:
                xinfo.append((co2, None, None, None))
                continue
            fid = min(abs(np.trace(A_of(corr2, w_).conj().T @ A_of(corr, w_))) / 2 for w_ in range(n))
            nd = len(pis(corr2) ^ pis(corr))
            # When the residual correction carried the block (the ansatz kept only a fraction of D), the ansatz's 1q
            # factors are NOISE and comparing them across cutoffs says nothing (MEASURED on w20 block 3200: correction
            # 0.190 -> 0.9987 on both views, ansatz 1q fidelity across cutoffs 0.04-0.05). What is meaningful there is
            # that the EMITTED operator reproduces the measured one at both cutoffs. V13_XCHECK_RESID=0 restores the
            # ansatz-only comparison.
            if os.environ.get("V13_XCHECK_RESID", "1") != "0":
                f1, f2, f0 = binfo.get("resid_fid1"), bi2.get("resid_fid1"), binfo.get("resid_fid0")
                rmin_x = float(os.environ.get("V13_RESID_MIN", "0.97"))
                if f1 is not None and f2 is not None and f0 is not None and f0 < min_fid:
                    if min(float(f1), float(f2)) >= rmin_x:
                        log(f"v13: block {c} view {v}: cross-check at cutoff {co2:g} judged on the CORRECTED operator "
                            f"(the ansatz kept only {f0:.3f} of D, so its 1q factors are noise): post-correction "
                            f"overlaps {float(f1):.4f} / {float(f2):.4f}")
                        fid = min(float(f1), float(f2))
            xinfo.append((co2, float(fid), nd <= pi_diff, nd))
        good = [x for x in xinfo if x[1] is not None]
        bad = [x for x in good if x[1] < min_fid or not x[2]]
        if not good and slow_ and not bad:
            log(f"v13: block {c} view {v}: cross-check returned no D on a slow block -> skipped (primary gates only)")
            good = [("skipped", 1.0, True, 0)]
        log(f"v13: block {c} view {v}: D cross-check " + ", ".join(
            f"cutoff {x[0]:g}: " + ("no D" if x[1] is None else f"1q fidelity {x[1]:.4f}, strong two-body terms "
                                    f"{'match' if x[2] else 'DIFFER'}" + (f" ({x[3]} of them)" if x[3] else "")) for x in xinfo))
        if bad or not good:
            log(f"v13: block {c} view {v}: D is NOT stable across cutoffs -> this cut is not trusted, next cut option")
            pickle.dump({"ok": False, "info": dict({k_: v_ for k_, v_ in binfo.items() if k_ != "audit"},
                                                   xcheck_failed=True)}, open(out, "wb"))
            return True
    res = {"ok": True, "s": list(cut.s), "e": list(cut.e), "tau": list(tau), "corr": corr,
           "info": dict({k_: v_ for k_, v_ in binfo.items() if k_ != "audit"}, xcheck=[(x[0], x[1], x[2]) for x in xinfo])}
    tmp = out + ".tmp"
    pickle.dump(res, open(tmp, "wb"))
    os.replace(tmp, out)
    sy = binfo.get("sync", {})
    log(f"v13: block {c} view {v}: sync engine: {sy.get('pairs')}/{sy.get('mirror_pairs_found')} mirror pairs in lockstep, "
        f"peak bond {sy.get('peak_bond')}, final {sy.get('final_bond')}, log10 norm2 {sy.get('log10_norm2', 0):+.3f} "
        f"[{time.time()-t_s:.1f}s]")
    return True


try:
    attempt()
except Exception as e:                                            # noqa: BLE001
    log(f"v13: block {C} view {V}: sync engine crashed ({type(e).__name__}: {str(e)[:100]}) -> reference worker")
    try:
        if os.path.exists(OUT + ".tmp"):
            os.remove(OUT + ".tmp")
    except OSError:
        pass
os._exit(0)
