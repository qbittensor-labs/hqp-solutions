#!/usr/bin/env python3
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

"""Hardening Quantum Proof solver (v3) — self-certifying peaked-circuit peak finder.

Strategy (proven milestone-1 approach, hardened):
  - Small circuits (<= 30 qubits): exact statevector.
  - Larger: TEBD matrix-product-state on GPU (torch), escalating the bond dimension
    χ. Truncation denoises the random background and concentrates amplitude on the
    embedded peak; canonical BEAM-SEARCH ARGMAX (not sampling) recovers it.

Because exact verification (<s|U|0>) is infeasible (treewidth ~ qubit count), the
solver CERTIFIES its answer with three independent signals and only reports
"success" when confident — on a binary exact-match grader a confidently-wrong
answer is worthless, so we fail CLOSED:
  1. Convergence  — argmax stable across consecutive χ levels.
  2. Exactness    — if the reached bond never hits the χ cap, truncation discarded
                    nothing => provably exact for that circuit.
  3. Cross-check  — re-run at top χ under different qubit orderings (independent
                    truncation errors) and majority-vote.

A wall-clock budget guard keeps a best-so-far answer and never overruns the 4 h kill.
"""
import os
import sys

# CRITICAL output-protocol fix: redirect the process's stderr (fd 2) onto stdout (fd 1)
# at the OS level, BEFORE anything (logging, torch/quimb warnings) writes a byte.
# The validator captures results via `docker logs` = stdout+stderr merged by Docker's
# per-chunk timestamps. Our unswap logging emits ~700KB to stderr right up to the end;
# with two separate pipes that ordering is non-deterministic and a stderr chunk can land
# inside/after the trailing base64 payload -> base64 truncated -> extraction fails ->
# solution silently REJECTED even when the peak is correct. Collapsing to one stream makes
# the base64 deterministically the last bytes (the trivial-circuit case that always works).
try:
    sys.stdout.flush()
    os.dup2(sys.stdout.fileno(), sys.stderr.fileno())
except Exception:
    pass

import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# BLAS thread cap for THIS process, applied before numpy loads OpenBLAS (the validator's container sets no OMP_*, so
# OpenBLAS would start one thread per visible CPU). The main process runs the discovery, the cut options and the
# synchronous D engine (and its forked attempts inherit this pool); the GPU workers and the ensemble members set their
# own caps. MEASURED 2026-09-24 (w20 rehearsal): with every process at full thread count the same synchronous
# measurement took 18 s on a quiet box and 361 s next to two GPU workers. HQP_MAIN_CPU_THREADS=0 leaves the default.
_MAIN_THREADS = os.environ.get("HQP_MAIN_CPU_THREADS", "4")
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_thread_vars_set = []
if _MAIN_THREADS not in ("", "0"):
    for _v in _THREAD_VARS:
        if _v not in os.environ:
            os.environ[_v] = _MAIN_THREADS
            _thread_vars_set.append(_v)

import numpy as np

# children (D workers, ensemble members) choose their own caps: do not leak this process's setting into their env
for _v in _thread_vars_set:
    os.environ.pop(_v, None)

import wall_watchdog

from enigma_challenges.hardening_quantum_proof import Solution, load_solver_input
from enigma_challenges.solution_output import build_solution_zip, write_solution_output, SOLUTION_OUTPUT_SEPARATOR


# ---- hardened payload emission (HQP_EMIT_HARDEN, default 1) ------------------------------------------------------------
# The validator reads `docker logs` as res.stdout + res.stderr CONCATENATED, splits on the FIRST separator, strips, keeps
# the leading run of [A-Za-z0-9+/=\s] (clean_base64_payload) and decodes with validate=True. So ANY text that reaches the
# container output after our payload -- or any container-stderr text at all, whenever it was written -- and starts with a
# letter, digit or '/' (a Python traceback, multiprocessing's resource_tracker 'leaked semaphore' warning, a child's import
# warning ...) is absorbed into the payload -> 'Excess data after padding' -> InvalidOutputBase64 (reported to the miner as
# UploadFailure). MEASURED 2026-09-25: validation 7648d493 of tx 0x5c0c42 failed this way on a correct solve; replaying the
# validator's extraction shows appended 'Traceback ...' or '/usr/local/lib/...resource_tracker...' breaks the decode, and
# a '#' line right after the payload makes every such case decode. Defences, in order: (1) SIGKILL every descendant process
# (ensemble workers, D workers, sync workers, the resource tracker) so nothing else can write; (2) point this process's
# fds 1 and 2 at /dev/null so other threads' prints vanish; (3) write separator + payload + terminator in ONE os.write on a
# saved copy of the real stdout; (4) the terminator line (HQP_EMIT_TERMINATOR, default '#', empty = none) stops the
# validator's base64 cleaner before any junk that still follows.
def _descendants(root_pid):
    """PIDs of every descendant of root_pid (from /proc; empty if /proc is unreadable)."""
    kids = {}
    try:
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat", "rb") as f:
                    st = f.read().decode("ascii", "replace")
                ppid = int(st[st.rindex(")") + 2:].split()[1])
                kids.setdefault(ppid, []).append(int(d))
            except Exception:
                continue
    except Exception:
        return []
    out, todo = [], [root_pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            if c != root_pid and c not in out:
                out.append(c)
                todo.append(c)
    return out


def _emit_payload(zip_bytes):
    if os.environ.get("HQP_EMIT_HARDEN", "1").strip() in ("0", "false", "False"):
        write_solution_output(zip_bytes)
        return
    import base64 as _b64
    import signal as _sig
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    for pid in _descendants(os.getpid()):                  # (1) nobody else may write from here on
        try:
            os.kill(pid, _sig.SIGKILL)
        except Exception:
            pass
    term = os.environ.get("HQP_EMIT_TERMINATOR", "#")
    blob = SOLUTION_OUTPUT_SEPARATOR + _b64.b64encode(zip_bytes) + b"\n" + ((term.encode() + b"\n") if term else b"")
    try:
        real = os.dup(1)                                   # (2) + (3)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
    except Exception:
        write_solution_output(zip_bytes)                   # fall back to the plain protocol writer
        return
    view = memoryview(blob)
    while view:
        try:
            k = os.write(real, view)
        except InterruptedError:
            continue
        except Exception:
            break
        view = view[k:]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver"))

START = time.time()
WALL_BUDGET = float(os.environ.get("HQP_WALL_BUDGET", "13000"))   # of the 14400s hard kill
SAFETY = float(os.environ.get("HQP_SAFETY", "300"))
# Reserve inside the extraction stage itself, so an abandoned extraction still leaves time to
# finish the remaining orderings and write the result.
EXTRACT_SAFETY = float(os.environ.get("HQP_EXTRACT_SAFETY", "420"))
CHI_LADDER = [int(x) for x in os.environ.get(
    "HQP_CHI_LADDER",
    "64,128,256,384,512,768,1024,1536,2048,3072,4096,6144,8192").split(",")]
BEAM = int(os.environ.get("HQP_BEAM", "512"))
N_XCHECK = int(os.environ.get("HQP_XCHECK", "2"))
STABLE_CHI = int(os.environ.get("HQP_STABLE_CHI", "512"))   # require χ>=this before trusting stability
MIN_WEIGHT = float(os.environ.get("HQP_MIN_WEIGHT", "0"))   # optional floor on top1 MPS weight

# --- unswap engine (primary for large circuits; validated to crack difficulty 2) ---
ENGINE = os.environ.get("HQP_ENGINE", "unswap")            # "unswap" | "tebd"
US_CUTOFF = float(os.environ.get("HQP_US_CUTOFF", "0.002"))      # loose: fast unswap, no livelock
US_FINAL_CUTOFF = float(os.environ.get("HQP_US_FINAL_CUTOFF", "1e-5"))  # sharp: resolves the peak
# Extraction (mpo_to_mps) bond is capped SEPARATELY from absorption: the post-unswap core
# MPO is tiny (bond ~50 once the mirror cancels) but the leftover front layers re-entangle up
# to the cap, and quimb's mpo.apply forms the full product -> a bond-4096 extraction can
# transiently need ~72 GiB and OOM on d3. Keep absorption high (to escape the bond-128
# plateau) but bound extraction to what the 96 GB card holds; default 2048.
US_FINAL_MAXBOND = int(os.environ.get("HQP_US_FINAL_MAXBOND", "2048"))
def _auto_maxbond():
    """Auto-scale the MPS bond cap to the GPU's memory so ONE submission tunes itself to whatever
    hardware runs it (no per-host config). Memory ~ bond^2; the 24GB 4090 safely runs bond ~1024
    (it OOM'd absorbing d3 at ~1133), so scale by sqrt(mem/24): ~1024 on 24GB, ~1536 on a 48GB
    PRO 6000, ~2048 on a 96GB PRO 6000 Blackwell. Conservative on purpose (won't OOM the validator);
    override with HQP_US_MAXBOND to push higher once you've confirmed headroom via nvidia-smi."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 1024
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        bond = int(1024 * (mem_gb / 24.0) ** 0.5)
        return max(1024, min(4096, (bond // 256) * 256))   # clamp [1024,4096], round to 256
    except Exception:
        return 1024

US_MAXBOND = int(os.environ["HQP_US_MAXBOND"]) if "HQP_US_MAXBOND" in os.environ else _auto_maxbond()
US_EARLY_STOP = int(os.environ.get("HQP_US_EARLY_STOP", "30"))   # stop absorbing with <=N gates left (avoids tail livelock)
# IDEA #5 (A/B): unswap acceptance mode. "false"=strict (dimension-reducing swaps only, default/current);
# "none"=relaxed (accept dimension-preserving swaps to escape local minima, the paper's local-minimum escape);
# "true"=accept <=. Exposed for A/B testing whether relaxed acceptance breaks the 48q livelock.
_US_EQ = os.environ.get("HQP_US_EQUAL", "false").lower()
US_EQUAL = None if _US_EQ == "none" else (_US_EQ == "true")
US_SABRE_TRIALS = int(os.environ.get("HQP_SABRE_TRIALS", "10000"))
US_SEEDS = [int(s) for s in os.environ.get("HQP_US_SEEDS", "123,456,789").split(",")]
US_MIN_WEIGHT = float(os.environ.get("HQP_US_MIN_WEIGHT", "1e-3"))  # noise-floor guard (noise ~1e-6); also the best-effort report floor
US_MARGIN = float(os.environ.get("HQP_US_MARGIN", "5.0"))        # single-ordering trust bar: top1_w >= MARGIN*top2_w. Real peaks observed 7.8-9.4 -> clear 5.0 with cushion; a lone ordering at margin 3-5 (possible confidently-WRONG truncation artifact) no longer early-accepts -> forces a 2nd ordering / consensus (budget is ample when not timeout-limited)
US_CONSENSUS = int(os.environ.get("HQP_US_CONSENSUS", "4"))      # require this many ORDERINGS to agree on the same bitstring to trust via consensus (was 2). Higher = more robust to systematic truncation bias agreeing on a wrong attractor, at the cost of needing more orderings (c64 speed feeds this)
# Best-effort at the deadline: this grader is binary with NO penalty for a wrong answer
# (unlimited resubmissions), so a low-confidence guess strictly beats a guaranteed-zero
# fail-closed. When 1, emit the best above-noise candidate if we never reached confidence.
US_BEST_EFFORT = os.environ.get("HQP_BEST_EFFORT", "1") == "1"


def log(msg):
    # Serialised against the watchdog's emit: the base64 payload is split on the FIRST separator,
    # so a log line landing inside it would corrupt the answer.
    if wall_watchdog.silenced():
        return
    t = datetime.now(timezone.utc).strftime("%H:%M:%S")
    with wall_watchdog.emit_lock:
        if wall_watchdog.silenced():
            return
        print(f"[{t} +{time.time()-START:7.1f}s] {msg}", flush=True)


def time_left():
    return WALL_BUDGET - (time.time() - START)


def _shuffle(n, seed):
    return list(np.random.default_rng(seed).permutation(n))


def _load_circuit(qasm_file):
    from qiskit import qasm2
    from qiskit.circuit.library import UGate
    with open(qasm_file) as f:
        head = f.readline()
    if "3.0" in head:
        import qiskit.qasm3 as qasm3
        return qasm3.load(qasm_file)
    custom = [qasm2.CustomInstruction('u', 3, 1, lambda t, p, l: UGate(t, p, l), builtin=True)]
    return qasm2.load(qasm_file, custom_instructions=custom)


def _solve_statevector(circ):
    from qiskit.quantum_info import Statevector
    sv = Statevector(circ)
    probs = np.abs(np.asarray(sv.data)) ** 2
    idx = int(np.argmax(probs))
    return format(idx, f"0{circ.num_qubits}b")[::-1], float(probs[idx])


def _run_ordering(tebd, qc, chi, perm, dtype, dev):
    t0 = time.time()
    mps, _ = tebd.evolve(qc, chi, dtype=dtype, dev=dev, perm=perm, log=log)
    cands = tebd.topk(mps, beam=BEAM, k=8)
    reached = mps.max_bond()
    del mps
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return cands[0][0], cands[0][1], reached, time.time() - t0


def _solve_tebd(qc):
    import torch
    import tebd_mps as tebd
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.complex64 if os.environ.get("HQP_DTYPE", "c64") == "c64" else torch.complex128  # default c64 (~2-4x faster -> more cross-check orderings, which feeds the stricter consensus bar); HQP_DTYPE=c128 to force double
    n = qc.num_qubits
    log(f"TEBD engine on {dev} ({dtype}); {n} qubits, {qc.size()} gates")

    info = {"levels": [], "method": "tebd_mps_beam+xcheck", "n_qubits": n, "n_gates": qc.size()}
    best_bits = None
    top_chi = None
    primary_w = 0.0
    converged = exact = False
    last_dt = prev_dt = prev_chi = prev2_chi = None
    stable = 0
    prev_top1 = None

    # ---- Phase 1: escalate χ on the primary ordering ----
    for chi in CHI_LADDER:
        if last_dt is None:
            est = 200.0
        elif prev_dt is None or prev2_chi is None or last_dt <= prev_dt:
            est = last_dt * (chi / prev_chi) ** 2 * 1.3
        else:
            p = min(max(math.log(last_dt / prev_dt) / math.log(prev_chi / prev2_chi), 1.0), 3.0)
            est = last_dt * (chi / prev_chi) ** p * 1.4
        if time_left() - SAFETY < est:
            log(f"χ={chi}: skip (est {est:.0f}s > left {time_left()-SAFETY:.0f}s)")
            break
        try:
            top1, w, reached, dt = _run_ordering(tebd, qc, chi, None, dtype, dev)
        except Exception as e:
            log(f"χ={chi}: FAILED ({type(e).__name__}: {str(e)[:120]}); keep best")
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
            break
        prev_dt, last_dt = last_dt, dt
        prev2_chi, prev_chi = prev_chi, chi
        top_chi, primary_w, best_bits = chi, w, top1
        info["levels"].append({"chi": chi, "reached": reached, "top1_w": w, "secs": round(dt, 1)})
        stable = stable + 1 if top1 == prev_top1 else 0
        log(f"χ={chi}: top1_w={w:.4g} reached={reached} stable={stable} ({dt:.1f}s)")
        prev_top1 = top1
        if reached < chi:
            exact = True
            log(f"χ={chi}: bond unsaturated -> EXACT; stop.")
            break
        if stable >= 2 and chi >= STABLE_CHI:
            converged = True
            log(f"χ={chi}: argmax stable >=3 levels at χ>={STABLE_CHI}; stop.")
            break

    # ---- Phase 2: independent cross-check at top χ via different orderings ----
    results = [("primary", best_bits, primary_w)]
    if best_bits is not None and top_chi is not None:
        for k in range(1, N_XCHECK + 1):
            est = last_dt * 1.3 if last_dt else 400.0
            if time_left() - SAFETY < est:
                log(f"xcheck {k}: skip (budget)")
                break
            try:
                perm = _shuffle(n, 1000 + k)
                top1, w, reached, dt = _run_ordering(tebd, qc, top_chi, perm, dtype, dev)
                results.append((f"shuffle{k}", top1, w))
                log(f"xcheck {k} (χ={top_chi}): w={w:.4g} agree={top1 == best_bits} ({dt:.1f}s)")
            except Exception as e:
                log(f"xcheck {k}: FAILED ({type(e).__name__}); skip")
                break

    # ---- Selection: majority vote (robust to a single under-converged outlier) ----
    results = [r for r in results if r[1] is not None]
    consensus = 1
    if results:
        votes = Counter(r[1] for r in results)
        top_count = max(votes.values())
        winners = [b for b, c in votes.items() if c == top_count]
        primary_bits = results[0][1]
        if len(winners) == 1:
            best_bits = winners[0]
        elif primary_bits in winners:
            best_bits = primary_bits
        else:
            best_bits = max(winners, key=lambda b: sum(r[2] for r in results if r[1] == b))
        consensus = votes[best_bits]
        info["consensus"] = f"{consensus}/{len(results)}"
        info["n_distinct"] = len(votes)
        info["selected_w"] = max((r[2] for r in results if r[1] == best_bits), default=0.0)

    # ---- Confidence gate: fail CLOSED unless we trust the answer ----
    multi = len(results) >= 2
    trust = exact or converged or (multi and consensus >= 2)
    if MIN_WEIGHT > 0 and info.get("selected_w", primary_w) < MIN_WEIGHT and not exact:
        trust = False
        log(f"confidence: top1_w {info.get('selected_w'):.3g} < floor {MIN_WEIGHT}; distrust")
    info["exact"] = exact
    info["converged"] = converged
    info["trusted"] = trust
    log(f"VERDICT: exact={exact} converged={converged} consensus={consensus}/{len(results)} "
        f"trusted={trust}")
    return (best_bits if trust else None), info


def _unswap_once(circ, seed, to_backend, deadline=None, allow_abandon=True):
    """One unswap solve at a given ordering seed. Returns (logical_bits, top1_w, top2_w, final_bond).

    `deadline` (absolute time.time()) bounds the heavy MPO absorption: on a hard
    circuit the absorber stops early and extracts the best-so-far MPS rather than
    overrunning the hard kill with no output (graceful fail-closed, not a crash)."""
    import extract
    from unswap import mpo_compress_unswap, mpo_to_mps
    mpo, ll, lr, _ = mpo_compress_unswap(
        circ, seed=seed, to_backend=to_backend, cutoff=US_CUTOFF, max_bond=US_MAXBOND,
        unswap_threshold=1e6, center_ratio=0.5, equal=US_EQUAL, flip_freq=None,
        max_its=20, early_stopping_gates=US_EARLY_STOP, hows=("both", "left", "right"),
        deadline=deadline, allow_abandon=allow_abandon)
    # The one-shot extraction must be NUMERICALLY STABLE: the fast eigh-trick SVD (great for the
    # heavy absorption) squares the condition number and can emit NaNs on the large ill-conditioned
    # extraction compressions. Restore the stable gesvd for extraction (absorption already done).
    try:
        import fast_svd as _fs
        _fs.restore_svd()
    except Exception:
        pass
    # OOM-safe extraction: try the configured bond, then halve on CUDA OOM. The core MPO is
    # tiny post-unswap, so a lower extraction bond mainly truncates the leftover front layers
    # whose peak is already the dominant amplitude -> robust on the 96 GB card.
    from unswap import ExtractionDeadline, OrderingAbandoned
    fmb = US_FINAL_MAXBOND
    # Bound the extraction: this was the last unbounded stage, and an OOM ladder that halves the
    # bond and retries can otherwise run past the validator's hard kill (-> WallTimeFailure with
    # no answer at all). Leave EXTRACT_SAFETY for beam search + writing the result. Signalling with
    # OrderingAbandoned keeps this function's (bits, w1, w2, bond) contract: the caller already
    # treats that as "drop this ordering, keep the earlier ones".
    # Anchor on the WALL, never on `deadline`: the caller passes the ABSORPTION deadline
    # (compress_deadline = START + WALL_BUDGET - HQP_EXTRACT_RESERVE), and that reserve exists
    # precisely to pay for this extraction. Deriving from it put the extraction clock 420 s BEFORE
    # absorption was even allowed to stop, so every ordering that absorbed to its deadline -- the
    # hard-instance case -- was discarded unextracted and the whole run ended with no candidate.
    extract_deadline = START + WALL_BUDGET - EXTRACT_SAFETY
    while True:
        if time.time() > extract_deadline:
            try:
                import torch as _t
                _t.cuda.empty_cache()
            except Exception:
                pass
            raise OrderingAbandoned(reason=f"extraction budget exhausted before bond {fmb}")
        try:
            mps, perm = mpo_to_mps(mpo, ll[:-2], lr, cutoff=US_FINAL_CUTOFF,
                                   to_backend=to_backend, max_bond=fmb,
                                   deadline=extract_deadline)
            break
        except ExtractionDeadline as _e:
            try:
                import torch as _t
                _t.cuda.empty_cache()
            except Exception:
                pass
            raise OrderingAbandoned(reason=f"{_e} (no overrun)")
        except (RuntimeError, MemoryError) as _e:
            if "out of memory" in str(_e).lower() and fmb > 256:
                try:
                    import torch as _t
                    _t.cuda.empty_cache()
                except Exception:
                    pass
                fmb //= 2
                print(f"[extract] OOM at bond {fmb*2} -> retry at {fmb}", flush=True)
                continue
            raise
    cands = extract.beam_search(mps, beam=BEAM, k=8)
    top1, w1 = cands[0]
    w2 = cands[1][1] if len(cands) > 1 else 0.0
    logical = "".join(top1[i] for i in perm)
    return logical, w1, w2, int(mps.max_bond())


def tally(res):
    """Reduce accumulated orderings -> (best_bits, consensus_count, best_weight, best_margin).

    res is a list of (bitstring, weight, margin). The winner is the bitstring with the
    most votes (consensus). When orderings DISAGREE (no consensus / tie), pick the
    candidate we are most confident in: highest margin first (cleanest peak isolation),
    then highest weight — this is what gets handed back as the best-effort answer.
    """
    from collections import Counter
    if not res:
        return None, 0, 0.0, 0.0
    votes = Counter(b for b, _, _ in res)
    top = max(votes.values())
    winners = [b for b, c in votes.items() if c == top]
    best = winners[0] if len(winners) == 1 else max(
        winners,
        key=lambda b: (max(m for bb, w, m in res if bb == b),
                       max(w for bb, w, m in res if bb == b)))
    consensus = votes[best]
    best_w = max((w for b, w, m in res if b == best), default=0.0)
    best_margin = max((m for b, w, m in res if b == best), default=0.0)
    return best, consensus, best_w, best_margin


def is_confident(res):
    """Trust ONLY on strong evidence: 2+ orderings agree on the same bitstring, OR one
    sharply-peaked ordering (margin >= US_MARGIN). A bare weight floor is NOT enough —
    ambiguous orderings reach w~0.08 at margin~1.0, well above the 1e-3 floor."""
    if not res:
        return False
    _, consensus, best_w, best_margin = tally(res)
    # AND, not OR: trust only when BOTH hold -> >=US_CONSENSUS orderings agree on the same
    # bitstring AND that bitstring is sharply peaked (margin >= US_MARGIN). A lone sharp
    # ordering is NOT enough (can be a confidently-wrong truncation artifact); mushy
    # agreement is NOT enough (could be systematic bias agreeing on a wrong attractor).
    return consensus >= US_CONSENSUS and best_margin >= US_MARGIN and best_w >= US_MIN_WEIGHT


def decide(res):
    """Final verdict from accumulated orderings -> (peak_or_None, confidence_label).

      high        : confident -> >=US_CONSENSUS orderings agree AND margin>=US_MARGIN -> trust.
      best_effort : not confident but a real above-noise candidate -> report anyway, since
                    on this binary grader a wrong answer == no answer (no penalty, unlimited
                    resubmissions), so a guess strictly beats a guaranteed-zero fail-closed.
      none        : nothing above the noise floor -> genuinely fail closed.
    """
    if not res:
        return None, "none"
    best, consensus, best_w, best_margin = tally(res)
    if is_confident(res):
        return best, "high"
    if US_BEST_EFFORT and best_w >= US_MIN_WEIGHT:
        return best, "best_effort"
    return None, "none"


def _solve_unswap(qc):
    """MPO iterative-cancellation unswapping + memory-bounded zip-up apply + canonical
    beam-search argmax. Unswapping reduces the effective entanglement so a feasible bond
    resolves the embedded peak (where plain TEBD drowns in noise). Cross-checks across
    qubit orderings and fails CLOSED unless confident."""
    import torch
    import fp32_patch  # noqa: F401  (patches quimb.sgn + torch SVD/QR for FP32)
    from unswap import OrderingAbandoned  # early-abandon signal (visible to run_ordering below)
    os.environ["HQP_SABRE_TRIALS"] = str(US_SABRE_TRIALS)
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
    from qiskit.transpiler import PassManager

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.complex64 if os.environ.get("HQP_DTYPE", "c64") == "c64" else torch.complex128  # default c64 (~2-4x faster -> more cross-check orderings, which feeds the stricter consensus bar); HQP_DTYPE=c128 to force double
    def to_backend(x):
        return torch.tensor(x, dtype=dtype, device=dev)

    n = qc.num_qubits
    log(f"unswap engine on {dev} ({dtype}); {n} qubits, {qc.size()} gates")
    circ = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(qc)

    info = {"method": "unswap_mpo_beam+xcheck", "n_qubits": n, "orderings": []}
    results = []          # (bits, w1)
    last_dt = None
    # Absolute wall-clock deadline for the heavy MPO absorption. Stop early enough
    # to leave room for MPS extraction + beam + stdout output before the hard kill,
    # so a single hard ordering degrades to a best-so-far answer, never a no-output crash.
    extract_reserve = float(os.environ.get("HQP_EXTRACT_RESERVE", "900"))
    compress_deadline = START + WALL_BUDGET - extract_reserve

    def run_ordering(label, seed, allow_abandon=True):
        """Run one ordering; record (bits, w1, margin). The seed loops advance unconditionally;
        is_confident(results) is the only stop. Returns None on success, False on failure/abandon."""
        nonlocal last_dt
        t0 = time.time()
        try:
            bits, w1, w2, fb = _unswap_once(circ, seed, to_backend, deadline=compress_deadline, allow_abandon=allow_abandon)
        except OrderingAbandoned as ab:
            log(f"{label} (seed {seed}): ABANDONED ({ab}) -> next ordering")
            try: torch.cuda.empty_cache()
            except Exception: pass
            return False
        except Exception as e:
            log(f"{label} (seed {seed}): FAILED ({type(e).__name__}: {str(e)[:120]})")
            try: torch.cuda.empty_cache()
            except Exception: pass
            return False
        last_dt = time.time() - t0
        if not (w1 == w1 and w2 == w2):        # NaN beam weights: not a result, never 'inf' margin
            log(f"{label} (seed {seed}): NaN beam weights -> ordering discarded")
            try: torch.cuda.empty_cache()
            except Exception: pass
            return False
        margin = w1 / w2 if w2 > 0 else float("inf")
        results.append((bits, w1, margin))
        wall_watchdog.publish(bits, {"method": "unswap", "margin": margin, "peak_prob": w1,
                                     "final_bond": fb}, trusted=False, score=margin,
                              stage=f"unswap/{label}", prio=1)
        info["orderings"].append({"seed": seed, "w1": w1, "w2": w2, "final_bond": fb, "secs": round(last_dt, 1)})
        log(f"{label} (seed {seed}): w1={w1:.4g} w2={w2:.4g} margin={margin:.2f} final_bond={fb} ({last_dt:.1f}s)")
        try: torch.cuda.empty_cache()
        except Exception: pass
        if w1 >= US_MIN_WEIGHT and margin >= US_MARGIN:
            # Sharp ordering, but under AND logic a single one is NOT sufficient — we still
            # need US_CONSENSUS orderings to agree before trusting. Do not stop here.
            log(f"{label}: sharp (margin>={US_MARGIN}) — need {US_CONSENSUS}-way consensus to accept")
        return None

    # Phase 1: default deterministic seeds.
    for i, seed in enumerate(US_SEEDS):
        est = (last_dt * 1.2) if last_dt else 1500.0
        if time_left() - SAFETY < est:
            log(f"ordering {i} (seed {seed}): skip (budget {time_left()-SAFETY:.0f}s < est {est:.0f}s)")
            break
        # FINAL-ordering guard: don't abandon the last default seed if there's no budget for the
        # random-seed fallback -- run it to completion for a best-effort answer rather than nothing.
        is_last_default = (i == len(US_SEEDS) - 1)
        fallback_budget = time_left() > extract_reserve + SAFETY + 1500.0
        allow_ab = not (is_last_default and not fallback_budget)
        run_ordering(f"ordering {i}", seed, allow_abandon=allow_ab)
        if is_confident(results):
            break

    # Phase 2: random-seed fallback, only if not yet confident AND budget is ample.
    # Strict timeline: a new ordering is started only if it can finish and still leave
    # extract_reserve + SAFETY for MPS extraction + beam + stdout before the 4h kill.
    if not is_confident(results) and time_left() > extract_reserve + SAFETY + 1500.0:
        log(f"default seeds inconclusive; random-seed fallback (time_left={time_left():.0f}s)")
        rng = np.random.default_rng(int(START))   # deterministic -> reproducible on the validator
        for rand_idx in range(int(os.environ.get("HQP_MAX_RANDOM_SEEDS", "20"))):  # need many orderings to reach consensus=4; the wall-clock budget gate is the real limiter, so a high cap is harmless
            if time_left() < extract_reserve + SAFETY + 1200.0:
                log(f"random ordering {rand_idx}: STOP (time_left={time_left():.0f}s below safe minimum to finish + extract)")
                break
            est = (last_dt * 1.2) if last_dt else 1500.0
            if time_left() - SAFETY < est:
                log(f"random ordering {rand_idx}: skip (budget {time_left()-SAFETY:.0f}s < est {est:.0f}s)")
                break
            seed = int(rng.integers(1000, 1000000))
            # If there's no budget to start another ordering after this one, don't abandon it.
            next_affordable = (time_left() - (last_dt or 1500.0)) > extract_reserve + SAFETY + 1200.0
            run_ordering(f"random ordering {rand_idx}", seed, allow_abandon=next_affordable)
            if is_confident(results):
                log(f"reached {US_CONSENSUS}-way consensus + margin>={US_MARGIN}; stop")
                break

    if not results:
        info["trusted"] = False
        info["confidence"] = "none"
        log("VERDICT: no ordering produced a result; fail-closed")
        return None, info

    best, consensus, best_w, best_margin = tally(results)
    peak, label = decide(results)
    info["consensus"] = f"{consensus}/{len(results)}"
    info["selected_w"] = best_w
    info["selected_margin"] = (round(best_margin, 3) if best_margin != float("inf") else "inf")
    info["trusted"] = (label == "high")
    info["confidence"] = label
    mstr = "inf" if best_margin == float("inf") else f"{best_margin:.2f}"
    log(f"VERDICT: {label} consensus={consensus}/{len(results)} w={best_w:.4g} margin={mstr} "
        f"-> {'report ' + peak if peak else 'fail-closed (None)'}")
    return peak, info


def _zx_simplify(circ):
    """IDEA #11: ZX-calculus simplification to cancel hidden identity (U.U^-1) structure and
    cut 2-qubit gates BEFORE simulation -> less entanglement for the engine to fight. Env-gated
    (HQP_ZX=1, default off). Correctness-safe: returns the ORIGINAL circuit on any failure OR
    if simplification doesn't actually reduce the 2q-gate count (these circuits are designed to
    resist transpiler-level simplification, so this is exploratory)."""
    if os.environ.get("HQP_ZX", "0") != "1":
        return circ
    try:
        import pyzx as zx
        from qiskit import qasm2
        n0 = sum(1 for inst in circ.data if len(inst.qubits) == 2)
        zc = zx.Circuit.from_qasm(qasm2.dumps(circ))
        g = zc.to_graph()
        zx.full_reduce(g)
        zc2 = zx.extract_circuit(g.copy()).to_basic_gates()
        out = qasm2.loads(zc2.to_qasm())
        n1 = sum(1 for inst in out.data if len(inst.qubits) == 2)
        log(f"[#11 ZX] 2q gates {n0} -> {n1} ({'KEEP' if n1 < n0 else 'no reduction, discard'})")
        return out if n1 < n0 else circ
    except Exception as e:
        log(f"[#11 ZX] failed ({type(e).__name__}: {str(e)[:120]}); using original circuit")
        return circ


def _solve_canonical(circ):
    """Fast-MPS canonical chi-ladder + beam (torch eigh-trick engine, mps_torch).

    Shallow all-to-all peaked circuits (difficulty-1) are directly simulable: a modest
    bond resolves the peak, which the heavy unswap instead over-processes into a flat
    result. Runs a cheap chi ladder; a bitstring that is sharply peaked (margin >= gate)
    AND stable across two rungs is trusted. cz is symmetric so mps_torch's gate
    convention is exact on raw u3+cz circuits (validated: d1_s1 chi512 -> Hamming 0)."""
    import mps_torch  # local torch engine
    n = circ.num_qubits
    ladder = [int(x) for x in os.environ.get("HQP_CANON_CHI", "256,512,1024").split(",")]
    # STABILITY across rungs is the discriminator: a directly-simulable peak (d1) locks the
    # same argmax at chi 512 and 1024; a circuit past the MPS wall (d3) flips its noise argmax
    # every rung. So trust a top that (a) repeats across two consecutive rungs and (b) clears a
    # low margin floor (a real peak edges out its neighbours; pure-noise ties do not). A single
    # very sharp rung (margin >= sharp) is trusted immediately.
    floor = float(os.environ.get("HQP_CANON_MARGIN", "1.15"))
    sharp = float(os.environ.get("HQP_CANON_SHARP", "3.0"))
    budget = float(os.environ.get("HQP_CANON_BUDGET", "2400"))  # leave the rest for unswap
    t0 = time.time()
    prev_top = None
    best = None
    for chi in ladder:
        if time.time() - t0 > budget or time_left() < SAFETY + 600:
            break
        try:
            mps, dt = mps_torch.evolve(circ, chi, log_every=0)
            cands = mps_torch.topk(mps, beam=512, k=8)
        except Exception as e:  # noqa: BLE001
            log(f"  canon chi={chi} failed: {type(e).__name__}: {str(e)[:80]}")
            break
        top0, w0 = cands[0]
        w1 = cands[1][1] if len(cands) > 1 else 0.0
        margin = w0 / max(w1, 1e-300)
        stable = top0 == prev_top
        log(f"  canon chi={chi}: {dt:.0f}s w0={w0:.3e} margin={margin:.2f} "
            f"stable={stable} top={top0[:24]}...")
        if best is None or margin > best[2]:
            best = (top0, w0, margin, chi)
        wall_watchdog.publish(top0, {"method": "canonical", "margin": margin, "chi": chi,
                                     "peak_prob": w0}, trusted=False, score=margin,
                              stage=f"canonical/chi{chi}", prio=0)
        if margin >= sharp or (stable and margin >= floor):
            why = "sharp" if margin >= sharp else f"stable-across-rungs (margin {margin:.2f})"
            log(f"  canon: {why} -> trust")
            return top0, {"method": "canonical", "confident": True, "margin": margin,
                          "chi": chi, "n_qubits": n, "trusted": True}
        prev_top = top0
    top0 = best[0] if best else None
    return top0, {"method": "canonical", "confident": False,
                  "margin": best[2] if best else 0.0, "n_qubits": n}


def solve(qasm_file):
    circ = _load_circuit(qasm_file)
    circ = _zx_simplify(circ)          # IDEA #11 (no-op unless HQP_ZX=1)
    nq = circ.num_qubits
    log(f"Circuit: {nq} qubits, {circ.size()} gates")
    if nq <= int(os.environ.get("HQP_STATEVECTOR_MAX_Q", "30")):      # env override exists for integration tests only
        log("Exact statevector")
        bits, p = _solve_statevector(circ)
        return bits, {"method": "statevector", "n_qubits": nq, "peak_prob": p,
                      "exact": True, "trusted": True}
    if ENGINE == "tebd":
        return _solve_tebd(circ)
    # Portfolio: fast-canonical first (cheap; cracks shallow all-to-all like d1 that unswap
    # over-processes). Trust it only if sharply peaked; else fall to the unswap heavy hammer.
    # tau-block excision (d3): strip the mirror blocks, then the remainder is d1-shaped and the
    # canonical ladder cracks it directly. Fails closed -> falls through to canonical/unswap.
    excision_best = None
    # v13 (2026-09-19): whole-block excision (cut tolerance 1e-3: the swept outer half of the big block), D measured
    # with the explicit swaps at the block centre + reference-free decode, concurrent D workers, and a multi-opinion
    # readout (ladder rungs, pooled multi-order ensemble, Pauli-path gate). HQP_V13=0 restores the v9 excision path.
    # Any failure falls through to the legacy stages below, which also keep their own published candidates.
    if os.environ.get("HQP_V13", "1").strip() not in ("0", "false", "False") and nq >= int(os.environ.get("V13_MIN_Q", "44")):
        try:
            import v13_solver
            _seq = [0]

            def _v13_publish(bits, meta):
                _seq[0] += 1
                wall_watchdog.publish(bits, dict(meta or {}, method="v13"), trusted=bool((meta or {}).get("trusted")),
                                      score=float(_seq[0]), stage="v13", prio=3)

            vbits, vinfo = v13_solver.solve_v13(qasm_file, log, time.time() + max(time_left() - SAFETY, 0.0),
                                                publish=_v13_publish)
            if vbits is not None:
                log(f"v13 answered (trusted={vinfo.get('trusted')}, {vinfo.get('verdict', {}).get('why')}) -> using it")
                return vbits, vinfo
            log("v13 produced no answer -> legacy excision path")
        except Exception as e:  # noqa: BLE001
            log(f"v13 stage error ({type(e).__name__}: {str(e)[:160]}) -> legacy excision path")
    try:
        import excision_solver
        if excision_solver.EXCISE_ON and nq >= excision_solver.MIN_Q:
            def _tb(x):
                import torch
                return torch.tensor(x, dtype=torch.complex64, device="cuda")
            ebits, einfo = excision_solver.solve_excision(
                circ, log=log, to_backend=_tb,
                deadline=(time.time() + max(time_left() - SAFETY, 0.0)))
            if ebits is not None and einfo.get("trusted"):
                log(f"excision stage solved it (margin {einfo.get('margin', 0):.2f}) -> using it")
                return ebits, einfo
            if ebits is not None:
                # untrusted but real candidate: keep it as the answer of last resort. A wrong
                # answer and no answer both fail validation, so returning it is weakly dominant.
                excision_best = (ebits, dict(einfo))
                log(f"excision stage left a best-effort candidate (margin {einfo.get('margin', 0):.2f})")
                if einfo.get("blocks_found", 0) >= 2 and \
                        os.environ.get("HQP_EXCISE_SKIP_FALLBACKS", "1").strip() not in ("0", "false", "False"):
                    # d3-structured circuit: the canonical stage on the ORIGINAL circuit can 'trust' a
                    # stable noise argmax and the unswap stage cannot finish inside the wall (E65/E73),
                    # so neither can beat the excision candidate. Return it now, inside the budget.
                    binfo = dict(einfo); binfo["trusted"] = False
                    log("d3-structured circuit -> skipping original-circuit fallbacks, returning best-effort")
                    return ebits, binfo
    except Exception as e:  # noqa: BLE001
        log(f"excision stage error ({type(e).__name__}: {str(e)[:140]}) -> continuing")

    if os.environ.get("HQP_CANON_FIRST", "1").strip() not in ("0", "false", "False"):
        try:
            cbits, cinfo = _solve_canonical(circ)
            if cinfo.get("confident"):
                log(f"canonical stage confident (margin {cinfo['margin']:.2f}) -> using it")
                return cbits, cinfo
            log("canonical stage not confident -> unswap heavy hammer")
        except Exception as e:  # noqa: BLE001
            log(f"canonical stage error ({type(e).__name__}: {str(e)[:100]}) -> unswap")
    if ENGINE != "unswap":
        # Cold-engine dispatch (v4-v8): a sibling module engine_<name>.py exposing
        # solve_engine(circ, log) -> (bits, info). Falls back to unswap if unavailable.
        try:
            import importlib
            mod = importlib.import_module(f"engine_{ENGINE}")
            log(f"dispatch to cold engine: {ENGINE}")
            return mod.solve_engine(circ, log=log)
        except Exception as e:
            log(f"cold engine '{ENGINE}' unavailable ({type(e).__name__}: {str(e)[:120]}); falling back to unswap")
    ubits, uinfo = _solve_unswap(circ)
    if ubits is None and excision_best is not None and \
            os.environ.get("HQP_EXCISE_BEST_EFFORT", "1").strip() not in ("0", "false", "False"):
        bbits, binfo = excision_best
        log(f"all stages untrusted -> returning the excision best-effort candidate "
            f"(margin {binfo.get('margin', 0):.2f}) rather than nothing")
        binfo["trusted"] = False
        return bbits, binfo
    return ubits, uinfo


def main():
    try:
        challenge_id, problem = load_solver_input(sys.argv)
    except Exception as err:
        print(f"Error loading HQP input:\n{err}")
        sys.exit(1)

    ts_start = datetime.now(timezone.utc).isoformat()
    log(f"HQP {challenge_id} difficulty={problem.difficulty} qasm={problem.qasm_file}")

    def emit(peak, info, reason=None):
        """Write the ONE payload and exit. Shared by the normal path and the wall watchdog.

        Never returns. Must stay cheap and allocation-light: the watchdog calls it while the main
        thread may be wedged inside a CUDA kernel.
        """
        if not wall_watchdog.claim_emit():
            # The watchdog won the race and is already printing the payload; it will os._exit the
            # process. Emitting too would put a SECOND separator on stdout and corrupt the answer.
            time.sleep(60)
            return
        status = "success" if peak else "failed"
        solution = Solution(status, peak)
        result_json = json.dumps(solution.to_dict(), indent=2)
        try:
            solve_info_json = json.dumps({
                "solution_status": status,
                "challenge_id": challenge_id,
                "timestamp_utc": ts_start,
                "solve_time_seconds": time.time() - START,
                "difficulty": problem.difficulty,
                "emit_reason": reason or "normal completion",
                **(info or {}),
            }, default=str)
        except Exception:
            solve_info_json = json.dumps({"solution_status": status, "emit_reason": str(reason)})

        output_dir = os.environ.get("OUTPUT_DIR")
        if output_dir:
            try:
                Path(output_dir).mkdir(exist_ok=True)
                Path(output_dir, "result.json").write_text(result_json)
                Path(output_dir, "solve_info.json").write_text(solve_info_json)
            except OSError:
                pass

        _emit_payload(build_solution_zip({
            "result.json": result_json,
            "solve_info.json": solve_info_json,
        }))
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os._exit(0 if status == "success" else 1)

    def watchdog_emit(bits, info, reason=None):
        # The log lock is already held by the watchdog, so print through the raw stream: log()
        # would deadlock on the RLock only if re-entered from another thread, and is silenced now.
        print(f"[watchdog] {reason}; emitting best candidate "
              f"({'none - reporting failure' if not bits else info.get('watchdog_stage') or 'unknown stage'})",
              flush=True)
        emit(bits, info, reason=reason)

    wall_watchdog.arm(watchdog_emit, log=log)

    info = {}
    try:
        peak, info = solve(problem.qasm_file)
    except Exception as e:
        import traceback
        log(f"Solver error: {type(e).__name__}: {e}")
        traceback.print_exc()
        peak = None
        # A crashed stage must not throw away a candidate an earlier stage already found.
        bits, binfo, trusted, stage = wall_watchdog.best()
        if bits:
            log(f"recovering the best published candidate from stage '{stage}' after the error")
            peak, info = bits, dict(binfo, recovered_after_error=True)

    if not peak:
        # solve() can also RETURN empty (e.g. unswap produced nothing and there was no excision
        # best-effort) while an earlier stage published a real candidate -- the canonical ladder's
        # unconfident argmax, say. A wrong answer and no answer both score zero, so emitting the
        # best published candidate is weakly dominant (same stance as HQP_BEST_EFFORT).
        bits, binfo, trusted, stage = wall_watchdog.best()
        if bits and US_BEST_EFFORT:
            log(f"solve() returned no peak -> emitting the best published candidate from '{stage}'")
            peak, info = bits, dict(binfo, recovered_from_publish=True)
    log(f"FINAL status={'success' if peak else 'failed'} peak={peak}")
    emit(peak, info)


if __name__ == "__main__":
    main()
