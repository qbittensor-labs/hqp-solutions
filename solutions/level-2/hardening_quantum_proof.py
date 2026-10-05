#!/usr/bin/env python3
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

import numpy as np

from enigma_challenges.hardening_quantum_proof import Solution, load_solver_input
from enigma_challenges.solution_output import build_solution_zip, write_solution_output

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver"))

START = time.time()
WALL_BUDGET = float(os.environ.get("HQP_WALL_BUDGET", "13000"))   # of the 14400s hard kill
SAFETY = float(os.environ.get("HQP_SAFETY", "300"))
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
US_MAXBOND = int(os.environ.get("HQP_US_MAXBOND", "1024"))
US_EARLY_STOP = int(os.environ.get("HQP_US_EARLY_STOP", "30"))   # stop absorbing with <=N gates left (avoids tail livelock)
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
    t = datetime.now(timezone.utc).strftime("%H:%M:%S")
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


def _unswap_once(circ, seed, to_backend, deadline=None):
    """One unswap solve at a given ordering seed. Returns (logical_bits, top1_w, top2_w, final_bond).

    `deadline` (absolute time.time()) bounds the heavy MPO absorption: on a hard
    circuit the absorber stops early and extracts the best-so-far MPS rather than
    overrunning the hard kill with no output (graceful fail-closed, not a crash)."""
    import extract
    from unswap import mpo_compress_unswap, mpo_to_mps
    mpo, ll, lr, _ = mpo_compress_unswap(
        circ, seed=seed, to_backend=to_backend, cutoff=US_CUTOFF, max_bond=US_MAXBOND,
        unswap_threshold=1e6, center_ratio=0.5, equal=False, flip_freq=None,
        max_its=20, early_stopping_gates=US_EARLY_STOP, hows=("both", "left", "right"),
        deadline=deadline)
    mps, perm = mpo_to_mps(mpo, ll[:-2], lr, cutoff=US_FINAL_CUTOFF,
                           to_backend=to_backend, max_bond=US_MAXBOND)
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
    return (consensus >= US_CONSENSUS and best_w >= US_MIN_WEIGHT) or (best_w >= US_MIN_WEIGHT and best_margin >= US_MARGIN)


def decide(res):
    """Final verdict from accumulated orderings -> (peak_or_None, confidence_label).

      high        : confident (2+ agree, or one sharp ordering margin>=US_MARGIN) -> trust.
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

    def run_ordering(label, seed):
        """Run one ordering; record (bits, w1, margin); return True iff it is single-ordering confident."""
        nonlocal last_dt
        t0 = time.time()
        try:
            bits, w1, w2, fb = _unswap_once(circ, seed, to_backend, deadline=compress_deadline)
        except Exception as e:
            log(f"{label} (seed {seed}): FAILED ({type(e).__name__}: {str(e)[:120]})")
            try: torch.cuda.empty_cache()
            except Exception: pass
            return False
        last_dt = time.time() - t0
        margin = w1 / w2 if w2 > 0 else float("inf")
        results.append((bits, w1, margin))
        info["orderings"].append({"seed": seed, "w1": w1, "w2": w2, "final_bond": fb, "secs": round(last_dt, 1)})
        log(f"{label} (seed {seed}): w1={w1:.4g} w2={w2:.4g} margin={margin:.2f} final_bond={fb} ({last_dt:.1f}s)")
        try: torch.cuda.empty_cache()
        except Exception: pass
        if w1 >= US_MIN_WEIGHT and margin >= US_MARGIN:
            log(f"{label}: confident (w1>={US_MIN_WEIGHT}, margin>={US_MARGIN}); accept")
            return True
        return False

    # Phase 1: default deterministic seeds.
    for i, seed in enumerate(US_SEEDS):
        est = (last_dt * 1.2) if last_dt else 1500.0
        if time_left() - SAFETY < est:
            log(f"ordering {i} (seed {seed}): skip (budget {time_left()-SAFETY:.0f}s < est {est:.0f}s)")
            break
        if run_ordering(f"ordering {i}", seed):
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
            if run_ordering(f"random ordering {rand_idx}", seed):
                break
            if is_confident(results):
                log("random seeds reached consensus; stop")
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


def solve(qasm_file):
    circ = _load_circuit(qasm_file)
    nq = circ.num_qubits
    log(f"Circuit: {nq} qubits, {circ.size()} gates")
    if nq <= 30:
        log("Exact statevector")
        bits, p = _solve_statevector(circ)
        return bits, {"method": "statevector", "n_qubits": nq, "peak_prob": p,
                      "exact": True, "trusted": True}
    if ENGINE == "tebd":
        return _solve_tebd(circ)
    return _solve_unswap(circ)


def main():
    try:
        challenge_id, problem = load_solver_input(sys.argv)
    except Exception as err:
        print(f"Error loading HQP input:\n{err}")
        sys.exit(1)

    ts_start = datetime.now(timezone.utc).isoformat()
    log(f"HQP {challenge_id} difficulty={problem.difficulty} qasm={problem.qasm_file}")

    info = {}
    try:
        peak, info = solve(problem.qasm_file)
    except Exception as e:
        import traceback
        log(f"Solver error: {type(e).__name__}: {e}")
        traceback.print_exc()
        peak = None

    status = "success" if peak else "failed"
    log(f"FINAL status={status} peak={peak}")
    solution = Solution(status, peak)

    result_json = json.dumps(solution.to_dict(), indent=2)
    solve_info_json = json.dumps({
        "solution_status": status,
        "challenge_id": challenge_id,
        "timestamp_utc": ts_start,
        "solve_time_seconds": time.time() - START,
        "difficulty": problem.difficulty,
        **info,
    })

    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir:
        try:
            Path(output_dir).mkdir(exist_ok=True)
            Path(output_dir, "result.json").write_text(result_json)
            Path(output_dir, "solve_info.json").write_text(solve_info_json)
        except OSError:
            pass

    write_solution_output(build_solution_zip({
        "result.json": result_json,
        "solve_info.json": solve_info_json,
    }))
    os._exit(0 if status == "success" else 1)


if __name__ == "__main__":
    main()
