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

"""Hardening Quantum Proof solver — peaked-circuit peak finder.

Strategy: evolve the circuit as a truncated matrix-product state on GPU at an
escalating bond dimension, extracting the highest-probability bitstring (the
peak) via canonical beam search.  Truncation denoises the random background and
concentrates probability on the embedded peak; beam-search argmax (NOT sampling)
recovers it even at modest bond dimension.

Two layers of verification (exact single-amplitude contraction is infeasible:
these circuits have treewidth ~= qubit count, ~2^58 flops/amplitude):
  1. Convergence: the argmax is stable as the bond dimension grows.
  2. Independent cross-check: re-run at the top bond dimension with different
     qubit orderings (independent truncation errors). The answer is taken from
     the most-peaked run (highest MPS weight); agreement across orderings is a
     strong correctness signal.

Output: stdout solution protocol (logs, separator, base64 zip of result.json +
solve_info.json), matching enigma_challenges.solution_output.
"""
from datetime import datetime, timezone
import json
import math
import os
import sys
import time

from enigma_challenges.hardening_quantum_proof import Solution, load_solver_input
from enigma_challenges.solution_output import build_solution_zip, write_solution_output

START = time.time()
WALL_BUDGET = float(os.environ.get("HQP_WALL_BUDGET", "13000"))  # of the 4h=14400s hard kill
SAFETY = float(os.environ.get("HQP_SAFETY", "300"))               # output/teardown margin
CHI_LADDER = [int(x) for x in os.environ.get(
    "HQP_CHI_LADDER",
    "48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096,6144,8192").split(",")]
BEAM = int(os.environ.get("HQP_BEAM", "512"))
# number of extra qubit orderings for the independent cross-check at top chi
N_XCHECK = int(os.environ.get("HQP_XCHECK", "2"))


def log(msg: str) -> None:
    t = datetime.now(timezone.utc).strftime("%H:%M:%S")
    el = time.time() - START
    print(f"[{t} +{el:7.1f}s] {msg}", flush=True)


def time_left() -> float:
    return WALL_BUDGET - (time.time() - START)


def _shuffle(n, seed):
    """Deterministic permutation of range(n) (initial MPS chain layout)."""
    import numpy as np
    return list(np.random.default_rng(seed).permutation(n))


def run_ordering(mps_gpu, qc, chi, perm, beam, log):
    """One MPS evolution + argmax. Returns (top1, weight, reached_chi, secs)."""
    t0 = time.time()
    mps, _ = mps_gpu.evolve(qc, chi, perm=perm, log=log)
    cands = mps_gpu.topk(mps, beam=beam, k=8)
    reached = mps.max_bond()
    del mps
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return cands[0][0], cands[0][1], reached, time.time() - t0


def main() -> None:
    try:
        challenge_id, problem = load_solver_input(sys.argv)
    except Exception as err:
        print(f"Error loading HQP input:\n{err}")
        sys.exit(1)

    ts_start = datetime.now(timezone.utc).isoformat()
    log(f"HQP challenge {challenge_id} difficulty={problem.difficulty}")
    log(f"QASM: {problem.qasm_file}  wall_budget={WALL_BUDGET:.0f}s")

    best_bits = None
    info = {"levels": [], "method": "gpu_mps_canonical_beam+ordering_xcheck"}

    try:
        from qload import load_qiskit
        import mps_gpu
        qc = load_qiskit(problem.qasm_file)
        n = qc.num_qubits
        log(f"Circuit: {n} qubits, {len(qc.data)} gates")
        info["n_qubits"] = n
        info["n_gates"] = len(qc.data)

        # ---- Phase 1: escalate bond dimension on the primary ordering ----
        last_dt = prev_dt = prev_chi = prev2_chi = None
        stable_count = 0
        prev_top1 = None
        top_chi = None
        primary_w = 0.0
        for chi in CHI_LADDER:
            if last_dt is None:
                est = 150.0
            elif prev_dt is None or prev2_chi is None or last_dt <= prev_dt:
                est = last_dt * (chi / prev_chi) ** 2 * 1.3
            else:
                p = math.log(last_dt / prev_dt) / math.log(prev_chi / prev2_chi)
                p = min(max(p, 1.0), 3.0)
                est = last_dt * (chi / prev_chi) ** p * 1.4
            if time_left() - SAFETY < est:
                log(f"χ={chi}: skip (est {est:.0f}s > budget left {time_left()-SAFETY:.0f}s)")
                break
            try:
                top1, w, reached, dt = run_ordering(mps_gpu, qc, chi, None, BEAM, log)
                prev_dt, last_dt = last_dt, dt
                prev2_chi, prev_chi = prev_chi, chi
                top_chi, primary_w, best_bits = chi, w, top1
                info["levels"].append({"chi": chi, "reached_chi": reached,
                                       "top1_w": w, "secs": round(dt, 1)})
                log(f"χ={chi}: top1 w={w:.4g} reachedχ={reached} ({dt:.1f}s) "
                    f"stable={stable_count}")
                stable_count = stable_count + 1 if top1 == prev_top1 else 0
                prev_top1 = top1
                if reached < chi:
                    log(f"χ={chi}: bond unsaturated (exact); stop escalation.")
                    break
                if stable_count >= 2 and chi >= 1024:
                    log(f"χ={chi}: argmax stable ≥3 levels at χ≥1024; stop escalation.")
                    break
            except Exception as e:
                log(f"χ={chi}: FAILED ({type(e).__name__}: {str(e)[:140]}); keep best")
                try:
                    import cupy
                    cupy.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                break

        # ---- Phase 2: independent cross-check at the top bond dimension ----
        results = [("primary", best_bits, primary_w)]
        if best_bits is not None and top_chi is not None:
            for k in range(1, N_XCHECK + 1):
                est = last_dt * 1.2 if last_dt else 300.0
                if time_left() - SAFETY < est:
                    log(f"xcheck {k}: skip (budget); ")
                    break
                try:
                    perm = _shuffle(n, 1000 + k)
                    top1, w, reached, dt = run_ordering(mps_gpu, qc, top_chi, perm, BEAM, log)
                    results.append((f"shuffle{k}", top1, w))
                    log(f"xcheck {k} (χ={top_chi}): w={w:.4g} agree_primary="
                        f"{top1 == best_bits} ({dt:.1f}s)")
                except Exception as e:
                    log(f"xcheck {k}: FAILED ({type(e).__name__}); skip")
                    break

        # Select by MAJORITY VOTE across independent orderings (robust to a
        # single ordering being an under-converged outlier — max-weight is NOT
        # reliable when weights are close). Tie-break: prefer the primary
        # (most-converged via escalation), else highest summed weight.
        results = [r for r in results if r[1] is not None]
        if results:
            from collections import Counter
            votes = Counter(r[1] for r in results)
            top_count = max(votes.values())
            winners = [b for b, c in votes.items() if c == top_count]
            primary_bits = results[0][1]
            if len(winners) == 1:
                best_bits = winners[0]
            elif primary_bits in winners:
                best_bits = primary_bits
            else:
                best_bits = max(winners,
                                key=lambda b: sum(r[2] for r in results if r[1] == b))
            consensus = votes[best_bits]
            info["consensus"] = f"{consensus}/{len(results)}"
            info["n_distinct"] = len(votes)
            info["selected_w"] = max((r[2] for r in results if r[1] == best_bits),
                                     default=0.0)
            log(f"VERIFY: majority {consensus}/{len(results)} "
                f"({len(votes)} distinct, primary_in_majority="
                f"{primary_bits == best_bits})")
    except Exception as e:
        log(f"Solver error: {type(e).__name__}: {e}")

    status = "success" if best_bits else "failed"
    solution = Solution(status, best_bits)
    log(f"FINAL: status={status} peak={best_bits}")

    result_json = json.dumps(solution.to_dict(), indent=2)
    solve_info_json = json.dumps({
        "challenge_id": challenge_id,
        "timestamp_utc": ts_start,
        "solution_status": status,
        "solve_time_seconds": time.time() - START,
        "difficulty": problem.difficulty,
        **info,
    })

    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir:
        try:
            from pathlib import Path
            Path(output_dir).mkdir(exist_ok=True)
            Path(output_dir, "result.json").write_text(result_json)
            Path(output_dir, "solve_info.json").write_text(solve_info_json)
        except OSError:
            pass

    zip_bytes = build_solution_zip({
        "result.json": result_json,
        "solve_info.json": solve_info_json,
    })
    write_solution_output(zip_bytes)
    os._exit(0 if status == "success" else 1)


if __name__ == "__main__":
    main()
