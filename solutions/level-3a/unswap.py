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

from quimb.tensor import MatrixProductOperator, Circuit, CircuitMPS

from qiskit_quimb import quimb_circuit
from qiskit import QuantumCircuit

from circuit_mpo import apply_circuit, apply_swaps, mpo_from_circuit

from utils import iter_layers, merge_layers, elem_counts, merge_gates, get_tn_info

import numpy as np
import time

import logging

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    level=logging.INFO
)

# ------------------------------------------------------------------
#  Rewiring
# ------------------------------------------------------------------
import os as _os
from qiskit.transpiler.passes import ElidePermutations, SabreSwap
from qiskit.transpiler import CouplingMap

# Reference uses trials=10000 (pcs/unswap.py:33); we defaulted to 200. Worse still it was read at
# MODULE scope, so hardening_quantum_proof.py's `from unswap import ...` (line 410) ran BEFORE its
# os.environ["HQP_SABRE_TRIALS"]=... (line 411) and froze the value at 200 whatever the env said.
# Read it per call instead, and default to the reference's value: routing quality decides how many
# transpiler SWAPs exist, and those SWAPs are the only thing unswapping can actually cancel.
def _sabre_trials():
    return int(_os.environ.get("HQP_SABRE_TRIALS", "10000"))
_FRAME_FIX = _os.environ.get("HQP_UNSWAP_FRAME_FIX", "1") == "1"

# TRUTH METER for truncation damage. Every gate absorbed into the core is UNITARY, and a unitary
# preserves ||W||_F exactly -- so any decrease in ||MPO||_F^2 is PURE truncation loss, measured
# rather than modelled. This settles the question the error model could not: whether the loss is
# many small truncations (fix = fewer of them) or few bad ones (fix = better gauge per truncation).
# Costs one boundary sweep, negligible while the core bond stays ~50.
_NORM_TRACK = _os.environ.get("HQP_NORM_TRACK", "0") == "1"
_NORM_EVERY = int(_os.environ.get("HQP_NORM_EVERY", "25"))

# IDEA #5 (anti-livelock escalation) + #6 (smart initial ordering) — env-gated so the
# validated baseline is unchanged when off.
_ANTILIVE   = _os.environ.get("HQP_US_ANTILIVELOCK", "0") == "1"   # escalate when absorption stalls
_STALL_TRIG = int(_os.environ.get("HQP_US_STALL_TRIGGER", "2"))    # consecutive unswaps w/o progress before escalating
_THRESH_GROW = float(_os.environ.get("HQP_US_THRESH_GROW", "2.0")) # grow unswap_threshold per escalation (push through the wall)
_THRESH_CAP  = float(_os.environ.get("HQP_US_THRESH_CAP", "16.0")) # cap on threshold multiplier
_SMART_ORDER = _os.environ.get("HQP_US_SMART_ORDER", "0") == "1"   # spectral initial qubit ordering
_FAITHFUL = _os.environ.get("HQP_FAITHFUL", "0") == "1"            # #13 faithful (variational) final extraction
_FAITHFUL_ITERS = int(_os.environ.get("HQP_FAITHFUL_ITERS", "8"))

# EARLY ABANDONMENT of a thrashing ordering: when an ordering absorbs < MIN units over WINDOW
# wall-seconds (even after #5 escalation), bail so the orchestrator tries the NEXT ordering — a
# different ordering is far likelier to help than infinite escalation. Env-gated; OFF by default.
_EARLY_ABANDON  = _os.environ.get("HQP_US_EARLY_ABANDON", "0") == "1"
_ABANDON_MIN    = int(_os.environ.get("HQP_ABANDON_MIN", "8"))        # units that count as 'progress'
_ABANDON_WINDOW = float(_os.environ.get("HQP_ABANDON_WINDOW", "600")) # wall-seconds with < MIN absorbed -> abandon


class OrderingAbandoned(Exception):
    """Thrash detected: this ordering absorbed < HQP_ABANDON_MIN units over HQP_ABANDON_WINDOW
    wall-seconds even after #5 escalation. _solve_unswap catches this and moves to the next seed."""
    def __init__(self, consumed=0, total=0, window=0.0, reason=None):
        self.consumed, self.total, self.window, self.reason = consumed, total, window, reason
        super().__init__(reason if reason is not None else
                         f"ordering abandoned: {consumed}/{total} units in {window:.0f}s window")


def _apply_mpo_to_mps(mpo, mps, max_bond, cutoff):
    """IDEA #13: the FINAL MPO->MPS applications (mpo_to_mps) are what DIRECTLY produce the peak
    amplitude. Faithful mode applies the MPO via the variational fit-zipup gate (gauged + refine
    -> re-allocates kept bond to the globally peak-carrying weight) instead of the greedy default
    compress that drifts the peak to a wrong attractor. Falls back to mpo.apply on any error."""
    out = None
    if _FAITHFUL:
        try:
            out = mps.gate_with_mpo(mpo, method="fit-zipup", max_bond=max_bond,
                                    cutoff=cutoff, max_iterations=_FAITHFUL_ITERS)
        except Exception as e:
            logging.info(f"    [#13 faithful extract failed ({type(e).__name__}: {str(e)[:80]}); fallback to apply]")
    if out is None:
        out = mpo.apply(mps, compress=True, max_bond=max_bond, cutoff=cutoff)
    # Scrub NaN/inf: a single truncation edge-case (zero-norm bond, rank-deficient SVD) otherwise
    # poisons the whole state -> beam sees w=nan and the run fail-closes, discarding a real (weak)
    # peak. Relative amplitudes are what the beam needs, so zeroing non-finite entries is safe;
    # normalization is irrelevant to argmax.
    try:
        import torch as _t

        def _scrub(x):
            return _t.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) if _t.is_tensor(x) else x
        out.apply_to_arrays(_scrub)
    except Exception:
        pass
    return out


def good_initial_ordering(circuit):
    """IDEA #6: a spectral (Fiedler) linear ordering of qubits by the 2-qubit gate coupling
    graph. Keeps strongly-coupled qubits adjacent on the line -> lower entanglement across
    cuts -> the unswap is far more likely to cancel the mirror structure instead of livelocking.
    Returns an int permutation; falls back to identity on any failure (correctness-safe)."""
    n = circuit.num_qubits
    try:
        idx = {q: i for i, q in enumerate(circuit.qubits)}
        W = np.zeros((n, n))
        for inst in circuit.data:
            qs = [idx[q] for q in inst.qubits]
            if len(qs) == 2:
                a, b = qs
                W[a, b] += 1.0
                W[b, a] += 1.0
        if W.sum() == 0:
            return np.arange(n, dtype=int)
        L = np.diag(W.sum(1)) - W                  # graph Laplacian
        _, vecs = np.linalg.eigh(L)
        order = np.argsort(vecs[:, 1]).astype(int) # Fiedler vector -> linear arrangement
        logging.info(f"[smart-order] spectral initial ordering applied (n={n})")
        return order
    except Exception as e:
        logging.info(f"[smart-order] failed ({type(e).__name__}: {e}); identity")
        return np.arange(n, dtype=int)


def rewire_layers(ls, perm, seed=None):
    nq = len(perm)
    qc = merge_layers(ls)
    qc = QuantumCircuit(nq).compose(qc, qubits=np.argsort(perm))

    qc = ElidePermutations()(qc)
    ss = SabreSwap(coupling_map=CouplingMap.from_line(ls[0].num_qubits), heuristic='decay', trials=_sabre_trials(), seed=seed)
    qc = ss(qc)

    return list(iter_layers(qc))



# ------------------------------------------------------------------
#  Unswapping
# ------------------------------------------------------------------

def get_bond_sizes(mpo: MatrixProductOperator):
    return np.array([mpo.bond_size(ii,ii+1) for ii in range(len(mpo.sites) - 1)])


def swap_perm(perm, swaps):
    for q0, q1 in swaps:
        (perm[q0], perm[q1]) = (perm[q1], perm[q0])
    return perm


def get_good_swaps(mpo, qubit_pairs, how, max_bond, cutoff, to_backend=None, equal=False):
    current_bonds = get_bond_sizes(mpo)
    #log_print("    [debug](select)(bond sizes before) -> ", current_bonds.tolist())

    swaps_l = qubit_pairs if how in ("left", "both") else []
    swaps_r = qubit_pairs if how in ("right", "both") else []

    mpo_tmp = apply_swaps(mpo, swaps_l=swaps_l, swaps_r=swaps_r, max_bond=max_bond, cutoff=cutoff, to_backend=to_backend)
    new_bonds = get_bond_sizes(mpo_tmp)
    if equal is None:
        new_bonds = new_bonds + (np.random.rand(*new_bonds.shape)-0.5)
        improved = np.nonzero(new_bonds < current_bonds)[0]
    elif equal:
        improved = np.nonzero(new_bonds <= current_bonds)[0]
    else:
        improved = np.nonzero(new_bonds < current_bonds)[0]

    return improved


# ---------------------------------------------------------------------------------------------
# Algorithm II.2 of arXiv:2604.21908 (the published method that breaks HQAP circuits), faithfully.
#
# The batch `unswap()` below is NOT that algorithm: get_good_swaps() applies a swap to EVERY pair
# in a parity class at once, measures the bonds of that single combined configuration, and then
# re-applies whichever SUBSET appeared to improve -- a configuration that was never evaluated. The
# per-swap gains do not compose, so swaps fight each other and the MPO thrashes instead of
# shrinking (measured: "swap treadmill", u_consumed stuck at 0 on d3).
#
# The paper instead works ONE bond at a time:
#   A <- {0..N-2}
#   while A:  i <- argmax_{j in A} bond_dim(j)          # largest bond first
#             try SWAP from left / right / both on that bond, compress each
#             if the best candidate REDUCES bond i: accept it, A <- (A - {i}) + neighbours(i)
#             else:                                    A <- A - {i}
# Each acceptance strictly reduces a bond dimension, which is bounded below, so it terminates.
# ---------------------------------------------------------------------------------------------
UNSWAP_ALGO = _os.environ.get("HQP_UNSWAP_ALGO", "batch").strip().lower()
UNSWAP_MAX_ACCEPT = int(_os.environ.get("HQP_UNSWAP_MAX_ACCEPT", "4000"))   # safety cap (relaxed mode)


def unswap_greedy(mpo: MatrixProductOperator, hows=("left", "right", "both"), max_bond=2048,
                  cutoff=1e-4, max_its=None, equal=False, to_backend=None, t0=0, deadline=None):
    n = len(mpo.sites)
    perm_left = list(range(n))
    perm_right = list(range(n))
    avail = set(range(n - 1))
    stats_data = []
    accepted = probes = 0
    logging.info("    [start unswap:greedy] -> " + str(get_tn_info(mpo)))
    while avail and accepted < UNSWAP_MAX_ACCEPT:
        if deadline is not None and time.time() > deadline:
            logging.info("    [unswap:greedy deadline reached] -> stop")
            break
        bonds = get_bond_sizes(mpo)
        i = max(avail, key=lambda j: bonds[j])          # largest available bond first
        cur = bonds[i]
        best = None
        for how in hows:
            if deadline is not None and time.time() > deadline:
                break
            sl = [(i, i + 1)] if how in ("left", "both") else []
            sr = [(i, i + 1)] if how in ("right", "both") else []
            try:
                cand = apply_swaps(mpo, swaps_l=sl, swaps_r=sr, max_bond=max_bond,
                                   cutoff=cutoff, to_backend=to_backend)
            except Exception as e:                       # a failed probe must not kill the sweep
                logging.info(f"    [unswap:greedy probe {how} @bond {i} failed: {type(e).__name__}]")
                continue
            probes += 1
            b = get_bond_sizes(cand)[i]
            if best is None or b < best[0]:
                best = (b, how, cand)
        take = best is not None and (best[0] <= cur if equal else best[0] < cur)
        if take:
            _, how, mpo = best
            if how in ("left", "both"):
                perm_left = swap_perm(perm_left, [(i, i + 1)])
            if how in ("right", "both"):
                perm_right = swap_perm(perm_right, [(i, i + 1)])
            accepted += 1
            avail.discard(i)
            if i - 1 >= 0:
                avail.add(i - 1)                          # the accepted swap changed the local
            if i + 1 <= n - 2:
                avail.add(i + 1)                          # structure: re-enable neighbours
            stats_data.append({"time": time.perf_counter() - t0, "stage": "unswapping",
                               "bond": int(i), "side": how, "from": int(cur), "to": int(best[0]),
                               **get_tn_info(mpo)})
        else:
            avail.discard(i)
    logging.info(f"    [end unswap:greedy] accepted={accepted} probes={probes} -> " + str(get_tn_info(mpo)))
    return mpo, (perm_left, perm_right), stats_data


def unswap(mpo: MatrixProductOperator, hows=("left", "right", "both"), max_bond=2048, cutoff=0.0001, max_its=25, equal=False, to_backend=None, t0=0, deadline=None):
    num_qubits = len(mpo.sites)
    all_pairs = [(i, i+1) for i in range(num_qubits-1)]

    perm_left = list(range(len(mpo.sites)))
    perm_right = list(range(len(mpo.sites)))

    logging.info("    [start unswap] -> " + str(get_tn_info(mpo)))
    num_improvements = 1
    start_counts = 1
    end_counts = 0
    ii = 0

    stats_data = []
    while num_improvements > 0 and ii < max_its and start_counts != end_counts:
        if deadline is not None and time.time() > deadline:
            logging.info("    [unswap deadline reached] -> stop")
            break
        num_improvements = 0
        start_counts = elem_counts(mpo)

        _bail = False
        for how in hows:
            if _bail:
                break
            for parity in [0, 1]:
                # PER-PROBE deadline check. Checking only once per outer iteration means a
                # single long iteration (6 probes, each a full MPO sweep) can overrun the
                # budget without the guard ever being evaluated -- a bound that silently
                # stops bounding.
                if deadline is not None and time.time() > deadline:
                    logging.info(f"    [unswap deadline reached MID-ITERATION at it={ii}, "
                                 f"how={how}, parity={parity}] -> stop")
                    _bail = True
                    break
                # Estimate which qubit pairs to swap
                new_swap_ids = get_good_swaps(mpo, qubit_pairs=all_pairs[parity::2], how=how, max_bond=max_bond, cutoff=cutoff, to_backend=to_backend, equal=equal)
                new_swaps = [all_pairs[i] for i in new_swap_ids if i % 2 == parity]

                # Apply the selected swaps
                swaps_l = new_swaps if how in ("left", "both") else []
                swaps_r = new_swaps if how in ("right", "both") else []
                mpo = apply_swaps(mpo, swaps_l=swaps_l, swaps_r=swaps_r, max_bond=max_bond, cutoff=cutoff, to_backend=to_backend)

                # Update the permutations
                if how in ("left", "both"):
                    perm_left = swap_perm(perm_left, new_swaps)
                if how in ("right", "both"):
                    perm_right = swap_perm(perm_right, new_swaps)
    
                # Track how many new swaps were applied
                num_improvements += len(new_swap_ids)
                stats_data.append({"time": time.perf_counter()-t0, "stage": "unswapping", "iteration": ii, "side": how, "parity": parity, "new_swaps": len(new_swap_ids), "total_swaps": num_improvements, **get_tn_info(mpo)})
                logging.info(f"    [{ii} | {how} | {parity}](new_swaps: {len(new_swap_ids)} | total: {num_improvements}) -> " + str(get_tn_info(mpo)))

        end_counts = elem_counts(mpo)
        ii += 1
        if _bail:
            break
    logging.info(f"    [end unswap] -> " + str(get_tn_info(mpo)))

    return mpo, (perm_left, perm_right), stats_data


# ------------------------------------------------------------------
#  MPO Cancellation + Unswapping
# ------------------------------------------------------------------

def mpo_compress_unswap(circuit: QuantumCircuit, max_bond=8192, cutoff=0.001, unswap_threshold=1e6, early_stopping_gates=100, center_ratio=0.5, equal=False, flip_freq=None, max_its=20, to_backend=None, seed=None, hows=("both", "left", "right"), mpo_core=None, deadline=None, allow_abandon=True):
    q2c = lambda qc: quimb_circuit(qc.decompose("unitary"), Circuit, to_backend=to_backend)
    t0 = time.perf_counter()

    # Split circuit into left and right
    if type(center_ratio) is float:
        C = int(len(circuit) * center_ratio)
    elif type(center_ratio) is int:
        C = center_ratio
    circuit_left = merge_gates(circuit[:C], circuit.num_qubits).inverse()
    circuit_right = merge_gates(circuit[C:], circuit.num_qubits)
    if "measure" not in circuit_right.count_ops():
        circuit_right.measure_all()
    if "measure" not in circuit_left.count_ops():
        circuit_left.measure_all()

    layers_left = list(iter_layers(circuit_left))
    layers_right = list(iter_layers(circuit_right))


    T_U = circuit.count_ops().get("unitary", 0)
    T_UL = circuit_left.count_ops().get("unitary", 0)
    T_UR = circuit_right.count_ops().get("unitary", 0)

    logging.info(f"Total unitaries: {T_U} = {T_UL} (left) + {T_UR} (right)")

    # Rewire layers (IDEA #6: spectral initial ordering instead of identity, when enabled)
    init_perm = good_initial_ordering(circuit) if _SMART_ORDER else np.arange(circuit.num_qubits, dtype=int)
    layers_left = rewire_layers(layers_left, init_perm, seed=seed)
    init_meas = layers_left[-2:]
    layers_left = layers_left[:-2]

    layers_right = rewire_layers(layers_right, init_perm, seed=seed)
    final_meas = layers_right[-2:]
    layers_right = layers_right[:-2]

    # Start the MPO and counters
    ii_left = 0
    ii_right = 0
    do_left = False
    if mpo_core is None:
        mpo_core = mpo_from_circuit(q2c(QuantumCircuit(circuit.num_qubits)))
    logging.info("[start compressing] -> " + str(get_tn_info(mpo_core)))


    total_u_consumed = 0
    current_u_consumed = 0
    total_u_consumed_left = 0
    total_u_consumed_right = 0

    stats_data = []

    # IDEA #5 anti-livelock escalation: when absorption stalls (the loop keeps re-unswapping
    # without absorbing — the d3 wall), escalate to relaxed acceptance (sideways swaps to escape
    # the local minimum) and grow the size budget to push the entangled region through.
    cur_threshold = unswap_threshold
    cur_equal = equal
    cur_max_its = max_its
    stall = 0
    last_tu_at_unswap = -1
    # EARLY-ABANDON window trackers: high-water mark of absorbed units + wall-clock anchor.
    abandon_hwm = total_u_consumed
    abandon_window_t0 = time.time()

    _fro0 = None
    _nsteps = 0
    if _NORM_TRACK:
        try:
            _fro0 = float(mpo_core.norm()) ** 2
            logging.info(f"[normtrack] baseline ||MPO||_F^2 = {_fro0:.6e}")
        except Exception as _e:
            logging.info(f"[normtrack] baseline unavailable ({type(_e).__name__}) -- disabled")

    # Start loop
    while ii_left < len(layers_left) or ii_right < len(layers_right):
        if deadline is not None and time.time() > deadline:
            logging.info(f"[deadline reached: {T_U - total_u_consumed} unitaries left] -> stop absorbing, extract best-so-far")
            break
        # Try both sides to see which one results in a smaller size
        if ii_left < len(layers_left):
            try:
                mpo_left = apply_circuit(mpo_core, q2c(layers_left[ii_left].inverse()), side="right", max_bond=max_bond, cutoff=cutoff)
            except KeyboardInterrupt:
                break
            counts_left = elem_counts(mpo_left)
        else:
            mpo_left = None
            counts_left = 1e20

        if ii_right < len(layers_right):
            try:
                mpo_right = apply_circuit(mpo_core, q2c(layers_right[ii_right]), side="left", max_bond=max_bond, cutoff=cutoff)
            except KeyboardInterrupt:
                break
            counts_right = elem_counts(mpo_right)
        else:
            mpo_right = None
            counts_right = 1e20
        
        if flip_freq is None:
            do_left = counts_left < counts_right
        else:
            if mpo_left is None:
                do_left = False
            elif mpo_right is None:
                do_left = True
            elif (ii_right + ii_left) % flip_freq == 0:
                do_left = not do_left

        # Select the smallest one (cur_threshold grows under anti-livelock escalation)
        if [counts_right, counts_left][int(do_left)] < cur_threshold:
            if do_left:
                mpo_core = mpo_left
                # Update counts
                new_ops = dict(layers_left[ii_left].count_ops())
                new_us = new_ops.get('unitary', 0)
                new_swaps = new_ops.get('swap', 0)
                total_u_consumed += new_us
                current_u_consumed += new_us
                total_u_consumed_left += new_us

                # Log
                side_chosen = "L"
                ii_left += 1
            else:
                mpo_core = mpo_right
                # Update counts
                new_ops = dict(layers_right[ii_right].count_ops())
                new_us = new_ops.get('unitary', 0)  
                new_swaps = new_ops.get('swap', 0)
                total_u_consumed += new_us
                current_u_consumed += new_us
                total_u_consumed_right += new_us
            
                # Log
                side_chosen = "R"
                ii_right += 1            
            
            logging.info((f"[{ii_right}R/{len(layers_right)}]" if side_chosen == "R" else f"[{ii_left}L/{len(layers_left)}]") + 
                         f"(swap: {new_swaps}, u: {new_us} | c_u: {current_u_consumed} | t_u_l: {total_u_consumed_left}/{T_UL} | t_u_r: {total_u_consumed_right}/{T_UR} | t_u: {total_u_consumed}/{T_U}) -> " +
                         str(get_tn_info(mpo_core)))
            _nsteps += 1
            if _fro0 is not None and (_nsteps % _NORM_EVERY == 0):
                try:
                    _fro = float(mpo_core.norm()) ** 2
                    _keep = _fro / _fro0
                    _per = (1.0 - _keep ** (1.0 / max(_nsteps, 1)))
                    logging.info(f"[normtrack] step={_nsteps} t_u={total_u_consumed} "
                                 f"keep={_keep:.6e} log10_keep={np.log10(max(_keep,1e-300)):.3f} "
                                 f"mean_discard_per_absorb={_per:.3e}")
                except Exception as _e:
                    # Never swallow this: a silent `pass` here is why the meter looked dead on a
                    # run where the baseline had printed fine. mpo.norm() contracts the MPO with
                    # its conjugate and can OOM or raise once the bond grows.
                    logging.info(f"[normtrack] DISABLED at step={_nsteps}: "
                                 f"{type(_e).__name__}: {_e}")
                    _fro0 = None
            stats_data.append({"time": time.perf_counter() - t0, "stage": "absorbing", "absorb_side": "left", 
                                "it_left": ii_left, "it_right": ii_right, "layers_left": len(layers_left), "layers_right": len(layers_right),
                                "u_consumed_total_left": total_u_consumed_left, "u_consumed_total_right": total_u_consumed_right, "u_consumed_total": total_u_consumed,
                                "swap_consumed": new_swaps, "u_consumed": new_us, "u_consumed_after_unswap": current_u_consumed, 
                                **get_tn_info(mpo_core)})
        
        # Unswap if both sides go over the size budget
        else:
            # IDEA #5: detect a livelock (no absorption since the last unswap) and escalate.
            if _ANTILIVE:
                if total_u_consumed <= last_tu_at_unswap:
                    stall += 1
                    if stall >= _STALL_TRIG:
                        cur_equal = None  # relaxed: accept dimension-preserving swaps to escape the local minimum
                    if stall >= _STALL_TRIG + 1 and cur_threshold < unswap_threshold * _THRESH_CAP:
                        cur_threshold = min(cur_threshold * _THRESH_GROW, unswap_threshold * _THRESH_CAP)
                        cur_max_its = min(cur_max_its + 5, 40)
                    logging.info(f"[anti-livelock] stall={stall} -> equal={cur_equal} threshold=x{cur_threshold/unswap_threshold:.0f} max_its={cur_max_its}")
                else:
                    stall = 0
                    cur_equal = equal  # progressing again -> back to strict
                last_tu_at_unswap = total_u_consumed
            # EARLY ABANDONMENT: evaluate AFTER #5 escalation so anti-livelock gets first try.
            if _EARLY_ABANDON and allow_abandon:
                now = time.time()
                if (total_u_consumed - abandon_hwm) >= _ABANDON_MIN:
                    abandon_hwm = total_u_consumed          # real progress -> reset the window
                    abandon_window_t0 = now
                elif (now - abandon_window_t0) >= _ABANDON_WINDOW:
                    logging.info(f"[early-abandon] only {total_u_consumed - abandon_hwm} units in "
                                 f"{now - abandon_window_t0:.0f}s (< {_ABANDON_MIN}); abandon ordering")
                    raise OrderingAbandoned(consumed=total_u_consumed, total=T_U, window=now - abandon_window_t0)
            # Apply unswapping
            try:
                _unswap_fn = unswap_greedy if UNSWAP_ALGO == "greedy" else unswap
                mpo_core, (new_perm_left, new_perm_right), new_unswap_stats = _unswap_fn(mpo_core, hows=hows, max_bond=max_bond, cutoff=cutoff, max_its=cur_max_its, equal=cur_equal, to_backend=to_backend, t0=t0, deadline=deadline)
                stats_data += new_unswap_stats
            except KeyboardInterrupt:
                break        
            # Rewire left circuit
            if ii_left < len(layers_left):
                layers_left = rewire_layers(layers_left[(ii_left):] + init_meas, new_perm_left, seed=seed)
                init_meas = layers_left[-2:]
                layers_left = layers_left[:-2]
            else:
                # FRAME FIX (HQP_UNSWAP_FRAME_FIX=0 to disable): once this side's layers are
                # exhausted, later unswaps still permute its MPO legs. Without rewiring the stored
                # measure layer the returned frame goes silently stale (wrong readout permutation
                # whenever the right side empties first, e.g. center_ratio > 0.5).
                if _FRAME_FIX:
                    init_meas = rewire_layers(init_meas, new_perm_left, seed=seed)[-2:]
                layers_left = []
            
            # Rewire right circuit
            if ii_right < len(layers_right):
                layers_right = rewire_layers(layers_right[(ii_right):] + final_meas, new_perm_right, seed=seed)
                final_meas = layers_right[-2:]
                layers_right = layers_right[:-2]
            else:
                if _FRAME_FIX:
                    final_meas = rewire_layers(final_meas, new_perm_right, seed=seed)[-2:]
                layers_right = []
            
            ii_left = 0
            ii_right = 0
            current_u_consumed = 0

            # Stop early if there are few gates left
            if (T_U - total_u_consumed) <= early_stopping_gates:
                break
    
    # Remove any leftover layers
    layers_left = layers_left[(ii_left):] if ii_left < len(layers_left) else []
    layers_left += init_meas
    layers_right = layers_right[(ii_right):] if ii_right < len(layers_right) else []
    layers_right += final_meas

    logging.info(f"[end compressing](left: {len(layers_left)}, right: {len(layers_right)}) -> " + str(get_tn_info(mpo_core)))

    return mpo_core, layers_left, layers_right, stats_data


class ExtractionDeadline(RuntimeError):
    """Raised when mpo_to_mps runs out of wall time. Callers must treat the extraction as
    unfinished rather than letting it overrun the validator's hard kill."""


def mpo_to_mps(mpo_core, layers_left, layers_right, max_bond=4096, cutoff=0.001, to_backend=None,
               deadline=None):
    """Apply the compressed MPO (plus leftover layers) to |0> and read off the final permutation.

    `deadline` (absolute time.time()) bounds the work: every layer application checks the clock, so
    a slow extraction stops instead of running past the wall. Without it this loop is unbounded --
    it was the one un-deadlined stage left in the solver.
    """
    def _tick(where):
        if deadline is not None and time.time() > deadline:
            raise ExtractionDeadline(f"extraction deadline reached at {where}")
    q2c = lambda qc: quimb_circuit(qc.decompose("unitary"), Circuit, to_backend=to_backend)
    # Use the compressed MPO to get the MPS by applying it to |0> state
    final_mps = quimb_circuit(
        QuantumCircuit(len(mpo_core.sites)),
        quimb_circuit_class=CircuitMPS,
        to_backend=to_backend,
    ).psi

    # First take the leftover front layers
    layers_left = list(iter_layers(merge_layers(layers_left).inverse())) if len(layers_left) > 0 else []
    
    for ii_left in range(len(layers_left)):
        l_left = layers_left[ii_left]
        new_ops = dict(l_left.count_ops())
        _tick(f"left layer {ii_left}/{len(layers_left)}")
        layer_mpo = mpo_from_circuit(q2c(l_left))
        final_mps = _apply_mpo_to_mps(layer_mpo, final_mps, max_bond, cutoff)  # #13 faithful
        logging.info(f"[Left {ii_left} / {len(layers_left)}] -> " + str(get_tn_info(final_mps)))

    logging.info("[Left MPS] -> " + str(get_tn_info(final_mps)))

    # Then apply the compressed MPO to the layers
    _tick("core MPO")
    final_mps = _apply_mpo_to_mps(mpo_core, final_mps, max_bond, cutoff)  # #13 faithful
    logging.info("[Left MPS + Core MPO] -> " + str(get_tn_info(final_mps)))

    # Then iterate through final layers if there are any
    final_meas = []
    for ii_right in range(len(layers_right)):
        l_right = layers_right[ii_right]
        new_ops = dict(l_right.count_ops())
        if "barrier" in new_ops or "measure" in new_ops:
            final_meas.append(l_right)
        else:
            _tick(f"right layer {ii_right}/{len(layers_right)}")
            layer_mpo = mpo_from_circuit(q2c(l_right))
            final_mps = _apply_mpo_to_mps(layer_mpo, final_mps, max_bond, cutoff)  # #13 faithful
            logging.info(f"[Front MPS + Core MPO + Right {ii_right} / {len(layers_right)}] -> " + str(get_tn_info(final_mps)))
    
    logging.info(f"[Front MPS + Core MPO + Right MPS] -> " + str(get_tn_info(final_mps)))

    # Extract final permutation from measurements
    final_perm = [g.qubits[0]._index for g in final_meas[-1]]

    # Return MPS and final perm
    return final_mps, final_perm

