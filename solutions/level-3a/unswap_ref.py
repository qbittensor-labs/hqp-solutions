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

from circuit_mpo_ref import apply_circuit, apply_swaps, mpo_from_circuit

from utils import iter_layers, merge_layers, elem_counts, merge_gates, get_tn_info

import numpy as np
import time
import os
import math

import logging

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    level=logging.INFO
)

# ------------------------------------------------------------------
#  Rewiring
# ------------------------------------------------------------------
from qiskit.transpiler.passes import ElidePermutations, SabreSwap
from qiskit.transpiler import CouplingMap

def rewire_layers(ls, perm, seed=None):
    nq = len(perm)
    qc = merge_layers(ls)
    qc = QuantumCircuit(nq).compose(qc, qubits=np.argsort(perm))

    qc = ElidePermutations()(qc)
    ss = SabreSwap(coupling_map=CouplingMap.from_line(ls[0].num_qubits), heuristic='decay', trials=10000, seed=seed)
    qc = ss(qc)

    return list(iter_layers(qc))



# ------------------------------------------------------------------
#  Unswapping
# ------------------------------------------------------------------

def get_bond_sizes(mpo: MatrixProductOperator):
    return np.array([mpo.bond_size(ii,ii+1) for ii in range(len(mpo.sites) - 1)])


# ------------------------------------------------------------------
#  DEVICE-ADAPTIVE backend (NOT in the reference; off unless the caller passes an AdaptiveBackend)
#  The absorber spends about half of its wall on operators with bond <= 64, where every step is a few hundred
#  tiny kernels: pure launch/sync latency on the GPU, and it is what several concurrent D workers fight over.
#  MEASURED 2026-09-19: the same steps run ~10x faster on the CPU while the bond is small, and the CPU is
#  hopeless at bond 256-512. So: tensors live on the CPU while max bond <= HQP_DEV_CPU_BOND and move to the GPU
#  once it reaches HQP_DEV_GPU_BOND (hysteresis in between).
# ------------------------------------------------------------------
class AdaptiveBackend:
    def __init__(self, dtype=None, cpu_bond=None, gpu_bond=None, start="cuda"):
        import torch
        self.torch = torch
        self.dtype = dtype or torch.complex64
        self.cpu_bond = int(os.environ.get("HQP_DEV_CPU_BOND", "64")) if cpu_bond is None else cpu_bond
        self.gpu_bond = int(os.environ.get("HQP_DEV_GPU_BOND", "192")) if gpu_bond is None else gpu_bond
        self.dev = start
        self.moves = 0

    def __call__(self, x):
        return self.torch.tensor(x, dtype=self.dtype, device=self.dev)


def _maybe_move(mpo, to_backend):
    if not isinstance(to_backend, AdaptiveBackend) or mpo is None:
        return mpo
    b = int(get_bond_sizes(mpo).max())
    want = "cpu" if b <= to_backend.cpu_bond else ("cuda" if b >= to_backend.gpu_bond else to_backend.dev)
    cur = None
    for t in mpo.tensors:
        cur = "cuda" if getattr(t.data, "is_cuda", False) else "cpu"
        break
    if want != cur:
        mpo.apply_to_arrays(lambda a: a.to(want))
        to_backend.moves += 1
    to_backend.dev = want
    return mpo


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


def unswap(mpo: MatrixProductOperator, hows=("left", "right", "both"), max_bond=2048, cutoff=0.0001, max_its=25, equal=False, to_backend=None, t0=0):
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
        num_improvements = 0
        start_counts = elem_counts(mpo)

        for how in hows:
            for parity in [0, 1]:
                mpo = _maybe_move(mpo, to_backend)
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
    logging.info(f"    [end unswap] -> " + str(get_tn_info(mpo)))

    return mpo, (perm_left, perm_right), stats_data


# ------------------------------------------------------------------
#  Frobenius-norm meter (NOT in the reference; logging only; HQP_NORM_LOG=0 disables)
# ------------------------------------------------------------------
# A unitary's MPO has ||U||_F^2 = 2^n exactly and every exact gate application preserves it, so
# log10(||M||_F^2 / 2^n) is a direct, cheap fidelity meter of the truncation loss so far.
# MEASURED on the cached hybrid cores: d2 (solved, amp2 0.135) = -2.47; d3 (no peak) = -36.4.
_NORM_LOG = os.environ.get("HQP_NORM_LOG", "1") == "1"


def mpo_log10_frob2_ratio(mpo):
    """log10(||M||_F^2 / 2^n) via quimb's own full contraction of <M|M>.

    Layout-agnostic on purpose: between absorption and unswap the MPO tensors are NOT in the neat
    (bond, bond, k_i, b_i) form -- a contracted gate leaves e.g. {k4, b4, b5, bond, bond} on one
    tensor (MEASURED 2026-09-18, the hand-rolled transfer contraction failed on every pre-unswap
    MPO). quimb mangles the bra's inner indices when the two copies are joined, so M.H @ M is the
    Frobenius norm squared. Promoted to complex128 so a heavily truncated core (d3: 1e-22 absolute)
    cannot underflow; a contraction this size costs milliseconds.
    """
    try:
        import torch
        n = len(mpo.sites)
        m = mpo.copy()
        m.apply_to_arrays(lambda x: x.to(torch.complex128) if torch.is_tensor(x)
                          else torch.as_tensor(np.asarray(x)).to(torch.complex128))
        val = m.H @ m
        v = float(abs(val)) if not torch.is_tensor(val) else float(val.abs().item())
        if not (v > 0.0 and math.isfinite(v)):
            return float("nan")
        return math.log10(v) - n * math.log10(2)
    except Exception as e:  # noqa: BLE001  a meter must never take the run down
        logging.info(f"    [norm meter failed: {type(e).__name__}: {e}]")
        return float("nan")


def _norm_log(tag, mpo):
    if _NORM_LOG:
        logging.info(f"    [norm] {tag}: log10(|M|_F^2/2^n) = {mpo_log10_frob2_ratio(mpo):.2f}")


# ------------------------------------------------------------------
#  MPO Cancellation + Unswapping
# ------------------------------------------------------------------

def mpo_compress_unswap(circuit: QuantumCircuit, max_bond=8192, cutoff=0.001, unswap_threshold=1e6, early_stopping_gates=100, center_ratio=0.5, equal=False, flip_freq=None, max_its=20, to_backend=None, seed=None, hows=("both", "left", "right"), mpo_core=None, deadline=None, allow_abandon=True, stall_max=None, adapt_stop=None):
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

    # Rewire layers
    layers_left = rewire_layers(layers_left, np.arange(circuit.num_qubits, dtype=int), seed=seed)
    init_meas = layers_left[-2:]
    layers_left = layers_left[:-2]

    layers_right = rewire_layers(layers_right, np.arange(circuit.num_qubits, dtype=int), seed=seed)
    final_meas = layers_right[-2:]
    layers_right = layers_right[:-2]

    # Start the MPO and counters
    ii_left = 0
    ii_right = 0
    do_left = False
    if mpo_core is None:
        mpo_core = mpo_from_circuit(q2c(QuantumCircuit(circuit.num_qubits)))
    logging.info("[start compressing] -> " + str(get_tn_info(mpo_core)))
    _norm_log("start", mpo_core)


    total_u_consumed = 0
    current_u_consumed = 0
    total_u_consumed_left = 0
    total_u_consumed_right = 0

    stats_data = []

    # --- GUARDS (not in the reference; env-gated, defaults) -------------------------------------
    # The reference outer loop has no exit other than consuming every layer or early_stopping_gates:
    # when unswap() finds nothing to accept, the SAME layers are re-tried forever. MEASURED
    # 2026-09-18 (d2, c128 SVD, early_stopping_gates=0): stuck at 921/924 for 7.5 h, 1309 rounds.
    #   HQP_REF_STALL_MAX (default 12): stop after this many consecutive unswap rounds that consumed
    #     no unitary; the leftover layers are returned exactly as the reference returns them.
    #   deadline: absolute time.time() bound (production kwarg); allow_abandon is accepted for
    #     signature compatibility and ignored.
    if stall_max is None:
        stall_max = int(os.environ.get("HQP_REF_STALL_MAX", "12"))
    stall = 0
    # FRAME FIX (HQP_UNSWAP_FRAME_FIX=0 to disable) -- a bug in the reference: once one side's layers
    # are exhausted its stored measure layer is never rewired again, yet later unswaps keep permuting
    # that side's MPO legs, so the returned readout frame goes silently stale. Harmless for the input
    # frame (|0..0> is permutation-invariant) but it scrambles the OUTPUT frame whenever the right side
    # empties first, i.e. center_ratio != 0.5 -- which every excised block has. MEASURED 2026-09-18:
    # D-measurement draws returning model err 1.000 / unitarity 0.00 (pure garbage) on real and
    # generated blocks. Same fix as our fork (validated on d2/d3): rewire the measure layer anyway.
    frame_fix = os.environ.get("HQP_UNSWAP_FRAME_FIX", "1") == "1"
    # ANTI-LIVELOCK ESCALATION (HQP_REF_ANTILIVE=0 to disable) -- ported from our fork, where it is
    # what let d3_s2's block 1445 absorb at seed 123 (fit 0.053); the plain reference livelocks there
    # (MEASURED 2026-09-18: stall at 197/394 and at 1207 leftover, twice, deterministically). After
    # HQP_REF_STALL_TRIG consecutive no-progress unswap rounds accept dimension-preserving swaps
    # (equal=None, the reference's own randomised tie-break, made deterministic per seed below);
    # one round later grow the element budget x HQP_REF_THRESH_GROW per round up to
    # x HQP_REF_THRESH_CAP and let unswap iterate longer. Progress resets acceptance to strict; the
    # budget stays where it climbed to. The stall guard above remains the hard stop.
    antilive = os.environ.get("HQP_REF_ANTILIVE", "1") == "1"
    stall_trig = int(os.environ.get("HQP_REF_STALL_TRIG", "2"))
    thresh_grow = float(os.environ.get("HQP_REF_THRESH_GROW", "2.0"))
    thresh_cap = float(os.environ.get("HQP_REF_THRESH_CAP", "16.0"))
    cur_threshold = unswap_threshold
    cur_equal = equal
    cur_max_its = max_its
    if antilive and seed is not None:
        np.random.seed(int(seed))      # equal=None draws np.random.rand: pin it to the draw's seed
    # ADAPTIVE STOP + ROLLBACK (HQP_ADAPT_STOP=0 to disable) -- absorb only while the circuit CANCELS.
    # The mirror part of these circuits is bigger than any twin-detectable cut: MEASURED on reduced
    # d3_s1 from the block-1 junction, ~150 unitaries per side cancel cleanly (post-unswap bond 6-16,
    # 10^-0.0023 of norm per unitary) and then the genuinely trained base begins -- from there every
    # absorbed gate costs 10x more norm (10^-0.022 each), the absorber thrashes, and by the end the
    # operator is worthless (the midpoint-centred run ended at 10^-8 and read out 3 bits wrong). The
    # right move at that point is to STOP, keep the last good operator, and leave the rest of the
    # circuit as residual layers on the state, which is how d2 reads out at p=0.21.
    #   collapsed : post-unswap bond <= HQP_ADAPT_GOOD_BOND after >= HQP_ADAPT_MIN_ABSORB unitaries
    #   snapshot  : taken at every such round (operator + remaining rewired layers + frames)
    #   trigger   : log10 norm has fallen HQP_ADAPT_MAX_LOSS below the snapshot's -> restore, stop
    # adapt_stop: None = env default; a D-MEASUREMENT must pass False (it needs the block absorbed completely:
    # a rollback there leaves leftovers = "INCOMPLETE -> not excised"), the operator-level readout passes True.
    adapt = (os.environ.get("HQP_ADAPT_STOP", "1") == "1") if adapt_stop is None else bool(adapt_stop)
    adapt_good_bond = int(os.environ.get("HQP_ADAPT_GOOD_BOND", "16"))
    adapt_min_absorb = int(os.environ.get("HQP_ADAPT_MIN_ABSORB", "60"))
    adapt_max_loss = float(os.environ.get("HQP_ADAPT_MAX_LOSS", "0.4"))
    # ARMING. The trigger must not fire in the EARLY rough phase that precedes every collapse (d2 loses
    # 10^-0.5 there before cancelling 700 more unitaries). MEASURED on h0: one good round at t_u 69, then
    # the early rough phase lost 0.42 and the stop fired at t_u 102, handing 883 unitaries to the state --
    # worse than no stop at all; d2 escaped only because its rough phase came before any snapshot
    # existed. So arm only after a SUSTAINED collapse: at least HQP_ADAPT_ARM_SNAPS good rounds spanning
    # at least HQP_ADAPT_ARM_SPAN absorbed unitaries (reduced d3_s1: 9 good rounds over t_u 71..299).
    adapt_arm_snaps = int(os.environ.get("HQP_ADAPT_ARM_SNAPS", "4"))
    adapt_arm_span = int(os.environ.get("HQP_ADAPT_ARM_SPAN", "80"))
    # SECOND way to arm: no recovery for a long time. MEASURED on h0 (wide cut, junction 0): the ~100-unitary
    # shell cancelled cleanly to t_u 114 (3 good rounds spanning 24 -> never armed by the rule above), then the
    # absorber thrashed for > 50 min (bond 32-64, 50 unitaries, norm -0.11 -> -0.88) with the stop disarmed.
    # A collapse that is not coming back looks exactly like that: HQP_ADAPT_ARM_ROUNDS consecutive rounds since
    # the last good one, none of which returned to bond <= GOOD_BOND.
    adapt_arm_rounds = int(os.environ.get("HQP_ADAPT_ARM_ROUNDS", "10"))
    rounds_since_snap = 0
    snap = None
    n_snaps = 0
    first_snap_tu = None
    rolled_back = False

    # Start loop
    while ii_left < len(layers_left) or ii_right < len(layers_right):
        if deadline is not None and time.time() > deadline:
            logging.info(f"[deadline reached: {T_U - total_u_consumed} unitaries left] -> stop absorbing")
            break
        mpo_core = _maybe_move(mpo_core, to_backend)
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

        # Select the smallest one
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
            stats_data.append({"time": time.perf_counter() - t0, "stage": "absorbing", "absorb_side": "left", 
                                "it_left": ii_left, "it_right": ii_right, "layers_left": len(layers_left), "layers_right": len(layers_right),
                                "u_consumed_total_left": total_u_consumed_left, "u_consumed_total_right": total_u_consumed_right, "u_consumed_total": total_u_consumed,
                                "swap_consumed": new_swaps, "u_consumed": new_us, "u_consumed_after_unswap": current_u_consumed, 
                                **get_tn_info(mpo_core)})
        
        # Unswap if both sides go over the size budget
        else: 
            # guard bookkeeping (not in the reference): UNITARIES consumed since the previous unswap.
            # MEASURED (d2 c128 livelock): every round consumed 3 swap-only layers (u: 0) and then hit
            # the threshold on the same unitary layer, 1237 rounds in a row -- so layers are not
            # progress, unitaries are. Healthy runs show up to 8 consecutive zero-unitary rounds
            # (d3 hybrid: 71 of 190 rounds), hence the default stall_max of 12.
            if current_u_consumed == 0:
                stall += 1
                if antilive:
                    if stall >= stall_trig:
                        cur_equal = None
                    if stall >= stall_trig + 1 and cur_threshold < unswap_threshold * thresh_cap:
                        cur_threshold = min(cur_threshold * thresh_grow, unswap_threshold * thresh_cap)
                        cur_max_its = min(cur_max_its + 5, 40)
                    logging.info(f"[anti-livelock] stall={stall} -> equal={cur_equal} threshold=x{cur_threshold/unswap_threshold:.0f} max_its={cur_max_its}")
            else:
                stall = 0
                cur_equal = equal
            if stall_max > 0 and stall >= stall_max:
                logging.info(f"[stall: {stall} consecutive unswap rounds consumed nothing, {T_U - total_u_consumed} unitaries left] -> stop absorbing")
                break
            # Apply unswapping
            _norm_log(f"pre-unswap (t_u {total_u_consumed}/{T_U})", mpo_core)
            try:
                mpo_core, (new_perm_left, new_perm_right), new_unswap_stats = unswap(mpo_core, hows=hows, max_bond=max_bond, cutoff=cutoff, max_its=cur_max_its, equal=cur_equal, to_backend=to_backend, t0=t0)
                stats_data += new_unswap_stats
            except KeyboardInterrupt:
                break        
            _norm_log(f"post-unswap (t_u {total_u_consumed}/{T_U})", mpo_core)
            # Rewire left circuit
            if ii_left < len(layers_left):
                layers_left = rewire_layers(layers_left[(ii_left):] + init_meas, new_perm_left, seed=seed)
                init_meas = layers_left[-2:]
                layers_left = layers_left[:-2]
            else:
                if frame_fix:
                    init_meas = rewire_layers(init_meas, new_perm_left, seed=seed)[-2:]
                layers_left = []
            
            # Rewire right circuit
            if ii_right < len(layers_right):
                layers_right = rewire_layers(layers_right[(ii_right):] + final_meas, new_perm_right, seed=seed)
                final_meas = layers_right[-2:]
                layers_right = layers_right[:-2]
            else:
                if frame_fix:
                    final_meas = rewire_layers(final_meas, new_perm_right, seed=seed)[-2:]
                layers_right = []
            
            ii_left = 0
            ii_right = 0
            current_u_consumed = 0

            if adapt:
                _bond_now = int(get_bond_sizes(mpo_core).max())
                _norm_now = mpo_log10_frob2_ratio(mpo_core)
                if _bond_now <= adapt_good_bond and total_u_consumed >= adapt_min_absorb and _norm_now == _norm_now:
                    snap = {"mpo": mpo_core.copy(), "ll": list(layers_left), "lr": list(layers_right),
                            "im": list(init_meas), "fm": list(final_meas), "norm": _norm_now,
                            "tu": total_u_consumed, "tul": total_u_consumed_left, "tur": total_u_consumed_right}
                    n_snaps += 1
                    rounds_since_snap = 0
                    if first_snap_tu is None:
                        first_snap_tu = total_u_consumed
                elif snap is not None:
                    rounds_since_snap += 1
                if (snap is not None and rounds_since_snap > 0
                        and ((n_snaps >= adapt_arm_snaps and snap["tu"] - first_snap_tu >= adapt_arm_span)
                             or rounds_since_snap >= adapt_arm_rounds)
                        and _norm_now == _norm_now and _norm_now < snap["norm"] - adapt_max_loss):
                    logging.info(f"[adaptive stop] cancellation ended: log10 norm {_norm_now:.2f} is {snap['norm'] - _norm_now:.2f} below "
                                 f"the last good operator ({snap['norm']:.2f} at t_u {snap['tu']}, bond <= {adapt_good_bond}); "
                                 f"rolling back {total_u_consumed - snap['tu']} unitaries and handing "
                                 f"{T_U - snap['tu']} to the state")
                    mpo_core = snap["mpo"]; layers_left = snap["ll"]; layers_right = snap["lr"]
                    init_meas = snap["im"]; final_meas = snap["fm"]
                    total_u_consumed = snap["tu"]; total_u_consumed_left = snap["tul"]; total_u_consumed_right = snap["tur"]
                    rolled_back = True
                    break

            # Stop early if there are few gates left
            if (T_U - total_u_consumed) <= early_stopping_gates:
                break
    
    # Remove any leftover layers
    layers_left = layers_left[(ii_left):] if ii_left < len(layers_left) else []
    layers_left += init_meas
    layers_right = layers_right[(ii_right):] if ii_right < len(layers_right) else []
    layers_right += final_meas

    logging.info(f"[end compressing](left: {len(layers_left)}, right: {len(layers_right)}) -> " + str(get_tn_info(mpo_core)))
    _norm_log("end", mpo_core)
    stats_data.append({"stage": "final", "u_consumed_final": total_u_consumed, "rolled_back": rolled_back,
                       "u_left_final": total_u_consumed_left, "u_right_final": total_u_consumed_right})

    return mpo_core, layers_left, layers_right, stats_data


def mpo_to_mps(mpo_core, layers_left, layers_right, max_bond=4096, cutoff=0.001, to_backend=None):
    q2c = lambda qc: quimb_circuit(qc.decompose("unitary"), Circuit, to_backend=to_backend)
    # Use the compressed MPO to get the MPS by applying it to |0> state
    final_mps = quimb_circuit(
        QuantumCircuit(len(mpo_core.sites)),
        quimb_circuit_class=CircuitMPS,
        to_backend=to_backend,
    ).psi

    # First take the leftover front layers
    _meas_l = [l for l in layers_left if {"measure","barrier"} & set(dict(l.count_ops()))]
    _gate_l = [l for l in layers_left if l not in _meas_l]
    layers_left = list(iter_layers(merge_layers(_gate_l).inverse())) if len(_gate_l) > 0 else []
    
    for ii_left in range(len(layers_left)):
        l_left = layers_left[ii_left]
        new_ops = dict(l_left.count_ops())
        layer_mpo = mpo_from_circuit(q2c(l_left))
        final_mps = layer_mpo.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff)
        logging.info(f"[Left {ii_left} / {len(layers_left)}] -> " + str(get_tn_info(final_mps)))

    logging.info("[Left MPS] -> " + str(get_tn_info(final_mps)))

    # Then apply the compressed MPO to the layers
    final_mps = mpo_core.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff)
    logging.info("[Left MPS + Core MPO] -> " + str(get_tn_info(final_mps)))

    # Then iterate through final layers if there are any
    final_meas = []
    for ii_right in range(len(layers_right)):
        l_right = layers_right[ii_right]
        new_ops = dict(l_right.count_ops())
        if "barrier" in new_ops or "measure" in new_ops:
            final_meas.append(l_right)
        else:
            layer_mpo = mpo_from_circuit(q2c(l_right))
            final_mps = layer_mpo.apply(final_mps, compress=True, max_bond=max_bond, cutoff=cutoff)
            logging.info(f"[Front MPS + Core MPO + Right {ii_right} / {len(layers_right)}] -> " + str(get_tn_info(final_mps)))
    
    logging.info(f"[Front MPS + Core MPO + Right MPS] -> " + str(get_tn_info(final_mps)))

    # Extract final permutation from measurements
    final_perm = [g.qubits[0]._index for g in final_meas[-1]]

    # Return MPS and final perm
    return final_mps, final_perm

