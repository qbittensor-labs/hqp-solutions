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

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import ElidePermutations, SabreSwap
from ..frames.state import FrameState
from . import telemetry
from .layers import elem_counts, get_tn_info, iter_layers, merge_layers
from .mpo import apply_swaps

@dataclass(frozen=True)
class UnswapConfig:
    max_bond: int = 2048
    cutoff: float = 0.0001
    apply_cutoff: float | None = None
    probe_cutoff: float | None = None
    max_its: int = 25
    hows: tuple[str, ...] = ('both', 'left', 'right')
    equal: bool = False
    align_weight: float = 0.0
    align_protect: float | None = None
    sabre_trials: int = 200
    compress_method: str = 'zipup'
    equalize_norms: bool = False
    cycle_seconds: float | None = None

@dataclass
class UnswapCycleResult:
    perm_left: list[int]
    perm_right: list[int]
    iterations: int = 0
    swaps_applied: int = 0
    probe_batches: int = 0
    deadline_hit: bool = False
    cycle_budget_hit: bool = False
    stats: list[dict[str, Any]] = field(default_factory=list)

def rewire_layers(layers, perm, seed=None, sabre_trials: int=200):
    num_qubits = len(perm)
    circuit = merge_layers(layers)
    circuit = QuantumCircuit(num_qubits, circuit.num_clbits).compose(circuit, qubits=np.argsort(perm).tolist())
    circuit = ElidePermutations()(circuit)
    sabre = SabreSwap(coupling_map=CouplingMap.from_line(num_qubits), heuristic='decay', trials=sabre_trials, seed=seed)
    circuit = sabre(circuit)
    return list(iter_layers(circuit))

def measurement_permutation(measurement_layers) -> list[int]:
    mapping: dict[int, int] = {}
    for layer in measurement_layers:
        for instruction in layer.data:
            if instruction.operation.name != 'measure':
                continue
            clbit = layer.find_bit(instruction.clbits[0]).index
            qubit = layer.find_bit(instruction.qubits[0]).index
            mapping[clbit] = qubit
    if not mapping:
        raise ValueError('no measurement operations found in measurement layers')
    size = max(mapping) + 1
    if sorted(mapping) != list(range(size)):
        raise ValueError('measurement layers do not cover a contiguous clbit range')
    return [mapping[c] for c in range(size)]

def get_bond_sizes(mpo) -> np.ndarray:
    return np.array([mpo.bond_size(i, i + 1) for i in range(len(mpo.sites) - 1)])

def swap_perm(perm, swaps):
    for q0, q1 in swaps:
        perm[q0], perm[q1] = (perm[q1], perm[q0])
    return perm

def permutation_alignment_score(perm_left, perm_right) -> int:
    inv_left = np.argsort(perm_left)
    inv_right = np.argsort(perm_right)
    return int(np.abs(inv_left - inv_right).sum())

def alignment_delta_for_swaps(perm_left, perm_right, swaps, how) -> int:
    current = permutation_alignment_score(perm_left, perm_right)
    next_left = list(perm_left)
    next_right = list(perm_right)
    if how in ('left', 'both'):
        next_left = swap_perm(next_left, list(swaps))
    if how in ('right', 'both'):
        next_right = swap_perm(next_right, list(swaps))
    return permutation_alignment_score(next_left, next_right) - current

def get_good_swaps(mpo, qubit_pairs, how, max_bond, cutoff, to_backend=None, equal: bool=False, compress_method: str='zipup', equalize_norms: bool=False):
    current_bonds = get_bond_sizes(mpo)
    swaps_left = qubit_pairs if how in ('left', 'both') else []
    swaps_right = qubit_pairs if how in ('right', 'both') else []
    with telemetry.capture() as probe_ledger:
        probe_mpo = apply_swaps(mpo, swaps_left=swaps_left, swaps_right=swaps_right, max_bond=max_bond, cutoff=cutoff, to_backend=to_backend, compress_method=compress_method, equalize_norms=equalize_norms)
    new_bonds = get_bond_sizes(probe_mpo)
    if equal:
        improved = np.nonzero(new_bonds <= current_bonds)[0]
    else:
        improved = np.nonzero(new_bonds < current_bonds)[0]
    gains = current_bonds.astype(float) - np.asarray(new_bonds, dtype=float)
    return (improved, probe_mpo, probe_ledger, gains)

def unswap(mpo, config: UnswapConfig, frame_left: FrameState, frame_right: FrameState, to_backend=None, deadline: float | None=None, event_cb: Callable[[dict[str, Any]], None] | None=None) -> tuple[Any, UnswapCycleResult]:
    num_qubits = len(mpo.sites)
    all_pairs = [(i, i + 1) for i in range(num_qubits - 1)]
    result = UnswapCycleResult(perm_left=list(range(num_qubits)), perm_right=list(range(num_qubits)))

    def emit(payload: dict[str, Any]) -> None:
        if event_cb is not None:
            event_cb(payload)
    cycle_deadline = time.time() + config.cycle_seconds if config.cycle_seconds is not None else None

    def deadline_reached(stage: str) -> bool:
        now = time.time()
        if deadline is not None and now > deadline:
            result.deadline_hit = True
            emit({'event': 'unswap_deadline', 'stage': stage})
            return True
        if cycle_deadline is None or now <= cycle_deadline:
            return False
        result.cycle_budget_hit = True
        emit({'event': 'unswap_cycle_budget', 'stage': stage})
        return True
    emit({'event': 'unswap_cycle_start', **get_tn_info(mpo)})
    num_improvements = 1
    start_counts = 1
    end_counts = 0
    iteration = 0
    while num_improvements > 0 and iteration < config.max_its and (start_counts != end_counts):
        if deadline_reached('before_iteration'):
            break
        num_improvements = 0
        start_counts = elem_counts(mpo)
        for how in config.hows:
            for parity in (0, 1):
                if deadline_reached('before_probe'):
                    emit({'event': 'unswap_cycle_end', 'iterations': result.iterations, **get_tn_info(mpo)})
                    return (mpo, result)
                candidate_pairs = all_pairs[parity::2]
                probe_cut = config.probe_cutoff if config.probe_cutoff is not None else config.cutoff
                improved_ids, probe_mpo, probe_ledger, gains = get_good_swaps(mpo, qubit_pairs=candidate_pairs, how=how, max_bond=config.max_bond, cutoff=probe_cut, to_backend=to_backend, equal=config.equal, compress_method=config.compress_method, equalize_norms=config.equalize_norms)
                result.probe_batches += 1
                if deadline_reached('after_probe'):
                    emit({'event': 'unswap_cycle_end', 'iterations': result.iterations, **get_tn_info(mpo)})
                    return (mpo, result)
                new_swaps = [all_pairs[i] for i in improved_ids if i % 2 == parity]
                if config.align_weight > 0 and new_swaps:
                    kept = []
                    for i in improved_ids:
                        i = int(i)
                        if i % 2 != parity:
                            continue
                        pair = all_pairs[i]
                        gain = float(gains[i])
                        if config.align_protect is not None and gain >= config.align_protect:
                            kept.append(pair)
                            continue
                        delta = alignment_delta_for_swaps(result.perm_left, result.perm_right, [pair], how)
                        if gain - config.align_weight * delta > 0:
                            kept.append(pair)
                    new_swaps = kept
                swaps_left = new_swaps if how in ('left', 'both') else []
                swaps_right = new_swaps if how in ('right', 'both') else []
                apply_cut = config.apply_cutoff if config.apply_cutoff is not None else config.cutoff
                if len(new_swaps) == len(candidate_pairs) and apply_cut == config.cutoff and (probe_cut == config.cutoff):
                    mpo = probe_mpo
                    telemetry.commit(probe_ledger)
                elif not new_swaps:
                    pass
                else:
                    mpo = apply_swaps(mpo, swaps_left=swaps_left, swaps_right=swaps_right, max_bond=config.max_bond, cutoff=apply_cut, to_backend=to_backend, compress_method=config.compress_method, equalize_norms=config.equalize_norms)
                if how in ('left', 'both'):
                    result.perm_left = swap_perm(result.perm_left, new_swaps)
                    for site_a, site_b in new_swaps:
                        frame_left.apply_site_transposition(site_a, site_b)
                if how in ('right', 'both'):
                    result.perm_right = swap_perm(result.perm_right, new_swaps)
                    for site_a, site_b in new_swaps:
                        frame_right.apply_site_transposition(site_a, site_b)
                result.swaps_applied += len(new_swaps)
                num_improvements += len(improved_ids)
                result.stats.append({'stage': 'unswapping', 'iteration': iteration, 'side': how, 'parity': parity, 'improved_candidates': len(improved_ids), 'swaps_applied': len(new_swaps), 'retained_local_frobenius_log10': telemetry.retained_local_frobenius_log10(), **get_tn_info(mpo)})
                emit(result.stats[-1] | {'event': 'unswap_step'})
        end_counts = elem_counts(mpo)
        iteration += 1
        result.iterations = iteration
    emit({'event': 'unswap_cycle_end', 'iterations': result.iterations, **get_tn_info(mpo)})
    return (mpo, result)
