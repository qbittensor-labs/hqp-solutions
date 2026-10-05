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
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Sequence
import numpy as np
from qiskit import QuantumCircuit
from ..structure.gadgets import CircuitStructure, collect_structure, discover_unique_inverse_pairs
from .permutation import Permutation, PermutationError
from .schedule import FrameScheduleError, ModuleFrame

@dataclass(frozen=True)
class StructuralConfig:
    inverse_tol: float = 1e-11
    min_support: float = 0.95
    min_pairs: int = 16
    cz_layer_tolerance: float = 4.0
    null_trials: int = 1000
    null_seed: int = 7

    def __post_init__(self) -> None:
        if not 0.0 <= self.inverse_tol < 1.0:
            raise ValueError('inverse_tol must satisfy 0 <= tol < 1')
        if not 0.0 < self.min_support <= 1.0:
            raise ValueError('min_support must lie in (0, 1]')
        if self.min_pairs <= 0:
            raise ValueError('min_pairs must be positive')
        if self.cz_layer_tolerance < 0:
            raise ValueError('cz_layer_tolerance must be nonnegative')
        if self.null_trials < 0:
            raise ValueError('null_trials must be nonnegative')

def _longest_mirror_ordered_subset(pairs: list) -> tuple[list, list]:
    if not pairs:
        return ([], [])
    best_lengths = [1] * len(pairs)
    predecessors = [-1] * len(pairs)
    for i in range(len(pairs)):
        for j in range(i):
            if pairs[j].late_layer > pairs[i].late_layer and best_lengths[j] + 1 > best_lengths[i]:
                best_lengths[i] = best_lengths[j] + 1
                predecessors[i] = j
    end = max(range(len(pairs)), key=lambda i: best_lengths[i])
    kept_indices = []
    while end != -1:
        kept_indices.append(end)
        end = predecessors[end]
    kept_set = set(kept_indices)
    kept = [pairs[i] for i in sorted(kept_set)]
    violations = [pairs[i] for i in range(len(pairs)) if i not in kept_set]
    return (kept, violations)

def _timed_cz_rate(early_edges: Sequence, late_layers_by_edge: dict[tuple[int, int], list[int]], permutation: Sequence[int], center: float, tolerance: float) -> tuple[int, float]:
    hits = 0
    for gate in early_edges:
        mapped = tuple(sorted((permutation[gate.q0], permutation[gate.q1])))
        predicted_layer = 2.0 * center - gate.layer
        if any((abs(layer - predicted_layer) <= tolerance for layer in late_layers_by_edge.get(mapped, ()))):
            hits += 1
    return (hits, hits / len(early_edges) if early_edges else 0.0)

def evaluate_eligibility(evidence: dict[str, Any], config: StructuralConfig) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    required = ('support_fraction', 'mapped_qubits', 'num_qubits', 'cz_early_count', 'cz_late_count', 'cz_timed_rate', 'cz_null_max')
    for key in required:
        value = evidence.get(key)
        if value is None or (isinstance(value, float) and (not math.isfinite(value))):
            reasons.append(f'missing or non-finite evidence field {key!r}')
    if reasons:
        return (False, reasons)
    if evidence['support_fraction'] < config.min_support:
        reasons.append(f"inverse support {evidence['support_fraction']:.4f} below threshold {config.min_support:.4f}")
    matched = evidence.get('matched_pairs')
    if matched is not None and matched < config.min_pairs:
        reasons.append(f'only {matched} mirror-ordered inverse pairs; require {config.min_pairs}')
    if evidence['mapped_qubits'] != evidence['num_qubits']:
        reasons.append(f"permutation supported on {evidence['mapped_qubits']}/{evidence['num_qubits']} qubits")
    if evidence['cz_early_count'] == 0 or evidence['cz_late_count'] == 0:
        reasons.append('no held-out CZ control in one or both spans')
    elif evidence.get('null_trials', 0) <= 0:
        reasons.append('no random-permutation null trials were run')
    elif evidence['cz_timed_rate'] <= evidence['cz_null_max']:
        reasons.append(f"held-out CZ timing rate {evidence['cz_timed_rate']:.4f} does not exceed null max {evidence['cz_null_max']:.4f}")
    return (not reasons, reasons)

def verify_module(circuit: QuantumCircuit, hypothesis: dict[str, Any], config: StructuralConfig, *, qasm_sha256: str, instance_id: str | None=None, structure: CircuitStructure | None=None) -> ModuleFrame:
    try:
        sigma = Permutation(hypothesis['permutation'])
    except (KeyError, PermutationError) as exc:
        raise FrameScheduleError(f'invalid module permutation: {exc}') from exc
    if len(sigma) != circuit.num_qubits:
        raise FrameScheduleError(f'permutation size {len(sigma)} does not match circuit {circuit.num_qubits}')
    module_id = int(hypothesis['module_id'])
    early_min = int(hypothesis['early_layer_min'])
    early_max = int(hypothesis['early_layer_max'])
    late_min = int(hypothesis['late_layer_min'])
    late_max = int(hypothesis['late_layer_max'])
    if not early_min <= early_max < late_min <= late_max:
        raise FrameScheduleError(f'module {module_id}: spans must satisfy early <= late without overlap')
    structure = structure or collect_structure(circuit)
    in_span_gates = [gate for gate in structure.oneq if early_min <= gate.layer <= early_max or late_min <= gate.layer <= late_max]
    discovery = discover_unique_inverse_pairs(in_span_gates, tolerance=config.inverse_tol)
    module_pairs = [pair for pair in discovery.pairs if early_min <= pair.early_layer <= early_max and late_min <= pair.late_layer <= late_max]
    unmatched: list[dict[str, Any]] = []
    consistent: list = []
    for pair in module_pairs:
        if sigma[pair.early_qubit] == pair.late_qubit:
            consistent.append(pair)
        else:
            unmatched.append({'kind': 'permutation_mismatch', 'early_qubit': pair.early_qubit, 'late_qubit': pair.late_qubit, 'early_layer': pair.early_layer, 'late_layer': pair.late_layer})
    kept_pairs: list = []
    by_wire: dict[int, list] = defaultdict(list)
    for pair in consistent:
        by_wire[pair.early_qubit].append(pair)
    for wire in sorted(by_wire):
        ordered = sorted(by_wire[wire], key=lambda pair: pair.early_layer)
        kept, violations = _longest_mirror_ordered_subset(ordered)
        kept_pairs.extend(kept)
        for pair in violations:
            unmatched.append({'kind': 'dependency_order_violation', 'early_qubit': pair.early_qubit, 'late_qubit': pair.late_qubit, 'early_layer': pair.early_layer, 'late_layer': pair.late_layer})
    support_fraction = len(kept_pairs) / len(module_pairs) if module_pairs else 0.0
    mapped_qubits = len({pair.early_qubit for pair in kept_pairs})
    paired_gate_count = 2 * len(module_pairs)
    unpaired_gate_count = len(in_span_gates) - paired_gate_count
    if kept_pairs:
        center = float(np.median([pair.midpoint for pair in kept_pairs]))
    else:
        center = (early_max + late_min) / 2.0
    early_cz = [g for g in structure.cz if early_min <= g.layer <= early_max]
    late_cz = [g for g in structure.cz if late_min <= g.layer <= late_max]
    late_by_edge: dict[tuple[int, int], list[int]] = defaultdict(list)
    for gate in late_cz:
        late_by_edge[gate.edge].append(gate.layer)
    timed_hits, timed_rate = _timed_cz_rate(early_cz, late_by_edge, sigma.mapping, center, config.cz_layer_tolerance)
    null_rates: list[float] = []
    if config.null_trials > 0 and early_cz:
        rng = np.random.default_rng(config.null_seed + module_id)
        for _ in range(config.null_trials):
            random_permutation = rng.permutation(circuit.num_qubits)
            _, rate = _timed_cz_rate(early_cz, late_by_edge, random_permutation, center, config.cz_layer_tolerance)
            null_rates.append(rate)
    evidence: dict[str, Any] = {'num_qubits': circuit.num_qubits, 'support_fraction': support_fraction, 'matched_pairs': len(kept_pairs), 'module_pair_count': len(module_pairs), 'in_span_oneq_gates': len(in_span_gates), 'unpaired_oneq_gates': unpaired_gate_count, 'mapped_qubits': mapped_qubits, 'unmatched_operations': unmatched, 'boundaries': {'early_layer_min': early_min, 'early_layer_max': early_max, 'late_layer_min': late_min, 'late_layer_max': late_max}, 'center': center, 'involutive': sigma.is_involution(), 'fixed_points': sigma.fixed_points(), 'cz_early_count': len(early_cz), 'cz_late_count': len(late_cz), 'cz_timed_hits': timed_hits, 'cz_timed_rate': timed_rate, 'cz_null_mean': float(np.mean(null_rates)) if null_rates else None, 'cz_null_max': float(max(null_rates)) if null_rates else None, 'cz_empirical_p': (1 + sum((rate >= timed_rate for rate in null_rates))) / (1 + len(null_rates)) if null_rates else None, 'null_trials': len(null_rates), 'config': asdict(config)}
    eligible, reasons = evaluate_eligibility(evidence, config)
    return ModuleFrame(instance_id=instance_id, qasm_sha256=qasm_sha256, module_id=module_id, num_qubits=circuit.num_qubits, enter_layer=early_max + 1, exit_layer=late_max + 1, permutation=sigma, evidence=evidence, eligible=eligible, reasons=tuple(reasons))
