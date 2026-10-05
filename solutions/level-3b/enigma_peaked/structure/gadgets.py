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
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import pickle
import sys
import time
import warnings
from typing import Any, Sequence
import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator
from scipy.optimize import linear_sum_assignment
HEURISTIC_WARNING = 'HEURISTIC ONLY: inverse-U and CZ correlations do not prove that a module equals the recovered permutation. This circuit is not equivalence-certified.'

@dataclass(frozen=True)
class OneQGate:
    index: int
    layer: int
    qubit: int
    name: str
    matrix: np.ndarray = field(repr=False, compare=False)

@dataclass(frozen=True)
class CZGate:
    layer: int
    q0: int
    q1: int

    @property
    def edge(self) -> tuple[int, int]:
        return (self.q0, self.q1)

@dataclass(frozen=True)
class InversePair:
    early_index: int
    late_index: int
    early_layer: int
    late_layer: int
    early_qubit: int
    late_qubit: int
    fidelity: float

    @property
    def midpoint(self) -> float:
        return (self.early_layer + self.late_layer) / 2.0

@dataclass
class PairDiscovery:
    pairs: list[InversePair]
    candidate_pair_count: int
    ambiguous_gate_count: int
    unmatched_gate_count: int

@dataclass(frozen=True)
class AnalysisConfig:
    inverse_tol: float = 1e-11
    inverse_batch: int = 256
    cluster_gap: float = 4.0
    min_pairs: int = 16
    min_support: float = 0.75
    cz_layer_tolerance: float = 4.0
    null_trials: int = 100
    null_seed: int = 3

@dataclass
class ModuleEvidence:
    module_id: int
    pair_count: int
    center: float
    midpoint_min: float
    midpoint_max: float
    early_layer_min: int
    early_layer_max: int
    late_layer_min: int
    late_layer_max: int
    permutation: list[int]
    transpositions: list[list[int]]
    fixed_points: list[int]
    involutive: bool
    mapped_qubits: int
    support_count: int
    support_fraction: float
    mean_inverse_fidelity: float
    cz_early_count: int
    cz_late_count: int
    cz_timed_hits: int
    cz_timed_rate: float
    cz_bag_hits: int
    cz_bag_rate: float
    cz_null_mean: float
    cz_null_std: float
    cz_null_max: float
    cz_empirical_p: float
    replacement_eligible: bool
    replacement_reasons: list[str]

    @property
    def full_layer_span(self) -> tuple[int, int]:
        return (self.early_layer_min, self.late_layer_max)

@dataclass
class DiagnosticReport:
    circuit_name: str
    num_qubits: int
    num_clbits: int
    operation_count: int
    asap_depth: int
    oneq_gate_count: int
    cz_gate_count: int
    other_twoq_counts: dict[str, int]
    skipped_nonunitary_oneq: int
    inverse_candidate_pairs: int
    unique_inverse_pairs: int
    ambiguous_inverse_gates: int
    unmatched_inverse_gates: int
    config: dict[str, Any]
    modules: list[ModuleEvidence]
    warning: str = HEURISTIC_WARNING

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

@dataclass
class CircuitStructure:
    oneq: list[OneQGate]
    cz: list[CZGate]
    layers: list[Any]
    other_twoq_counts: Counter[str]
    skipped_nonunitary_oneq: int

def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value

def inverse_fidelity(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.complex128)
    v = np.asarray(v, dtype=np.complex128)
    if u.shape != (2, 2) or v.shape != (2, 2):
        raise ValueError('inverse_fidelity expects two 2x2 matrices')
    return float(abs(np.trace(v @ u)) / 2.0)

def collect_structure(circuit: QuantumCircuit) -> CircuitStructure:
    layers = list(circuit_to_dag(circuit).layers())
    oneq: list[OneQGate] = []
    cz: list[CZGate] = []
    other_twoq: Counter[str] = Counter()
    skipped = 0
    for layer_index, layer in enumerate(layers):
        for node in layer['graph'].op_nodes():
            qubits = [circuit.find_bit(q).index for q in node.qargs]
            if len(qubits) == 1:
                if node.op.name in {'barrier', 'delay', 'measure', 'reset'}:
                    continue
                try:
                    matrix = np.asarray(Operator(node.op).data, dtype=np.complex128)
                except Exception:
                    skipped += 1
                    continue
                if matrix.shape != (2, 2):
                    skipped += 1
                    continue
                oneq.append(OneQGate(index=len(oneq), layer=layer_index, qubit=qubits[0], name=node.op.name, matrix=matrix))
            elif len(qubits) == 2:
                a, b = sorted(qubits)
                if node.op.name == 'cz':
                    cz.append(CZGate(layer_index, a, b))
                else:
                    other_twoq[node.op.name] += 1
    return CircuitStructure(oneq, cz, layers, other_twoq, skipped)

def discover_unique_inverse_pairs(gates: Sequence[OneQGate], *, tolerance: float=1e-11, batch_size: int=256) -> PairDiscovery:
    if tolerance < 0.0 or tolerance >= 1.0:
        raise ValueError('tolerance must satisfy 0 <= tolerance < 1')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if not gates:
        return PairDiscovery([], 0, 0, 0)
    left = np.stack([g.matrix.T.reshape(-1) for g in gates])
    right = np.stack([g.matrix.reshape(-1) for g in gates])
    adjacency: list[set[int]] = [set() for _ in gates]
    candidate_pairs = 0
    threshold = 1.0 - tolerance
    for start in range(0, len(gates), batch_size):
        scores = np.abs(left[start:start + batch_size] @ right.T) / 2.0
        for local_i, j_raw in np.argwhere(scores >= threshold):
            i = start + int(local_i)
            j = int(j_raw)
            if gates[i].layer >= gates[j].layer:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)
            candidate_pairs += 1
    pairs: list[InversePair] = []
    for i, candidates in enumerate(adjacency):
        if len(candidates) != 1:
            continue
        j = next(iter(candidates))
        if i >= j or len(adjacency[j]) != 1:
            continue
        early, late = (gates[i], gates[j])
        if early.layer > late.layer:
            early, late = (late, early)
        pairs.append(InversePair(early_index=early.index, late_index=late.index, early_layer=early.layer, late_layer=late.layer, early_qubit=early.qubit, late_qubit=late.qubit, fidelity=inverse_fidelity(early.matrix, late.matrix)))
    ambiguous = sum((len(candidates) > 1 for candidates in adjacency))
    unmatched = sum((len(candidates) == 0 for candidates in adjacency))
    pairs.sort(key=lambda p: (p.midpoint, p.early_layer, p.early_qubit))
    return PairDiscovery(pairs, candidate_pairs, ambiguous, unmatched)

def cluster_local_pairs(pairs: Sequence[InversePair], *, max_midpoint_gap: float=4.0, min_pairs: int=16) -> list[list[InversePair]]:
    if max_midpoint_gap < 0:
        raise ValueError('max_midpoint_gap must be nonnegative')
    if min_pairs <= 0:
        raise ValueError('min_pairs must be positive')
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda p: p.midpoint)
    groups: list[list[InversePair]] = [[ordered[0]]]
    for pair in ordered[1:]:
        if pair.midpoint - groups[-1][-1].midpoint > max_midpoint_gap:
            groups.append([pair])
        else:
            groups[-1].append(pair)
    return [group for group in groups if len(group) >= min_pairs]

def _timed_cz_rate(early_edges: Sequence[CZGate], late_layers_by_edge: dict[tuple[int, int], list[int]], permutation: Sequence[int], center: float, tolerance: float) -> tuple[int, float]:
    hits = 0
    for gate in early_edges:
        mapped = tuple(sorted((permutation[gate.q0], permutation[gate.q1])))
        predicted_layer = 2.0 * center - gate.layer
        if any((abs(layer - predicted_layer) <= tolerance for layer in late_layers_by_edge.get(mapped, ()))):
            hits += 1
    return (hits, hits / len(early_edges) if early_edges else 0.0)

def infer_module(pair_cluster: Sequence[InversePair], *, module_id: int, num_qubits: int, cz_gates: Sequence[CZGate], config: AnalysisConfig) -> ModuleEvidence:
    counts = np.zeros((num_qubits, num_qubits), dtype=np.int64)
    for pair in pair_cluster:
        counts[pair.early_qubit, pair.late_qubit] += 1
    rows, cols = linear_sum_assignment(-counts)
    permutation = np.empty(num_qubits, dtype=np.int64)
    permutation[rows] = cols
    consistent = [pair for pair in pair_cluster if permutation[pair.early_qubit] == pair.late_qubit]
    if not consistent:
        raise ValueError('assignment has zero supporting inverse pairs')
    support_count = len(consistent)
    support_fraction = support_count / len(pair_cluster)
    mapped_qubits = sum((counts[q, permutation[q]] > 0 for q in range(num_qubits)))
    involutive = bool(np.array_equal(permutation[permutation], np.arange(num_qubits)))
    fixed_points = [q for q in range(num_qubits) if permutation[q] == q]
    transpositions = [[q, int(permutation[q])] for q in range(num_qubits) if q < permutation[q]] if involutive else []
    center = float(np.median([pair.midpoint for pair in consistent]))
    early_min = min((pair.early_layer for pair in consistent))
    early_max = max((pair.early_layer for pair in consistent))
    late_min = min((pair.late_layer for pair in consistent))
    late_max = max((pair.late_layer for pair in consistent))
    early_cz = [g for g in cz_gates if early_min <= g.layer <= early_max]
    late_cz = [g for g in cz_gates if late_min <= g.layer <= late_max]
    late_by_edge: dict[tuple[int, int], list[int]] = defaultdict(list)
    for gate in late_cz:
        late_by_edge[gate.edge].append(gate.layer)
    timed_hits, timed_rate = _timed_cz_rate(early_cz, late_by_edge, permutation, center, config.cz_layer_tolerance)
    mapped_early_bag = Counter((tuple(sorted((int(permutation[g.q0]), int(permutation[g.q1])))) for g in early_cz))
    late_bag = Counter((g.edge for g in late_cz))
    bag_hits = sum((mapped_early_bag & late_bag).values())
    bag_rate = bag_hits / max(len(early_cz), len(late_cz), 1)
    null_rates: list[float] = []
    if config.null_trials > 0 and early_cz:
        rng = np.random.default_rng(config.null_seed + module_id)
        for _ in range(config.null_trials):
            random_permutation = rng.permutation(num_qubits)
            _, rate = _timed_cz_rate(early_cz, late_by_edge, random_permutation, center, config.cz_layer_tolerance)
            null_rates.append(rate)
    null_mean = float(np.mean(null_rates)) if null_rates else 0.0
    null_std = float(np.std(null_rates)) if null_rates else 0.0
    null_max = float(max(null_rates)) if null_rates else 0.0
    empirical_p = (1 + sum((rate >= timed_rate for rate in null_rates))) / (1 + len(null_rates)) if null_rates else float('nan')
    reasons: list[str] = []
    if support_fraction < config.min_support:
        reasons.append(f'inverse-pair support {support_fraction:.3f} < {config.min_support:.3f}')
    if mapped_qubits != num_qubits:
        reasons.append(f'mapping supported on {mapped_qubits}/{num_qubits} qubits')
    if not involutive:
        reasons.append('fitted permutation is not involutive')
    if not early_cz or not late_cz:
        reasons.append('no independent CZ control in one or both spans')
    elif null_rates and timed_rate <= null_max:
        reasons.append(f'CZ timing rate {timed_rate:.3f} does not exceed null max {null_max:.3f}')
    elif not null_rates and timed_rate <= 0.0:
        reasons.append('CZ timing control has zero hits')
    return ModuleEvidence(module_id=module_id, pair_count=len(pair_cluster), center=center, midpoint_min=min((pair.midpoint for pair in pair_cluster)), midpoint_max=max((pair.midpoint for pair in pair_cluster)), early_layer_min=early_min, early_layer_max=early_max, late_layer_min=late_min, late_layer_max=late_max, permutation=[int(x) for x in permutation], transpositions=transpositions, fixed_points=fixed_points, involutive=involutive, mapped_qubits=mapped_qubits, support_count=support_count, support_fraction=support_fraction, mean_inverse_fidelity=float(np.mean([p.fidelity for p in consistent])), cz_early_count=len(early_cz), cz_late_count=len(late_cz), cz_timed_hits=timed_hits, cz_timed_rate=timed_rate, cz_bag_hits=bag_hits, cz_bag_rate=bag_rate, cz_null_mean=null_mean, cz_null_std=null_std, cz_null_max=null_max, cz_empirical_p=empirical_p, replacement_eligible=not reasons, replacement_reasons=reasons)

def analyze_circuit(circuit: QuantumCircuit, config: AnalysisConfig | None=None) -> DiagnosticReport:
    config = config or AnalysisConfig()
    structure = collect_structure(circuit)
    discovery = discover_unique_inverse_pairs(structure.oneq, tolerance=config.inverse_tol, batch_size=config.inverse_batch)
    clusters = cluster_local_pairs(discovery.pairs, max_midpoint_gap=config.cluster_gap, min_pairs=config.min_pairs)
    modules = [infer_module(cluster, module_id=i, num_qubits=circuit.num_qubits, cz_gates=structure.cz, config=config) for i, cluster in enumerate(clusters)]
    return DiagnosticReport(circuit_name=circuit.name, num_qubits=circuit.num_qubits, num_clbits=circuit.num_clbits, operation_count=len(circuit.data), asap_depth=len(structure.layers), oneq_gate_count=len(structure.oneq), cz_gate_count=len(structure.cz), other_twoq_counts=dict(structure.other_twoq_counts), skipped_nonunitary_oneq=structure.skipped_nonunitary_oneq, inverse_candidate_pairs=discovery.candidate_pair_count, unique_inverse_pairs=len(discovery.pairs), ambiguous_inverse_gates=discovery.ambiguous_gate_count, unmatched_inverse_gates=discovery.unmatched_gate_count, config=asdict(config), modules=modules)

def _eligible_nonoverlapping_modules(report: DiagnosticReport) -> list[ModuleEvidence]:
    modules = sorted((module for module in report.modules if module.replacement_eligible), key=lambda module: module.full_layer_span)
    for previous, current in zip(modules, modules[1:]):
        if previous.full_layer_span[1] >= current.full_layer_span[0]:
            raise ValueError(f'eligible module spans overlap; refusing heuristic replacement: {previous.full_layer_span} vs {current.full_layer_span}')
    return modules

def build_heuristic_reduction(circuit: QuantumCircuit, report: DiagnosticReport, *, module_ids: set[int] | None=None, replacement: str='permutation') -> QuantumCircuit:
    warnings.warn("build_heuristic_reduction is a deprecated diagnostic; it is not equivalence-preserving and never authorizes gate deletion. Use the exact frame schedule from 'enigma-peaked frame-report' instead.", DeprecationWarning, stacklevel=2)
    if circuit.num_clbits:
        raise ValueError('heuristic builder currently supports unitary circuits only')
    if replacement not in {'permutation', 'identity'}:
        raise ValueError("replacement must be 'permutation' or 'identity'")
    modules = _eligible_nonoverlapping_modules(report)
    if module_ids is not None:
        modules = [module for module in modules if module.module_id in module_ids]
    if not modules:
        raise ValueError('no full, CZ-validated modules are replacement-eligible')
    layers = list(circuit_to_dag(circuit).layers())
    by_start = {module.full_layer_span[0]: module for module in modules}
    covered: dict[int, ModuleEvidence] = {}
    for module in modules:
        start, stop = module.full_layer_span
        for layer in range(start, stop + 1):
            covered[layer] = module
    reduced = QuantumCircuit(circuit.num_qubits, name=f'{circuit.name}_HEURISTIC_gadget_{replacement}_reduction')
    for layer_index, layer in enumerate(layers):
        if layer_index in by_start:
            module = by_start[layer_index]
            if replacement == 'permutation':
                for a, b in module.transpositions:
                    reduced.swap(a, b)
        if layer_index in covered:
            continue
        for node in layer['graph'].op_nodes():
            if node.cargs or node.op.name in {'measure', 'reset'}:
                raise ValueError('heuristic builder encountered a non-unitary/classical operation')
            qargs = [reduced.qubits[circuit.find_bit(q).index] for q in node.qargs]
            reduced.append(node.op, qargs)
    reduced.metadata = {**(circuit.metadata or {}), 'HEURISTIC_NOT_EQUIVALENCE_PROVEN': True, 'warning': HEURISTIC_WARNING, 'replacement': replacement, 'replaced_modules': [{'module_id': module.module_id, 'layers': list(module.full_layer_span), 'permutation': module.permutation} for module in modules]}
    return reduced

def beam_from_aer_mps(aer_mps: tuple[Any, Any], *, beam_width: int=1024, topk: int=8) -> list[dict[str, Any]]:
    if beam_width <= 0 or topk <= 0:
        raise ValueError('beam_width and topk must be positive')
    gammas, lambdas = aer_mps
    beams: list[tuple[str, np.ndarray, float]] = [('', np.ones(1, dtype=np.complex128), 1.0)]
    for site_index, site in enumerate(gammas):
        candidates: list[tuple[str, np.ndarray, float]] = []
        for bits, vector, _ in beams:
            for bit in (0, 1):
                next_vector = vector @ np.asarray(site[bit])
                if site_index < len(lambdas):
                    next_vector = next_vector * np.asarray(lambdas[site_index])
                weight = float(np.vdot(next_vector, next_vector).real)
                candidates.append((bits + str(bit), next_vector, weight))
        candidates.sort(key=lambda item: item[2], reverse=True)
        beams = candidates[:beam_width]
    beams.sort(key=lambda item: item[2], reverse=True)
    return [{'site_bits_q0_to_qN': bits, 'qiskit_bitstring_qN_to_q0': bits[::-1], 'probability': weight} for bits, _, weight in beams[:topk]]

def _aer_site_matrices(aer_mps: tuple[Any, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    gammas, lambdas = aer_mps
    matrices: list[tuple[np.ndarray, np.ndarray]] = []
    for site_index, site in enumerate(gammas):
        bit_matrices = []
        for bit in (0, 1):
            matrix = np.asarray(site[bit], dtype=np.complex128)
            if matrix.ndim != 2:
                raise ValueError('Aer MPS Gamma entries must be matrices')
            if site_index < len(lambdas):
                values = np.asarray(lambdas[site_index], dtype=np.float64)
                if matrix.shape[1] != len(values):
                    raise ValueError('Aer MPS Gamma/lambda bond dimensions disagree')
                matrix = matrix * values[np.newaxis, :]
            bit_matrices.append(matrix)
        matrices.append((bit_matrices[0], bit_matrices[1]))
    return matrices

def exact_one_site_marginals_from_aer_mps(aer_mps: tuple[Any, Any]) -> dict[str, Any]:
    matrices = _aer_site_matrices(aer_mps)
    n = len(matrices)
    if n == 0:
        return {'site_bits_q0_to_qN': '', 'qiskit_bitstring_qN_to_q0': '', 'marginals': [], 'min_margin': 0.0, 'mean_margin': 0.0}
    left_env: list[np.ndarray] = [np.ones((1, 1), dtype=np.complex128)]
    for a0, a1 in matrices:
        current = left_env[-1]
        next_env = a0.conj().T @ current @ a0 + a1.conj().T @ current @ a1
        left_env.append(next_env)
    right_env: list[np.ndarray] = [np.empty((0, 0), dtype=np.complex128) for _ in range(n + 1)]
    right_env[n] = np.ones((1, 1), dtype=np.complex128)
    for i in range(n - 1, -1, -1):
        a0, a1 = matrices[i]
        current = right_env[i + 1]
        right_env[i] = a0 @ current @ a0.conj().T + a1 @ current @ a1.conj().T
    rows: list[dict[str, Any]] = []
    bits: list[str] = []
    margins: list[float] = []
    for i, (a0, a1) in enumerate(matrices):
        left = left_env[i]
        right = right_env[i + 1]
        raw = []
        for matrix in (a0, a1):
            value = np.trace(left @ matrix @ right @ matrix.conj().T)
            real_value = float(np.real_if_close(value).real)
            if real_value < -1e-10:
                raise ValueError(f'negative marginal weight at site {i}: {real_value}')
            raw.append(max(0.0, real_value))
        total = raw[0] + raw[1]
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f'invalid marginal normalization at site {i}: {total}')
        p0, p1 = (raw[0] / total, raw[1] / total)
        chosen = '0' if p0 >= p1 else '1'
        margin = abs(p0 - p1)
        bits.append(chosen)
        margins.append(margin)
        rows.append({'qubit': i, 'p0': p0, 'p1': p1, 'chosen_bit': chosen, 'margin': margin})
    site_bits = ''.join(bits)
    return {'site_bits_q0_to_qN': site_bits, 'qiskit_bitstring_qN_to_q0': site_bits[::-1], 'marginals': rows, 'min_margin': min(margins), 'mean_margin': float(np.mean(margins)), 'normalization': float(np.real_if_close(left_env[-1][0, 0]).real)}

def run_aer_mps_probe(circuit: QuantumCircuit, *, max_bond: int=64, cutoff: float=1e-10, use_lapack: bool=False, swap_direction: str='left', ordering_seed: int | None=None, beam_width: int=1024, topk: int=8, save_path: str | Path | None=None) -> dict[str, Any]:
    try:
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise RuntimeError('--aer-probe requires the optional qiskit-aer package') from exc
    if max_bond <= 0:
        raise ValueError('max_bond must be positive')
    if cutoff < 0.0:
        raise ValueError('cutoff must be nonnegative')
    if swap_direction not in {'left', 'right'}:
        raise ValueError("swap_direction must be 'left' or 'right'")
    work = circuit.remove_final_measurements(inplace=False)
    logical_to_site = list(range(work.num_qubits))
    if ordering_seed is not None:
        logical_to_site = [int(value) for value in np.random.default_rng(ordering_seed).permutation(work.num_qubits)]
        reordered = QuantumCircuit(work.num_qubits, name=f'{work.name}_ord{ordering_seed}')
        reordered.compose(work, qubits=logical_to_site, inplace=True)
        work = reordered
    simulator = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension=max_bond, matrix_product_state_truncation_threshold=cutoff, mps_lapack=use_lapack, mps_swap_direction=f'mps_swap_{swap_direction}')
    compiled = transpile(work, simulator, optimization_level=0)
    compiled.save_matrix_product_state(label='gadget_mps')
    started = time.perf_counter()
    result = simulator.run(compiled).result()
    if not result.success:
        raise RuntimeError(f'Aer MPS simulation failed: {result.status}')
    aer_mps = result.data(0)['gadget_mps']
    elapsed = time.perf_counter() - started
    gammas, lambdas = aer_mps
    reached_bond = max((len(values) for values in lambdas), default=1)
    blind_marginal_site = exact_one_site_marginals_from_aer_mps(aer_mps)
    site_bits = blind_marginal_site['site_bits_q0_to_qN']
    logical_bits = ''.join((site_bits[logical_to_site[q]] for q in range(work.num_qubits)))
    logical_rows = []
    for logical_qubit, site_qubit in enumerate(logical_to_site):
        row = dict(blind_marginal_site['marginals'][site_qubit])
        row['site_qubit'] = site_qubit
        row['qubit'] = logical_qubit
        logical_rows.append(row)
    blind_marginal = {**blind_marginal_site, 'site_bits_in_simulator_order': site_bits, 'site_bits_q0_to_qN': logical_bits, 'qiskit_bitstring_qN_to_q0': logical_bits[::-1], 'marginals': logical_rows}
    beam = beam_from_aer_mps(aer_mps, beam_width=beam_width, topk=topk)
    for candidate in beam:
        candidate_site_bits = candidate['site_bits_q0_to_qN']
        candidate_logical_bits = ''.join((candidate_site_bits[logical_to_site[q]] for q in range(work.num_qubits)))
        candidate['site_bits_in_simulator_order'] = candidate_site_bits
        candidate['site_bits_q0_to_qN'] = candidate_logical_bits
        candidate['qiskit_bitstring_qN_to_q0'] = candidate_logical_bits[::-1]
    probe = {'circuit_name': circuit.name, 'num_qubits': circuit.num_qubits, 'configured_max_bond': max_bond, 'reached_saved_bond': reached_bond, 'cutoff': cutoff, 'mps_lapack': use_lapack, 'mps_swap_direction': swap_direction, 'ordering_seed': ordering_seed, 'logical_to_site': logical_to_site, 'beam_width': beam_width, 'topk': beam, 'blind_one_site_marginal': blind_marginal, 'seconds': elapsed, 'warning': 'Aer state and beam are approximate; this is not a solve certificate.'}
    if save_path is not None:
        path = Path(save_path)
        with path.open('wb') as handle:
            pickle.dump({'mps': aer_mps, 'probe': probe, 'warning': HEURISTIC_WARNING}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        probe['saved_mps'] = str(path)
    return probe

def format_report(report: DiagnosticReport) -> str:
    lines = [f'circuit={report.circuit_name} qubits={report.num_qubits} ops={report.operation_count} ASAP_depth={report.asap_depth}', f"1q={report.oneq_gate_count} cz={report.cz_gate_count} other_2q={report.other_twoq_counts or '{}'}", f'inverse candidates={report.inverse_candidate_pairs} mutually_unique={report.unique_inverse_pairs} ambiguous_gates={report.ambiguous_inverse_gates} unmatched_gates={report.unmatched_inverse_gates}', f'localized_modules={len(report.modules)}']
    for module in report.modules:
        status = 'HEURISTIC-ELIGIBLE' if module.replacement_eligible else 'REPORT-ONLY'
        lines.extend(['', f'module {module.module_id}: center={module.center:.1f} midpoints={module.midpoint_min:.1f}..{module.midpoint_max:.1f} pairs={module.pair_count}', f'  spans early={module.early_layer_min}..{module.early_layer_max} late={module.late_layer_min}..{module.late_layer_max}', f'  U support={module.support_count}/{module.pair_count} ({module.support_fraction:.3f}) mapped={module.mapped_qubits}/{report.num_qubits} involutive={module.involutive} fixed={len(module.fixed_points)}', f'  CZ timed={module.cz_timed_hits}/{module.cz_early_count} ({module.cz_timed_rate:.3f}) null={module.cz_null_mean:.3f}+/-{module.cz_null_std:.3f} max={module.cz_null_max:.3f} p={module.cz_empirical_p:.4g}', f'  CZ bag={module.cz_bag_hits}/max({module.cz_early_count},{module.cz_late_count})={module.cz_bag_rate:.3f}', f'  status={status}', f'  permutation={module.permutation}'])
        if module.replacement_reasons:
            lines.append('  reasons=' + '; '.join(module.replacement_reasons))
    lines.extend(['', 'WARNING: ' + report.warning])
    return '\n'.join(lines)

def _write_heuristic_qasm(path: str | Path, circuit: QuantumCircuit) -> None:
    text = qasm2.dumps(circuit)
    header = f'// HEURISTIC ONLY -- NOT EQUIVALENCE CERTIFIED\n// {HEURISTIC_WARNING}\n'
    Path(path).write_text(header + text)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('qasm')
    parser.add_argument('--inverse-tol', type=float, default=1e-11)
    parser.add_argument('--inverse-batch', type=int, default=256)
    parser.add_argument('--cluster-gap', type=float, default=4.0)
    parser.add_argument('--min-pairs', type=int, default=16)
    parser.add_argument('--min-support', type=float, default=0.75)
    parser.add_argument('--cz-layer-tolerance', type=float, default=4.0)
    parser.add_argument('--null-trials', type=int, default=100)
    parser.add_argument('--null-seed', type=int, default=3)
    parser.add_argument('--json-out')
    parser.add_argument('--heuristic-qasm')
    parser.add_argument('--heuristic-modules')
    parser.add_argument('--heuristic-replacement', choices=('permutation', 'identity'), default='permutation')
    parser.add_argument('--aer-probe', choices=('original', 'heuristic'))
    parser.add_argument('--aer-max-bond', type=int, default=64)
    parser.add_argument('--aer-cutoff', type=float, default=1e-10)
    parser.add_argument('--aer-lapack', action='store_true')
    parser.add_argument('--aer-swap-direction', choices=('left', 'right'), default='left')
    parser.add_argument('--aer-ordering-seed', type=int)
    parser.add_argument('--aer-beam', type=int, default=1024)
    parser.add_argument('--aer-topk', type=int, default=8)
    parser.add_argument('--aer-save')
    return parser

def main(argv: Sequence[str] | None=None) -> int:
    args = _build_parser().parse_args(argv)
    config = AnalysisConfig(inverse_tol=args.inverse_tol, inverse_batch=args.inverse_batch, cluster_gap=args.cluster_gap, min_pairs=args.min_pairs, min_support=args.min_support, cz_layer_tolerance=args.cz_layer_tolerance, null_trials=args.null_trials, null_seed=args.null_seed)
    circuit = QuantumCircuit.from_qasm_file(args.qasm)
    report = analyze_circuit(circuit, config)
    print(format_report(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report.to_dict(), indent=2) + '\n')
        print(f'wrote diagnostic JSON: {args.json_out}')
    heuristic: QuantumCircuit | None = None
    if args.heuristic_qasm or args.aer_probe == 'heuristic':
        print('WARNING: ' + HEURISTIC_WARNING, file=sys.stderr)
        module_ids = None
        if args.heuristic_modules:
            module_ids = {int(value) for value in args.heuristic_modules.split(',')}
        heuristic = build_heuristic_reduction(circuit, report, module_ids=module_ids, replacement=args.heuristic_replacement)
    if args.heuristic_qasm:
        _write_heuristic_qasm(args.heuristic_qasm, heuristic)
        print(f'wrote HEURISTIC, non-certified QASM: {args.heuristic_qasm}')
    if args.aer_probe:
        selected = circuit if args.aer_probe == 'original' else heuristic
        probe = run_aer_mps_probe(selected, max_bond=args.aer_max_bond, cutoff=args.aer_cutoff, use_lapack=args.aer_lapack, swap_direction=args.aer_swap_direction, ordering_seed=args.aer_ordering_seed, beam_width=args.aer_beam, topk=args.aer_topk, save_path=args.aer_save)
        print('AER_MPS_PROBE ' + json.dumps(probe, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
