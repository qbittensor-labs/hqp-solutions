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
from dataclasses import asdict, dataclass
import heapq
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Operator, Statevector
PAULI = (np.array([[1, 0], [0, 1]], dtype=np.complex128), np.array([[0, 1], [1, 0]], dtype=np.complex128), np.array([[0, -1j], [1j, 0]], dtype=np.complex128), np.array([[1, 0], [0, -1]], dtype=np.complex128))
PAULI_NAMES = ('I', 'X', 'Y', 'Z')

@dataclass(frozen=True)
class SolverConfig:
    top_k: int | None = 512
    relative_threshold: float = 0.0
    max_pauli_weight: int | None = None
    coefficient_epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError('top_k must be positive or None')
        if self.relative_threshold < 0:
            raise ValueError('relative_threshold must be nonnegative')
        if self.max_pauli_weight is not None and self.max_pauli_weight < 0:
            raise ValueError('max_pauli_weight must be nonnegative or None')
        if self.coefficient_epsilon < 0:
            raise ValueError('coefficient_epsilon must be nonnegative')

@dataclass(frozen=True)
class CompiledGate:
    kind: str
    qubits: tuple[int, ...]
    oneq_table: tuple[tuple[tuple[int, complex], ...], ...] | None = None

@dataclass
class PropagationStats:
    gates_visited: int = 0
    gates_active: int = 0
    prune_calls: int = 0
    terms_dropped: int = 0
    max_terms_before_prune: int = 1

def _label_from_masks(xmask: int, zmask: int, qubit: int) -> int:
    x = xmask >> qubit & 1
    z = zmask >> qubit & 1
    if not x:
        return 3 if z else 0
    return 2 if z else 1

def _replace_label(xmask: int, zmask: int, qubit: int, label: int) -> tuple[int, int]:
    bit = 1 << qubit
    xmask &= ~bit
    zmask &= ~bit
    if label in (1, 2):
        xmask |= bit
    if label in (2, 3):
        zmask |= bit
    return (xmask, zmask)

def _oneq_transfer(unitary: np.ndarray, *, epsilon: float=1e-14):
    unitary = np.asarray(unitary, dtype=np.complex128)
    if unitary.shape != (2, 2):
        raise ValueError('one-qubit gate matrix must be 2x2')
    table: list[tuple[tuple[int, complex], ...]] = []
    for source in PAULI:
        transformed = unitary.conj().T @ source @ unitary
        row = []
        for label, target in enumerate(PAULI):
            coefficient = np.trace(target.conj().T @ transformed) / 2.0
            if abs(coefficient) > epsilon:
                if abs(coefficient.imag) < epsilon:
                    coefficient = complex(float(coefficient.real), 0.0)
                row.append((label, complex(coefficient)))
        table.append(tuple(row))
    return tuple(table)

def _build_twoq_clifford_table(unitary: np.ndarray):
    table: dict[tuple[int, int], tuple[int, int, complex]] = {}
    for label0 in range(4):
        for label1 in range(4):
            source = np.kron(PAULI[label1], PAULI[label0])
            transformed = unitary.conj().T @ source @ unitary
            matches = []
            for out0 in range(4):
                for out1 in range(4):
                    target = np.kron(PAULI[out1], PAULI[out0])
                    coefficient = np.trace(target.conj().T @ transformed) / 4.0
                    if abs(coefficient) > 1e-12:
                        matches.append((out0, out1, complex(coefficient)))
            if len(matches) != 1 or not math.isclose(abs(matches[0][2]), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError('two-qubit gate is not a Pauli-permuting Clifford')
            out0, out1, coefficient = matches[0]
            if abs(coefficient.imag) < 1e-12:
                coefficient = complex(float(coefficient.real), 0.0)
            table[label0, label1] = (out0, out1, coefficient)
    return table
CZ_TABLE = _build_twoq_clifford_table(np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128))

def compile_circuit(circuit: QuantumCircuit) -> list[CompiledGate]:
    compiled: list[CompiledGate] = []
    for instruction in circuit.data:
        operation = instruction.operation
        qubits = tuple((circuit.find_bit(q).index for q in instruction.qubits))
        if operation.name in {'barrier', 'delay'}:
            continue
        if operation.num_clbits or operation.name in {'measure', 'reset'}:
            raise ValueError('Pauli marginal solver supports unitary circuits only')
        if len(qubits) == 1:
            matrix = np.asarray(Operator(operation).data, dtype=np.complex128)
            compiled.append(CompiledGate(kind='oneq', qubits=qubits, oneq_table=_oneq_transfer(matrix)))
        elif len(qubits) == 2 and operation.name == 'cz':
            compiled.append(CompiledGate(kind='cz', qubits=qubits))
        elif len(qubits) == 2 and operation.name == 'swap':
            compiled.append(CompiledGate(kind='swap', qubits=qubits))
        else:
            raise ValueError(f'unsupported operation {operation.name!r} on {len(qubits)} qubits; supported gates are arbitrary 1q, CZ, and SWAP')
    return compiled

def _mask_unions(terms: dict[tuple[int, int], complex]) -> tuple[int, int]:
    x_union = 0
    support_union = 0
    for xmask, zmask in terms:
        x_union |= xmask
        support_union |= xmask | zmask
    return (x_union, support_union)

def _prune_terms(terms: dict[tuple[int, int], complex], config: SolverConfig, stats: PropagationStats) -> dict[tuple[int, int], complex]:
    stats.prune_calls += 1
    stats.max_terms_before_prune = max(stats.max_terms_before_prune, len(terms))
    original_count = len(terms)
    terms = {key: value for key, value in terms.items() if abs(value) > config.coefficient_epsilon}
    if config.max_pauli_weight is not None:
        limit = config.max_pauli_weight
        terms = {key: value for key, value in terms.items() if (key[0] | key[1]).bit_count() <= limit}
    if terms and config.relative_threshold:
        largest = max((abs(value) for value in terms.values()))
        cutoff = config.relative_threshold * largest
        terms = {key: value for key, value in terms.items() if abs(value) >= cutoff}
    if config.top_k is not None and len(terms) > config.top_k:
        terms = dict(heapq.nlargest(config.top_k, terms.items(), key=lambda item: abs(item[1])))
    stats.terms_dropped += original_count - len(terms)
    return terms

def propagate_z_observable(compiled: Sequence[CompiledGate], qubit: int, config: SolverConfig) -> tuple[dict[tuple[int, int], complex], PropagationStats]:
    if qubit < 0:
        raise ValueError('qubit must be nonnegative')
    terms: dict[tuple[int, int], complex] = {(0, 1 << qubit): 1.0 + 0j}
    stats = PropagationStats()
    x_union, support_union = _mask_unions(terms)
    for gate in reversed(compiled):
        stats.gates_visited += 1
        if gate.kind == 'oneq':
            q = gate.qubits[0]
            bit = 1 << q
            if not support_union & bit:
                continue
            stats.gates_active += 1
            assert gate.oneq_table is not None
            expanded: dict[tuple[int, int], complex] = {}
            for (xmask, zmask), coefficient in terms.items():
                source = _label_from_masks(xmask, zmask, q)
                if source == 0:
                    key = (xmask, zmask)
                    expanded[key] = expanded.get(key, 0j) + coefficient
                    continue
                for target, factor in gate.oneq_table[source]:
                    next_key = _replace_label(xmask, zmask, q, target)
                    expanded[next_key] = expanded.get(next_key, 0j) + coefficient * factor
            terms = _prune_terms(expanded, config, stats)
        elif gate.kind == 'cz':
            q0, q1 = gate.qubits
            if not x_union & (1 << q0 | 1 << q1):
                continue
            stats.gates_active += 1
            transformed: dict[tuple[int, int], complex] = {}
            for (xmask, zmask), coefficient in terms.items():
                label0 = _label_from_masks(xmask, zmask, q0)
                label1 = _label_from_masks(xmask, zmask, q1)
                out0, out1, phase = CZ_TABLE[label0, label1]
                next_x, next_z = _replace_label(xmask, zmask, q0, out0)
                next_x, next_z = _replace_label(next_x, next_z, q1, out1)
                key = (next_x, next_z)
                transformed[key] = transformed.get(key, 0j) + coefficient * phase
            terms = transformed
            if config.max_pauli_weight is not None:
                terms = _prune_terms(terms, config, stats)
        elif gate.kind == 'swap':
            q0, q1 = gate.qubits
            bits = 1 << q0 | 1 << q1
            if not support_union & bits:
                continue
            stats.gates_active += 1
            transformed = {}
            for (xmask, zmask), coefficient in terms.items():
                label0 = _label_from_masks(xmask, zmask, q0)
                label1 = _label_from_masks(xmask, zmask, q1)
                next_x, next_z = _replace_label(xmask, zmask, q0, label1)
                next_x, next_z = _replace_label(next_x, next_z, q1, label0)
                transformed[next_x, next_z] = coefficient
            terms = transformed
            if config.max_pauli_weight is not None:
                terms = _prune_terms(terms, config, stats)
        else:
            raise AssertionError(f'unknown compiled gate kind {gate.kind}')
        if not terms:
            break
        x_union, support_union = _mask_unions(terms)
    return (terms, stats)

def expectation_on_zero(terms: dict[tuple[int, int], complex]) -> complex:
    return sum((coefficient for (xmask, _), coefficient in terms.items() if xmask == 0))

def solve_marginal(compiled: Sequence[CompiledGate], qubit: int, config: SolverConfig) -> dict[str, Any]:
    terms, stats = propagate_z_observable(compiled, qubit, config)
    expectation = expectation_on_zero(terms)
    real_expectation = float(expectation.real)
    clipped = float(np.clip(real_expectation, -1.0, 1.0))
    p0 = (1.0 + clipped) / 2.0
    p1 = (1.0 - clipped) / 2.0
    retained_l2 = float(sum((abs(value) ** 2 for value in terms.values())))
    return {'qubit': qubit, 'bit': 0 if real_expectation >= 0.0 else 1, 'z_expectation': real_expectation, 'z_expectation_imag': float(expectation.imag), 'z_expectation_clipped': clipped, 'p0_clipped': p0, 'p1_clipped': p1, 'confidence': abs(clipped), 'final_terms': len(terms), 'zero_state_contributing_terms': sum((1 for xmask, _ in terms if xmask == 0)), 'retained_pauli_l2': retained_l2, 'stats': asdict(stats)}

def solve_all_marginals(circuit: QuantumCircuit, config: SolverConfig, *, qubits: Iterable[int] | None=None) -> dict[str, Any]:
    selected = list(range(circuit.num_qubits)) if qubits is None else list(qubits)
    if sorted(selected) != selected or len(set(selected)) != len(selected):
        raise ValueError('qubits must be unique and sorted')
    if any((q < 0 or q >= circuit.num_qubits for q in selected)):
        raise ValueError('qubit selection is out of range')
    compiled = compile_circuit(circuit)
    started = time.perf_counter()
    marginals = [solve_marginal(compiled, qubit, config) for qubit in selected]
    elapsed = time.perf_counter() - started
    bits = ''.join((str(row['bit']) for row in marginals))
    zero_confidence_count = sum((row['confidence'] == 0.0 for row in marginals))
    empty_expansion_count = sum((row['final_terms'] == 0 for row in marginals))
    return {'config': asdict(config), 'num_qubits': circuit.num_qubits, 'operation_count': len(circuit.data), 'selected_qubits': selected, 'blind_bitstring_q0_to_qN': bits, 'mean_confidence': float(np.mean([row['confidence'] for row in marginals])), 'min_confidence': float(min((row['confidence'] for row in marginals))), 'mean_retained_pauli_l2': float(np.mean([row['retained_pauli_l2'] for row in marginals])), 'zero_confidence_count': zero_confidence_count, 'empty_expansion_count': empty_expansion_count, 'elapsed_seconds': elapsed, 'marginals': marginals}

def exact_z_expectations(circuit: QuantumCircuit) -> list[float]:
    state = Statevector.from_instruction(circuit)
    probabilities = np.abs(np.asarray(state.data)) ** 2
    result = []
    indices = np.arange(len(probabilities), dtype=np.uint64)
    for qubit in range(circuit.num_qubits):
        signs = 1.0 - 2.0 * (indices >> qubit & 1)
        result.append(float(np.dot(probabilities, signs)))
    return result

def _random_validation_circuit(num_qubits: int, depth: int, rng: np.random.Generator) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    for layer in range(depth):
        for qubit in range(num_qubits):
            theta, phi, lam = rng.uniform(-math.pi, math.pi, size=3)
            circuit.append(UGate(theta, phi, lam), [qubit])
        order = rng.permutation(num_qubits)
        for offset in range(layer % 2, num_qubits - 1, 2):
            circuit.cz(int(order[offset]), int(order[offset + 1]))
        if num_qubits >= 2 and layer % 3 == 1:
            q0, q1 = rng.choice(num_qubits, size=2, replace=False)
            circuit.swap(int(q0), int(q1))
    return circuit

def validate_small_random_circuits() -> dict[str, Any]:
    rng = np.random.default_rng(20260709)
    cases = []
    worst = 0.0
    for num_qubits, depth in ((2, 4), (3, 5), (4, 5), (5, 4), (6, 3)):
        circuit = _random_validation_circuit(num_qubits, depth, rng)
        exact = exact_z_expectations(circuit)
        approximate = solve_all_marginals(circuit, SolverConfig(top_k=None, relative_threshold=0.0))
        observed = [row['z_expectation'] for row in approximate['marginals']]
        error = float(np.max(np.abs(np.asarray(exact) - np.asarray(observed))))
        worst = max(worst, error)
        cases.append({'num_qubits': num_qubits, 'depth_parameter': depth, 'operations': len(circuit.data), 'max_abs_error': error, 'exact': exact, 'pauli': observed})
    passed = worst < 1e-10
    return {'passed': passed, 'tolerance': 1e-10, 'worst_error': worst, 'cases': cases}

def _parse_positive_ints(text: str) -> list[int]:
    values = [int(value) for value in text.split(',') if value.strip()]
    if not values or any((value <= 0 for value in values)):
        raise argparse.ArgumentTypeError('expected comma-separated positive integers')
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('K sweep contains duplicates')
    return values

def _parse_qubits(text: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(',') if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('qubits must be comma-separated integers') from exc
    if not values:
        raise argparse.ArgumentTypeError('qubit list is empty')
    if values != sorted(set(values)):
        raise argparse.ArgumentTypeError('qubits must be unique and sorted')
    return values

def _summary_line(result: dict[str, Any]) -> str:
    config = result['config']
    suffix = ''
    if 'hamming_distance_to_previous_k' in result:
        suffix += f" delta_prev={result['hamming_distance_to_previous_k']}"
    return f"K={config['top_k']} bits={result['blind_bitstring_q0_to_qN']} mean|Z|={result['mean_confidence']:.6g} min|Z|={result['min_confidence']:.6g} mean_l2={result['mean_retained_pauli_l2']:.6g} zero_Z={result['zero_confidence_count']} empty={result['empty_expansion_count']} wall={result['elapsed_seconds']:.3f}s{suffix}"

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('qasm', nargs='?')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--top-k', type=int, default=512)
    parser.add_argument('--k-sweep', type=_parse_positive_ints)
    parser.add_argument('--relative-threshold', type=float, default=0.0)
    parser.add_argument('--max-pauli-weight', type=int)
    parser.add_argument('--coefficient-epsilon', type=float, default=0.0)
    parser.add_argument('--qubits', type=_parse_qubits)
    parser.add_argument('--json-out')
    return parser

def main(argv: Sequence[str] | None=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.self_test:
        validation = validate_small_random_circuits()
        print('SELF_TEST ' + json.dumps(validation, indent=2))
        if not validation['passed']:
            return 1
        if args.qasm is None:
            return 0
    if args.qasm is None:
        raise SystemExit('qasm is required unless --self-test is used')
    if args.top_k <= 0:
        raise SystemExit('--top-k must be positive')
    circuit = QuantumCircuit.from_qasm_file(args.qasm)
    k_values = args.k_sweep or [args.top_k]
    sweep = []
    previous_bits: str | None = None
    for top_k in k_values:
        config = SolverConfig(top_k=top_k, relative_threshold=args.relative_threshold, max_pauli_weight=args.max_pauli_weight, coefficient_epsilon=args.coefficient_epsilon)
        result = solve_all_marginals(circuit, config, qubits=args.qubits)
        if previous_bits is not None:
            result['hamming_distance_to_previous_k'] = sum((a != b for a, b in zip(previous_bits, result['blind_bitstring_q0_to_qN'])))
        previous_bits = result['blind_bitstring_q0_to_qN']
        sweep.append(result)
        print(_summary_line(result))
    output = {'method': 'truncated Heisenberg Pauli paths', 'qasm': str(args.qasm), 'bit_order': 'q0_to_qN', 'target_policy': 'blind; post-hoc evaluation only via evaluate-candidates', 'sweep': sweep}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(output, indent=2) + '\n')
        print(f'wrote JSON: {args.json_out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
