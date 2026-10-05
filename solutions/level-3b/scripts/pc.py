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
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import UGate
from qiskit.quantum_info import Operator
SOURCE_SHA = 'c09ba53721b60791bf888aaa3c161c5a2b7e310c30dfbcfdc79c26b475278354'
REGIONS = {'M0': (785, 2184), 'M1': (2405, 3816)}

@dataclass(frozen=True)
class _f0:
    uid: str
    name: str
    wires: tuple[int, ...]
    angles: tuple[float, ...] = ()

def _f1(circuit):
    gates = []
    for i, item in enumerate(circuit.data):
        if item.operation.name not in {'u', 'cz', 'z'}:
            raise ValueError('Only unitary u/cz/z input is supported')
        angles = tuple((float(x) for x in item.operation.params))
        if not all((np.isfinite(x) for x in angles)):
            raise ValueError('Nonfinite gate parameter')
        gates.append(_f0(str(i), item.operation.name, tuple((circuit.find_bit(q).index for q in item.qubits)), angles))
    return gates

def _f2(gates, n):
    circuit = QuantumCircuit(n)
    for g in gates:
        if g.name == 'u':
            circuit.u(*g.angles, g.wires[0])
        elif g.name == 'z':
            circuit.z(g.wires[0])
        elif g.name == 'cz':
            circuit.cz(*g.wires)
        else:
            raise ValueError('Unsupported gate')
    return circuit

def _f3(theta):
    quarter = int(np.rint(theta / np.pi))
    target = quarter * np.pi
    delta = theta - target
    return (target, quarter % 2, float(2 * abs(np.sin(delta / 4))))

def _f4(gates):
    previous = {}
    for j, g in enumerate(gates):
        if g.name != 'cz':
            continue
        pair = tuple(sorted(g.wires))
        if pair in previous:
            i = previous[pair]
            cost = sum((_f3(h.angles[0])[2] for h in gates[i + 1:j] if h.name == 'u' and h.wires[0] in pair))
            yield (cost, i, j)
        previous[pair] = j

def _f5(gates, i, j):
    first, last = (gates[i], gates[j])
    if first.name != 'cz' or last.name != 'cz' or set(first.wires) != set(last.wires):
        raise ValueError('Cancellation endpoints must be the same CZ')
    if i >= j:
        raise ValueError('Reversed cancellation')
    pair = first.wires
    middle, changes = ([], [])
    for g in gates[i + 1:j]:
        if g.name == 'u' and g.wires[0] in pair:
            target, odd, error = _f3(g.angles[0])
            adjusted = replace(g, angles=(target, *g.angles[1:]))
            middle.append(adjusted)
            partner = pair[1] if g.wires[0] == pair[0] else pair[0]
            if odd:
                middle.append(_f0(f'z:{first.uid}:{g.uid}', 'z', (partner,)))
            changes.append({'uid': g.uid, 'before': list(g.angles), 'after': list(adjusted.angles), 'wire': g.wires[0], 'z_byproduct_wire': partner if odd else None, 'operator_error_bound': error})
        else:
            middle.append(g)
    record = {'removed_cz_uids': [first.uid, last.uid], 'wires': list(pair), 'changes': changes, 'operator_error_bound': sum((c['operator_error_bound'] for c in changes))}
    return (gates[:i] + middle + gates[j + 1:], record)

def _f6(gates, budget):
    if not np.isfinite(budget) or budget < 0 or budget >= 2:
        raise ValueError('A finite, nonvacuous budget in [0,2) is required')
    current, records, spent = (list(gates), [], 0.0)
    while True:
        options = sorted(_f4(current))
        if not options or options[0][0] > budget - spent:
            break
        cost, i, j = options[0]
        current, record = _f5(current, i, j)
        spent += cost
        records.append(record)
    return (current, records)

def _f7(source, result, records):
    current = list(source)
    error_sum, worst_crossing, worst_bound_excess = (0.0, 0.0, 0.0)
    removed = set()
    for record in records:
        a, b = record['removed_cz_uids']
        if a in removed or b in removed:
            raise ValueError('Duplicate source consumption')
        ids = [g.uid for g in current]
        i, j = (ids.index(a), ids.index(b))
        updated, expected = _f5(current, i, j)
        if record != expected:
            raise ValueError('Certificate differs from source replay')
        for change in record['changes']:
            before = np.asarray(UGate(*change['before']).to_matrix(), dtype=np.complex128)
            after = np.asarray(UGate(*change['after']).to_matrix(), dtype=np.complex128)
            measured = float(np.linalg.norm(before - after, 2))
            worst_bound_excess = max(worst_bound_excess, measured - change['operator_error_bound'])
            left, right = (QuantumCircuit(2), QuantumCircuit(2))
            left.cz(0, 1)
            left.u(*change['after'], 0)
            right.u(*change['after'], 0)
            if change['z_byproduct_wire'] is not None:
                right.z(1)
            right.cz(0, 1)
            worst_crossing = max(worst_crossing, float(np.linalg.norm(Operator(left).data - Operator(right).data, 2)))
        current = updated
        removed.update((a, b))
        error_sum += record['operator_error_bound']
    if current != result or len({g.uid for g in result}) != len(result):
        raise ValueError('Result or ownership mismatch')
    if worst_crossing > 1e-12 or worst_bound_excess > 1e-12:
        raise ValueError('Local matrix audit failed')
    before = sum((g.name == 'cz' for g in source))
    after = sum((g.name == 'cz' for g in result))
    if before - after != len(removed):
        raise ValueError('CZ accounting mismatch')
    return {'source_cz': before, 'replacement_cz': after, 'removed_cz': len(removed), 'operator_error_bound': error_sum, 'max_crossing_matrix_residual': worst_crossing, 'max_local_bound_excess': worst_bound_excess, 'ownership_replay_passed': True}

def _f8(args):
    payload = Path(args.source).read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA:
        raise ValueError('Authoritative QASM identity mismatch')
    original = qasm2.loads(payload.decode(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    if original.num_qubits != 48 or len(original.data) != 4353:
        raise ValueError('Source dimensions mismatch')
    source = _f1(original)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    summary = {'source_sha256': SOURCE_SHA, 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scope': 'research-only analytic bounded approximation, no production substitution', 'arithmetic': 'complex128 local audits; analytic bounds, not outward-rounded intervals', 'per_block_budget': args.budget, 'regions': {}}
    replacements = {}
    for name, (lo, hi) in REGIONS.items():
        result, records = _f6(source[lo:hi], args.budget)
        checks = _f7(source[lo:hi], result, records)
        circuit = _f2(result, 48)
        text = qasm2.dumps(circuit) + '\n'
        restored = _f1(qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
        if [(g.name, g.wires, g.angles) for g in restored] != [(g.name, g.wires, g.angles) for g in result]:
            raise ValueError('Export changed circuit')
        (out / f'{name}.qasm').write_text(text)
        (out / f'{name}_certificate.json').write_text(json.dumps(records, indent=2) + '\n')
        summary['regions'][name] = {**checks, 'qasm_sha256': hashlib.sha256(text.encode()).hexdigest()}
        replacements[name] = result
    full = source[:785] + replacements['M0'] + source[2184:2405] + replacements['M1'] + source[3816:]
    full_text = qasm2.dumps(_f2(full, 48)) + '\n'
    (out / 's2_reduced.qasm').write_text(full_text)
    bound = sum((r['operator_error_bound'] for r in summary['regions'].values()))
    summary.update({'full_circuit_operator_error_bound': bound, 'full_circuit_any_event_probability_error_bound': min(1.0, bound), 'full_circuit_qasm_sha256': hashlib.sha256(full_text.encode()).hexdigest(), 'full_circuit_cz': sum((g.name == 'cz' for g in full)), 'unchanged_prefix_gap_suffix': True, 'status': 'complete'})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--budget', type=float, default=0.005)
    _f8(parser.parse_args())
