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
from qiskit import QuantumCircuit
_NON_UNITARY = {'barrier', 'measure', 'delay'}

def asap_layers(circuit: QuantumCircuit) -> list[int]:
    layers: list[int] = []
    frontier = [0] * circuit.num_qubits
    for instruction in circuit.data:
        if instruction.operation.name in _NON_UNITARY:
            raise ValueError(f'asap_layers expects a unitary-only circuit; strip {instruction.operation.name!r} operations first')
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        layer = max((frontier[q] for q in qubits))
        layers.append(layer)
        for q in qubits:
            frontier[q] = layer + 1
    return layers

def consolidated_layer_spans(raw: QuantumCircuit, consolidated: QuantumCircuit) -> list[tuple[int, int, int]]:
    layers = asap_layers(raw)
    per_qubit: list[list[int]] = [[] for _ in range(raw.num_qubits)]
    for index, instruction in enumerate(raw.data):
        for q in instruction.qubits:
            per_qubit[raw.find_bit(q).index].append(index)
    pointer = [0] * raw.num_qubits
    spans: list[tuple[int, int, int]] = []
    consumed_total = 0
    for cons_index, instruction in enumerate(consolidated.data):
        sites = {consolidated.find_bit(q).index for q in instruction.qubits}
        consumed: list[int] = []
        progress = True
        while progress:
            progress = False
            for site in sites:
                while pointer[site] < len(per_qubit[site]):
                    raw_index = per_qubit[site][pointer[site]]
                    raw_qubits = {raw.find_bit(q).index for q in raw.data[raw_index].qubits}
                    if not raw_qubits <= sites:
                        break
                    if any((per_qubit[q][pointer[q]] != raw_index for q in raw_qubits)):
                        break
                    for q in raw_qubits:
                        pointer[q] += 1
                    consumed.append(raw_index)
                    progress = True
        if not consumed:
            raise ValueError(f'consolidated instruction {cons_index} consumed no raw operations; the circuits do not correspond')
        consumed_total += len(consumed)
        block_layers = [layers[i] for i in consumed]
        spans.append((min(block_layers), max(block_layers), len(consumed)))
    if consumed_total != len(raw.data):
        raise ValueError(f'provenance walk consumed {consumed_total} of {len(raw.data)} raw operations; the circuits do not correspond')
    return spans

def cluster_window_order(circuit: QuantumCircuit, spans: list[tuple[int, int, int]], *, after_layer: int, before_layer: int, strict: bool=False) -> dict:
    from collections import deque
    total = len(circuit.data)
    if len(spans) != total:
        raise ValueError('spans must describe the same circuit')
    if strict:
        touching = {index for index, (low, high, _) in enumerate(spans) if low > after_layer and high < before_layer}
    else:
        touching = {index for index, (low, high, _) in enumerate(spans) if high > after_layer and low < before_layer}
    if not touching:
        raise ValueError(f'no instruction touches raw layers ({after_layer}, {before_layer})')
    successors: list[list[int]] = [[] for _ in range(total)]
    predecessors: list[list[int]] = [[] for _ in range(total)]
    last_on_qubit: dict[int, int] = {}
    for index, instruction in enumerate(circuit.data):
        for qubit in instruction.qubits:
            qubit_index = circuit.find_bit(qubit).index
            previous = last_on_qubit.get(qubit_index)
            if previous is not None:
                successors[previous].append(index)
                predecessors[index].append(previous)
            last_on_qubit[qubit_index] = index

    def closure(seed: set[int], neighbors: list[list[int]]) -> set[int]:
        reached: set[int] = set()
        queue = deque(seed)
        while queue:
            node = queue.popleft()
            for neighbor in neighbors[node]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        return reached
    descendants = closure(touching, successors)
    ancestors = closure(touching, predecessors)
    sandwiched = (descendants & ancestors) - touching
    core = touching | sandwiched
    window_start, window_end = (min(core), max(core) + 1)
    before: list[int] = []
    after: list[int] = []
    for index in range(window_start, window_end):
        if index in core:
            continue
        if index in descendants:
            after.append(index)
        else:
            before.append(index)
    order = list(range(window_start)) + before + sorted(core) + after + list(range(window_end, total))
    core_start = window_start + len(before)
    return {'order': order, 'core_window': [core_start, core_start + len(core)], 'core_size': len(core), 'touching': len(touching), 'sandwiched': len(sandwiched), 'moved_before': len(before), 'moved_after': len(after)}

def layer_window_in_consolidated(spans: list[tuple[int, int, int]], *, after_layer: int, before_layer: int) -> dict:
    if before_layer <= after_layer:
        raise ValueError('before_layer must exceed after_layer')
    touching = [index for index, (low, high, _) in enumerate(spans) if high > after_layer and low < before_layer]
    if not touching:
        raise ValueError(f'no consolidated instruction touches raw layers ({after_layer}, {before_layer})')
    inside = [index for index, (low, high, _) in enumerate(spans) if low > after_layer and high < before_layer]
    start, end = (min(touching), max(touching) + 1)
    return {'gap_layer_bounds_exclusive': [after_layer, before_layer], 'absorb_window': [start, end], 'window_instructions': end - start, 'instructions_touching_gap': len(touching), 'instructions_strictly_inside_gap': len(inside), 'flanking_instructions_in_window': end - start - len(touching), 'suggested_center': (start + end) // 2}
