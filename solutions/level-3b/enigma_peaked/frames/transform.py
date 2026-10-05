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
from dataclasses import dataclass
from typing import Any
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from .permutation import Permutation, permutation_to_transpositions
from .schedule import FrameSchedule
from .state import FrameState

class FrameTransformError(ValueError):
    pass
_PASSTHROUGH = {'barrier', 'delay'}
_NONUNITARY = {'measure', 'reset'}

@dataclass(frozen=True)
class LayeredOperation:
    layer: int
    operation: Any
    qubits: tuple[int, ...]
    clbits: tuple[int, ...]

def layered_operations(circuit: QuantumCircuit) -> tuple[list[LayeredOperation], int]:
    layers = list(circuit_to_dag(circuit).layers())
    result: list[LayeredOperation] = []
    for layer_index, layer in enumerate(layers):
        for node in layer['graph'].op_nodes():
            result.append(LayeredOperation(layer=layer_index, operation=node.op, qubits=tuple((circuit.find_bit(q).index for q in node.qargs)), clbits=tuple((circuit.find_bit(c).index for c in node.cargs))))
    return (result, len(layers))

def _check_schedule(circuit: QuantumCircuit, schedule: FrameSchedule, depth: int) -> None:
    if schedule.num_qubits != circuit.num_qubits:
        raise FrameTransformError(f'schedule is for {schedule.num_qubits} qubits, circuit has {circuit.num_qubits}')
    if schedule.max_layer() > depth:
        raise FrameTransformError(f'schedule references layer {schedule.max_layer()} beyond circuit depth {depth}')

def _events_by_layer(schedule: FrameSchedule) -> dict[int, list]:
    grouped: dict[int, list] = {}
    for event in schedule.events:
        grouped.setdefault(event.layer, []).append(event)
    return grouped

def materialize_frame_circuit(circuit: QuantumCircuit, schedule: FrameSchedule) -> QuantumCircuit:
    operations, depth = layered_operations(circuit)
    _check_schedule(circuit, schedule, depth)
    events = _events_by_layer(schedule)
    materialized = QuantumCircuit(circuit.num_qubits, circuit.num_clbits, name=f'{circuit.name}_framed')
    active: Permutation | None = None

    def process_events(layer: int) -> None:
        nonlocal active
        for event in events.get(layer, ()):
            seam = event.seam_permutation()
            for site_a, site_b in permutation_to_transpositions(seam):
                materialized.swap(site_a, site_b)
            active = event.permutation if event.kind == 'enter' else None
    current_layer = -1
    for entry in operations:
        while current_layer < entry.layer:
            current_layer += 1
            process_events(current_layer)
        name = entry.operation.name
        if name in _NONUNITARY or entry.clbits:
            if active is not None:
                raise FrameTransformError(f'non-unitary operation {name!r} inside an open frame interval')
            materialized.append(entry.operation, [materialized.qubits[q] for q in entry.qubits], [materialized.clbits[c] for c in entry.clbits])
            continue
        if name in _PASSTHROUGH:
            qubits = entry.qubits
        elif active is not None:
            qubits = tuple((active[q] for q in entry.qubits))
        else:
            qubits = entry.qubits
        materialized.append(entry.operation, [materialized.qubits[q] for q in qubits])
    while current_layer < depth:
        current_layer += 1
        process_events(current_layer)
    if active is not None:
        raise FrameTransformError('schedule left a frame open past the final layer')
    return materialized

def virtual_frame_program(circuit: QuantumCircuit, schedule: FrameSchedule) -> tuple[list[dict[str, Any]], FrameState]:
    operations, depth = layered_operations(circuit)
    _check_schedule(circuit, schedule, depth)
    events = _events_by_layer(schedule)
    frame = FrameState.identity(circuit.num_qubits)
    program: list[dict[str, Any]] = []

    def process_events(layer: int) -> None:
        for event in events.get(layer, ()):
            before = frame.logical_to_site
            if event.kind == 'enter':
                frame.enter_seam(event.permutation)
            else:
                frame.exit_seam(event.permutation)
            after = frame.logical_to_site
            site_permutation = Permutation((after[Permutation(before).inverse()[s]] for s in range(len(before))))
            program.append({'kind': 'seam', 'module_id': event.module_id, 'event': event.kind, 'site_transpositions': permutation_to_transpositions(site_permutation), 'logical_to_site_before': list(before), 'logical_to_site_after': list(after)})
    current_layer = -1
    for entry in operations:
        while current_layer < entry.layer:
            current_layer += 1
            process_events(current_layer)
        name = entry.operation.name
        if (name in _NONUNITARY or entry.clbits) and (not frame.is_identity()):
            raise FrameTransformError(f'non-unitary operation {name!r} inside an open frame interval')
        program.append({'kind': 'op', 'operation': entry.operation, 'logical_qubits': list(entry.qubits), 'sites': [frame.site_of(q) for q in entry.qubits], 'clbits': list(entry.clbits)})
    while current_layer < depth:
        current_layer += 1
        process_events(current_layer)
    return (program, frame)

def program_to_circuit(num_qubits: int, program: list[dict[str, Any]], num_clbits: int=0):
    circuit = QuantumCircuit(num_qubits, num_clbits, name='virtual_frame_program')
    for entry in program:
        if entry['kind'] == 'seam':
            for site_a, site_b in entry['site_transpositions']:
                circuit.swap(site_a, site_b)
        elif entry['kind'] == 'op':
            circuit.append(entry['operation'], [circuit.qubits[s] for s in entry['sites']], [circuit.clbits[c] for c in entry['clbits']])
        else:
            raise FrameTransformError(f"unknown program entry kind {entry['kind']!r}")
    return circuit
