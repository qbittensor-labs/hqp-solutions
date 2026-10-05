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
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

def iter_layers(circuit: QuantumCircuit):
    for layer in circuit_to_dag(circuit).layers():
        yield dag_to_circuit(layer['graph'])

def merge_layers(layers, barrier: bool=False) -> QuantumCircuit:
    iterator = iter(layers)
    merged = next(iterator)
    for layer in iterator:
        if barrier:
            merged.barrier()
        merged = merged.compose(layer)
    return merged

def merge_instructions(circuit: QuantumCircuit, start: int, stop: int) -> QuantumCircuit:
    merged = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data[start:stop]:
        if instruction.clbits:
            raise ValueError('merge_instructions expects a measurement-free circuit; strip final measurements before splitting')
        merged.append(instruction.operation, qargs=[circuit.find_bit(q).index for q in instruction.qubits])
    return merged

def elem_counts(tn) -> int:
    return sum((int(np.prod(t.shape)) for t in tn))

def get_tn_info(tn) -> dict[str, int]:
    shape_lengths = [len(t.shape) for t in tn]
    element_counts = [int(np.prod(t.shape)) for t in tn]
    return {'max_bond': int(tn.max_bond()), 'max_links': max(shape_lengths), 'total_elems': sum(element_counts), 'num_tensors': int(tn.num_tensors)}
