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
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import ElidePermutations, SabreSwap
NON_WORK = {'measure', 'barrier', 'delay', 'swap'}

def make_rewire_layers(horizon: int, iter_layers, merge_layers):

    def rewire_layers(layers, perm, seed=None, sabre_trials: int=200):
        num_qubits = len(perm)
        circuit = merge_layers(layers)
        circuit = QuantumCircuit(num_qubits, circuit.num_clbits).compose(circuit, qubits=np.argsort(perm).tolist())
        circuit = ElidePermutations()(circuit)
        coupling = CouplingMap.from_line(num_qubits)
        cut, work = (None, 0)
        for index, instruction in enumerate(circuit.data):
            if instruction.operation.name not in NON_WORK:
                work += 1
                if work == horizon:
                    cut = index + 1
                    break
        if cut is None or not any((ins.operation.name not in NON_WORK for ins in circuit.data[cut:])):
            return list(iter_layers(SabreSwap(coupling_map=coupling, heuristic='decay', trials=sabre_trials, seed=seed)(circuit)))
        head, tail = (circuit.copy_empty_like(), circuit.copy_empty_like())
        for instruction in circuit.data[:cut]:
            head.append(instruction)
        for instruction in circuit.data[cut:]:
            tail.append(instruction)
        head_pass = SabreSwap(coupling_map=coupling, heuristic='decay', trials=sabre_trials, seed=seed)
        head_routed = head_pass(head)
        final = head_pass.property_set['final_layout']
        placement = [final[head.qubits[wire]] for wire in range(num_qubits)] if final is not None else list(range(num_qubits))
        placed_tail = QuantumCircuit(num_qubits, circuit.num_clbits).compose(tail, qubits=placement)
        tail_routed = SabreSwap(coupling_map=coupling, heuristic='decay', trials=sabre_trials, seed=seed)(placed_tail)
        return list(iter_layers(head_routed.compose(tail_routed)))
    return rewire_layers
