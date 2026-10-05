# Copyright (C) 2026 qBitTensor Labs.
# Original author: an anonymous competition participant (Enigma / Hardening Quantum Proof competition).
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

"""Robust QASM loader + converters for HQP peaked circuits.

Loads OpenQASM 2.0/3.0 (the BlueQubit circuits use `u` + `cz`), exposes the
gate list, and builds a quimb tensor-network Circuit for contraction-cost
analysis and tensor-network simulation.
"""
import re
from pathlib import Path


def load_qiskit(qasm_file):
    """Load a QASM circuit with qiskit, handling 2.0 and 3.0 + builtin `u`."""
    from qiskit import qasm2
    from qiskit.circuit.library import UGate

    header = Path(qasm_file).read_text().splitlines()[0].strip()
    if "3.0" in header:
        import qiskit.qasm3 as qasm3
        return qasm3.load(qasm_file)
    custom = [
        qasm2.CustomInstruction("u", 3, 1, lambda t, p, lam: UGate(t, p, lam), builtin=True),
    ]
    return qasm2.load(qasm_file, custom_instructions=custom)


def gate_list(qc):
    """Return [(name, params, [qubit_indices]), ...] from a qiskit circuit."""
    out = []
    index = {q: i for i, q in enumerate(qc.qubits)}
    for inst in qc.data:
        op = inst.operation
        qs = [index[q] for q in inst.qubits]
        params = [float(p) for p in op.params]
        out.append((op.name, params, qs))
    return out


def build_quimb(qc, **circ_opts):
    """Build a quimb tensor-network Circuit from a qiskit circuit."""
    import quimb.tensor as qtn

    n = qc.num_qubits
    circ = qtn.Circuit(n, **circ_opts)
    for name, params, qs in gate_list(qc):
        nm = name.lower()
        if nm in ("u", "u3"):
            circ.apply_gate("U3", params[0], params[1], params[2], qs[0])
        elif nm == "cz":
            circ.apply_gate("CZ", qs[0], qs[1])
        elif nm == "cx":
            circ.apply_gate("CX", qs[0], qs[1])
        elif nm == "x":
            circ.apply_gate("X", qs[0])
        elif nm == "h":
            circ.apply_gate("H", qs[0])
        elif nm in ("rz", "rx", "ry"):
            circ.apply_gate(nm.upper(), params[0], qs[0])
        else:
            raise ValueError(f"unhandled gate {name}")
    return circ


def load_meta(meta_file):
    import json
    return json.loads(Path(meta_file).read_text())
