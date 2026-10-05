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
from functools import lru_cache
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from quimb.tensor import Circuit, CircuitMPS, MatrixProductOperator, tensor_network_1d_compress
_SKIP_OPS = {'barrier', 'measure', 'delay'}

def quimb_circuit_from_qiskit(circuit: QuantumCircuit, quimb_circuit_class=Circuit, to_backend=None):
    kwargs = {}
    if to_backend is not None:
        kwargs['to_backend'] = to_backend
    quimb_circ = quimb_circuit_class(circuit.num_qubits, **kwargs)
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in _SKIP_OPS:
            continue
        where = [circuit.find_bit(q).index for q in instruction.qubits]
        if name == 'swap':
            quimb_circ.apply_gate('SWAP', *where)
            continue
        matrix = Operator(instruction.operation).data
        if len(where) == 2:
            quimb_circ.apply_gate_raw(matrix, list(reversed(where)), contract='split-gate')
        else:
            quimb_circ.apply_gate_raw(matrix, list(reversed(where)))
    return quimb_circ

def mps_zero_state(num_qubits: int, to_backend=None):
    return quimb_circuit_from_qiskit(QuantumCircuit(num_qubits), quimb_circuit_class=CircuitMPS, to_backend=to_backend).psi

def mpo_from_circuit(circ) -> MatrixProductOperator:
    for qubit in range(circ.N):
        circ.u3(0, 0, 0, qubit)
    tn_uni = circ.get_uni()
    for site_tag in list(tn_uni.site_tags):
        tn_uni ^= site_tag
    tn_uni.fuse_multibonds_()
    mpo = tn_uni.view_as_(MatrixProductOperator, cyclic=False, L=circ.N)
    mpo.ensure_bonds_exist()
    return mpo

def strict_chain_local_gate_mpo(num_sites: int, local_dim: int, where, matrix, *, to_backend=None) -> MatrixProductOperator:
    if isinstance(local_dim, bool) or not isinstance(local_dim, int) or local_dim < 2:
        raise ValueError('local_dim must be an integer greater than one')
    sites = tuple((int(site) for site in where))
    if len(sites) not in {1, 2}:
        raise ValueError('strict-chain gate MPO supports only one- and two-site gates')
    if len(set(sites)) != len(sites):
        raise ValueError('gate sites must be distinct')
    if any((site < 0 or site >= num_sites for site in sites)):
        raise ValueError(f'gate sites {sites} outside 0..{num_sites - 1}')
    gate = np.asarray(matrix, dtype=np.complex128)
    identity = np.eye(local_dim, dtype=gate.dtype)
    if len(sites) == 1:
        expected = (local_dim, local_dim)
        if gate.shape != expected:
            raise ValueError(f'one-site gate matrix must be {local_dim}x{local_dim}, got {gate.shape}')
        site_a = site_b = sites[0]
        rank = 1
        left_ops = gate.reshape(1, local_dim, local_dim)
        right_ops = None
    else:
        combined_dim = local_dim ** 2
        expected = (combined_dim, combined_dim)
        if gate.shape != expected:
            raise ValueError(f'two-site gate matrix must be {combined_dim}x{combined_dim}, got {gate.shape}')
        site_a, site_b = sites
        if site_a > site_b:
            site_a, site_b = (site_b, site_a)
            gate = gate.reshape(local_dim, local_dim, local_dim, local_dim).transpose(1, 0, 3, 2).reshape(combined_dim, combined_dim)
        schmidt = gate.reshape(local_dim, local_dim, local_dim, local_dim).transpose(0, 2, 1, 3).reshape(combined_dim, combined_dim)
        left, values, right = np.linalg.svd(schmidt, full_matrices=False)
        tolerance = np.finfo(values.dtype).eps * max(schmidt.shape) * values[0]
        keep = values > tolerance
        values = values[keep]
        root = np.sqrt(values)
        left_ops = (left[:, keep] * root[None, :]).T.reshape(-1, local_dim, local_dim)
        right_ops = (root[:, None] * right[keep, :]).reshape(-1, local_dim, local_dim)
        rank = len(values)
    arrays = []
    for site in range(num_sites):
        if site < site_a or site > site_b:
            array = identity.reshape(1, 1, local_dim, local_dim)
        elif len(sites) == 1:
            array = left_ops.reshape(1, 1, local_dim, local_dim)
        elif site == site_a:
            array = left_ops.reshape(1, rank, local_dim, local_dim)
        elif site == site_b:
            array = right_ops.reshape(rank, 1, local_dim, local_dim)
        else:
            array = np.zeros((rank, rank, local_dim, local_dim), dtype=gate.dtype)
            for label in range(rank):
                array[label, label] = identity
        if num_sites == 1:
            array = array[0, 0]
        elif site == 0:
            array = array[0]
        elif site == num_sites - 1:
            array = array[:, 0]
        arrays.append(to_backend(array) if to_backend is not None else array)
    return MatrixProductOperator(arrays, shape='lrud')

def strict_chain_gate_mpo(num_sites: int, where, matrix, *, to_backend=None) -> MatrixProductOperator:
    return strict_chain_local_gate_mpo(num_sites, 2, where, matrix, to_backend=to_backend)

def strict_chain_circuit_mpo(circuit: QuantumCircuit, *, max_bond: int | None=None, cutoff: float=0.0, compress_method: str='zipup', equalize_norms: bool=False, to_backend=None) -> MatrixProductOperator:
    mpo = strict_chain_gate_mpo(circuit.num_qubits, (0,), np.eye(2, dtype=np.complex128), to_backend=to_backend)
    for instruction in circuit.data:
        if instruction.operation.name in _SKIP_OPS:
            continue
        where = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if len(where) not in {1, 2}:
            raise ValueError(f'strict-chain mode cannot apply {len(where)}-site operation {instruction.operation.name!r}')
        gate = strict_chain_gate_mpo(circuit.num_qubits, reversed(where), Operator(instruction.operation).data, to_backend=to_backend)
        mpo = apply_mpo(mpo, gate, side='left', max_bond=max_bond, cutoff=cutoff, compress_method=compress_method, equalize_norms=equalize_norms)
    try:
        phase_angle = float(circuit.global_phase)
    except (TypeError, ValueError) as exc:
        raise ValueError('circuit global phase must be numerically bound') from exc
    if phase_angle:
        phase = strict_chain_gate_mpo(circuit.num_qubits, (0,), np.exp(1j * phase_angle) * np.eye(2, dtype=np.complex128), to_backend=to_backend)
        mpo = apply_mpo(mpo, phase, side='left', max_bond=max_bond, cutoff=cutoff, compress_method=compress_method, equalize_norms=equalize_norms)
    return mpo

def apply_qiskit_circuit_strict_chain(mpo: MatrixProductOperator, circuit: QuantumCircuit, *, side: str, max_bond: int | None=None, cutoff: float=0.0, compress_method: str='zipup', equalize_norms: bool=False, to_backend=None) -> MatrixProductOperator:
    if side not in {'left', 'right'}:
        raise ValueError("side must be 'left' or 'right'")
    instructions = [instruction for instruction in circuit.data if instruction.operation.name not in _SKIP_OPS]
    if side == 'right':
        instructions.reverse()
    result = mpo
    for instruction in instructions:
        where = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if len(where) not in {1, 2}:
            raise ValueError(f'strict-chain mode cannot apply {len(where)}-site operation {instruction.operation.name!r}')
        gate = strict_chain_gate_mpo(circuit.num_qubits, reversed(where), Operator(instruction.operation).data, to_backend=to_backend)
        result = apply_mpo(result, gate, side=side, max_bond=max_bond, cutoff=cutoff, compress_method=compress_method, equalize_norms=equalize_norms)
    try:
        phase_angle = float(circuit.global_phase)
    except (TypeError, ValueError) as exc:
        raise ValueError('circuit global phase must be numerically bound') from exc
    if phase_angle:
        phase = strict_chain_gate_mpo(circuit.num_qubits, (0,), np.exp(1j * phase_angle) * np.eye(2, dtype=np.complex128), to_backend=to_backend)
        result = apply_mpo(result, phase, side=side, max_bond=max_bond, cutoff=cutoff, compress_method=compress_method, equalize_norms=equalize_norms)
    return result

def apply_mpo(mpo1: MatrixProductOperator, mpo2: MatrixProductOperator, side: str, max_bond: int | None=None, cutoff: float=0.0, compress: bool=True, compress_method: str='zipup', equalize_norms: bool=False) -> MatrixProductOperator:
    if side == 'right':
        product = mpo1.apply(mpo2, compress=False, contract=True)
        length = len(mpo1.sites)
    elif side == 'left':
        product = mpo2.apply(mpo1, compress=False, contract=True)
        length = len(mpo2.sites)
    else:
        raise ValueError("side must be 'left' or 'right'")
    if not compress:
        return product
    out = tensor_network_1d_compress(product, max_bond=max_bond, cutoff=cutoff, method=compress_method, optimize='auto-hq', permute_arrays=False, equalize_norms=equalize_norms, inplace=True)
    if not isinstance(out, MatrixProductOperator):
        out = out.view_as_(MatrixProductOperator, cyclic=False, L=length)
    out.ensure_bonds_exist()
    return out

def apply_circuit(mpo: MatrixProductOperator, circ, side: str, max_bond: int | None=None, cutoff: float=0.0, compress_method: str='zipup', equalize_norms: bool=False) -> MatrixProductOperator:
    return apply_mpo(mpo, mpo_from_circuit(circ), side=side, max_bond=max_bond, cutoff=cutoff, compress_method=compress_method, equalize_norms=equalize_norms)
_SWAP_BACKENDS: dict[int, object] = {}

@lru_cache(maxsize=8192)
def _swap_mpo_cached(num_qubits: int, swaps: tuple, invert: bool, backend_key: int | None):
    circuit = QuantumCircuit(num_qubits)
    for q0, q1 in swaps:
        circuit.swap(q0, q1)
    circuit = circuit.inverse() if invert else circuit
    circuit = circuit.decompose('swap')
    circ = quimb_circuit_from_qiskit(circuit, to_backend=_SWAP_BACKENDS.get(backend_key))
    return mpo_from_circuit(circ)

def _swap_mpo(num_qubits: int, swaps, invert: bool, to_backend):
    backend_key = None
    if to_backend is not None:
        backend_key = id(to_backend)
        _SWAP_BACKENDS[backend_key] = to_backend
    return _swap_mpo_cached(num_qubits, tuple((tuple(pair) for pair in swaps)), invert, backend_key).copy()

def apply_swaps(mpo: MatrixProductOperator, swaps_left, swaps_right, max_bond: int | None=None, cutoff: float=0.0, to_backend=None, compress_method: str='zipup', equalize_norms: bool=False) -> MatrixProductOperator:
    num_qubits = len(mpo.sites)
    if not swaps_left and (not swaps_right):
        return mpo
    mpo_out = mpo
    if swaps_left:
        swap_mpo = _swap_mpo(num_qubits, swaps_left, invert=True, to_backend=to_backend)
        mpo_out = apply_mpo(mpo_out, swap_mpo, side='right', max_bond=max_bond, cutoff=cutoff, compress_method=compress_method, equalize_norms=equalize_norms)
    if swaps_right:
        swap_mpo = _swap_mpo(num_qubits, swaps_right, invert=False, to_backend=to_backend)
        mpo_out = apply_mpo(mpo_out, swap_mpo, side='left', max_bond=max_bond, cutoff=cutoff, compress_method=compress_method, equalize_norms=equalize_norms)
    return mpo_out
