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

def make_local_apply(budget: int):
    from qiskit.quantum_info import Operator
    import enigma_peaked.engine.telemetry as telemetry
    from src.peaked.local_chain import LocalChain

    def local_apply(mpo, circuit, *, side, max_bond=None, cutoff=0.0, compress_method='zipup', equalize_norms=False, to_backend=None):
        probe = mpo[0].data
        if type(probe).__module__.startswith('torch'):
            import torch
            xp = torch

            def to_matrix(op, k):
                return torch.as_tensor(np.asarray(Operator(op).data), dtype=probe.dtype, device=probe.device)
        else:
            xp = np

            def to_matrix(op, k):
                return np.asarray(Operator(op).data, dtype=probe.dtype)
        cap = min(int(budget), int(max_bond)) if max_bond else int(budget)
        chain = LocalChain(LocalChain.arrays_from_mpo(mpo), xp=xp, cutoff=cutoff, max_bond=cap, record=telemetry._record, decomp='gram')
        chain.canonicalize(0)
        chain.apply_circuit(circuit, side, to_matrix)
        out = LocalChain.mpo_from_arrays(chain.a, like=mpo)
        try:
            phase = float(circuit.global_phase)
        except (TypeError, ValueError):
            phase = 0.0
        if phase:
            factor = np.exp(1j * phase)
            out[0].modify(data=out[0].data * (factor if xp is np else xp.tensor(factor, dtype=probe.dtype, device=probe.device)))
        return out
    return local_apply
