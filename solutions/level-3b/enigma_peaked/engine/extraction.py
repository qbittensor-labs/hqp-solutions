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

def _is_torch(array) -> bool:
    return type(array).__module__.startswith('torch')

def _site_arrays(mps):
    psi = mps.copy()
    psi.normalize()
    for name in ('canonicalize', 'canonize', 'right_canonize'):
        method = getattr(psi, name, None)
        if method is None:
            continue
        try:
            method(0) if name != 'right_canonize' else method()
            break
        except TypeError:
            method()
            break
    n = psi.L
    arrays = []
    for i in range(n):
        tensor = psi[i]
        physical = psi.site_ind(i)
        order = []
        if i > 0:
            order.append(psi.bond(i - 1, i))
        order.append(physical)
        if i < n - 1:
            order.append(psi.bond(i, i + 1))
        array = tensor.transpose(*order).data
        if i == 0:
            array = array.reshape((1,) + tuple(array.shape))
        if i == n - 1:
            array = array.reshape(tuple(array.shape) + (1,))
        arrays.append(array)
    return arrays

def beam_search(mps, beam: int=512, k: int=8):
    arrays = _site_arrays(mps)
    first = arrays[0]
    torch = None
    if _is_torch(first):
        import torch as torch_module
        torch = torch_module
        ones = torch.ones((1, 1), dtype=first.dtype, device=first.device)
    else:
        ones = np.ones((1, 1), dtype=first.dtype)

    def weight(vector) -> float:
        if torch is not None:
            return float((vector.conj() * vector).real.sum().item())
        return float((vector.conj() * vector).real.sum())
    beams = [('', ones)]
    for array in arrays:
        candidates = []
        for bits, vector in beams:
            for bit in (0, 1):
                next_vector = vector @ array[:, bit, :]
                candidates.append((weight(next_vector), bits + str(bit), next_vector))
        candidates.sort(key=lambda item: item[0], reverse=True)
        beams = [(bits, vector) for _, bits, vector in candidates[:beam]]
    results = [(bits, weight(vector)) for bits, vector in beams]
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:k]

def marginal_argmax(mps):
    arrays = _site_arrays(mps)
    bits = ''
    p0s = []
    for array in arrays:
        v0 = array[:, 0, :]
        v1 = array[:, 1, :]
        if _is_torch(array):
            n0 = float((v0.conj() * v0).real.sum().item())
            n1 = float((v1.conj() * v1).real.sum().item())
        else:
            n0 = float((v0.conj() * v0).real.sum())
            n1 = float((v1.conj() * v1).real.sum())
        total = n0 + n1 or 1.0
        p0 = n0 / total
        p0s.append(p0)
        bits += '0' if p0 >= 0.5 else '1'
    margins = [abs(p - 0.5) for p in p0s]
    return (bits, min(margins) if margins else 0.0, p0s)

def amp2(mps, site_bits: str) -> float:
    psi = mps.copy()
    psi.normalize()
    tn = psi.isel({psi.site_ind(i): int(b) for i, b in enumerate(site_bits)})
    amplitude = tn.contract(all, optimize='auto-hq')
    return float(abs(amplitude) ** 2)
