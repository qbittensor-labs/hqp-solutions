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

"""Peak extraction + confidence for a final MPS (quimb MatrixProductState).

The milestone-1 winner's decisive edge was ARGMAX extraction (not sampling) plus
a confidence signal. This module provides, backend-agnostic (numpy or torch):

  - beam_search(mps, beam, k): canonical beam search. Right-canonicalize so the
    left partial-norm of a prefix == its true marginal probability, then walk the
    most-probable bitstrings. Recovers a ~0.1-weight peak that sampling misses.
  - marginal_argmax(mps): cheap per-site marginal argmax (independent-bit vote).
  - amp2(mps, bits): exact |<bits|psi>|^2 for a confidence gate.

All return bitstrings in MPS-SITE order; the caller remaps to logical order.
"""
import numpy as np


def _is_torch(a):
    return type(a).__module__.startswith("torch")


def _site_arrays(mps):
    """Right-canonicalize and return per-site arrays as (Dl, 2, Dr) with singleton
    boundary bonds. After canonicalizing the center to site 0, sites 1..n-1 are
    right-orthonormal, so the left partial-norm equals the prefix marginal."""
    psi = mps.copy()
    psi.normalize()
    # center at site 0 -> right-canonical for the rest
    for nm in ("canonicalize", "canonize", "right_canonize"):
        fn = getattr(psi, nm, None)
        if fn is None:
            continue
        try:
            fn(0) if nm != "right_canonize" else fn()
            break
        except TypeError:
            fn()
            break
    n = psi.L
    arrs = []
    for i in range(n):
        t = psi[i]
        p = psi.site_ind(i)
        order = []
        if i > 0:
            order.append(psi.bond(i - 1, i))
        order.append(p)
        if i < n - 1:
            order.append(psi.bond(i, i + 1))
        a = t.transpose(*order).data
        if i == 0:
            a = a.reshape((1,) + tuple(a.shape))
        if i == n - 1:
            a = a.reshape(tuple(a.shape) + (1,))
        arrs.append(a)
    return arrs


def beam_search(mps, beam=512, k=8):
    """Canonical beam search. Returns [(bitstring_site_order, prob), ...] top-k."""
    arrs = _site_arrays(mps)
    a0 = arrs[0]
    torch = None
    if _is_torch(a0):
        import torch as _t
        torch = _t
        ones = torch.ones((1, 1), dtype=a0.dtype, device=a0.device)
    else:
        ones = np.ones((1, 1), dtype=a0.dtype)

    def w(v):  # squared norm
        if torch is not None:
            return float((v.conj() * v).real.sum().item())
        return float((v.conj() * v).real.sum())

    beams = [("", ones)]
    for a in arrs:
        cand = []
        for bits, vec in beams:
            for b in (0, 1):
                nv = vec @ a[:, b, :]
                cand.append((w(nv), bits + str(b), nv))
        cand.sort(key=lambda t: t[0], reverse=True)
        beams = [(bits, nv) for (_, bits, nv) in cand[:beam]]
    res = [(bits, w(nv)) for bits, nv in beams]
    res.sort(key=lambda t: t[1], reverse=True)
    return res[:k]


def marginal_argmax(mps):
    """Per-site marginal argmax. Returns (bitstring_site_order, min_margin, p0s)."""
    arrs = _site_arrays(mps)
    bits = ""
    p0s = []
    for a in arrs:
        # marginal P(site=0) = ||left-canonical prefix restricted to bit 0||^2,
        # but locally on a right-canonical site the bond envs are identity, so the
        # per-site reduced prob is sum over the bit slice. Use the orthonormal form:
        v0 = a[:, 0, :]
        v1 = a[:, 1, :]
        if _is_torch(a):
            n0 = float((v0.conj() * v0).real.sum().item())
            n1 = float((v1.conj() * v1).real.sum().item())
        else:
            n0 = float((v0.conj() * v0).real.sum())
            n1 = float((v1.conj() * v1).real.sum())
        tot = n0 + n1 or 1.0
        p0 = n0 / tot
        p0s.append(p0)
        bits += "0" if p0 >= 0.5 else "1"
    margins = [abs(p - 0.5) for p in p0s]
    return bits, (min(margins) if margins else 0.0), p0s


def amp2(mps, bits):
    """Exact |<bits|psi>|^2, bits in MPS-site order."""
    psi = mps.copy()
    psi.normalize()
    tn = psi.isel({psi.site_ind(i): int(b) for i, b in enumerate(bits)})
    amp = tn.contract(all, optimize="auto-hq")
    return float(abs(amp) ** 2)
