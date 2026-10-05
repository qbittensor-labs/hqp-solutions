# Copyright (C) 2026 qBitTensor Labs.
# Original author: Charlie (Enigma / Hardening Quantum Proof competition).
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

"""Optimized GPU Heisenberg Pauli-path propagator (5-10x faster than pauli_prop).

Speedups: precomputed SO(3) matrices per gate, complex64 coeffs, less-frequent
full merge (topk-prune between merges), minimal per-gate allocations. Supports
arbitrary Z-type starting observables (single Z_i and products Z_S) so the same
engine gives marginals AND pairwise correlators. Validated against pauli_prop.
"""
import time
import numpy as np
import torch

torch.set_grad_enabled(False)
import os as _os
_dev = _os.environ.get("PPS_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"
_CD = torch.complex64


def u3_so3(U):
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    S = [X, Y, Z]
    Ud = U.conj().T
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = 0.5 * np.trace(S[i] @ Ud @ S[j] @ U).real
    return R


def compile_gates(gates):
    """Replace ('u3',q,U) with ('u3',q,R_tensor). CZ unchanged. Precompute once."""
    out = []
    for g in gates:
        if g[0] == 'u3':
            R = u3_so3(np.asarray(g[2]))
            out.append(('u3', g[1], torch.tensor(R, dtype=torch.float32, device=_dev)))
        else:
            out.append(g)
    return out


class PS:
    def __init__(self, x, z, c):
        self.x = x; self.z = z; self.c = c

    @classmethod
    def z_string(cls, qubits):
        zz = 0
        for q in qubits:
            zz |= (1 << q)
        return cls(torch.zeros(1, dtype=torch.int64, device=_dev),
                   torch.tensor([zz], dtype=torch.int64, device=_dev),
                   torch.ones(1, dtype=_CD, device=_dev))

    def size(self):
        return self.x.numel()

    def merge(self):
        z = self.z; x = self.x; c = self.c
        idx = torch.argsort(z)
        x, z, c = x[idx], z[idx], c[idx]
        idx2 = torch.argsort(x, stable=True)
        x, z, c = x[idx2], z[idx2], c[idx2]
        newseg = torch.ones(x.numel(), dtype=torch.bool, device=_dev)
        newseg[1:] = (x[1:] != x[:-1]) | (z[1:] != z[:-1])
        seg = torch.cumsum(newseg.to(torch.int64), 0) - 1
        m = int(seg[-1]) + 1 if x.numel() else 0
        cs = torch.zeros(m, dtype=_CD, device=_dev).index_add_(0, seg, c)
        first = torch.nonzero(newseg, as_tuple=True)[0]
        self.x, self.z, self.c = x[first], z[first], cs

    def prune(self, K):
        if self.x.numel() > K:
            top = torch.topk(self.c.abs(), K).indices
            self.x, self.z, self.c = self.x[top], self.z[top], self.c[top]

    def cz(self, a, b):
        xa = (self.x >> a) & 1
        xb = (self.x >> b) & 1
        za = (self.z >> a) & 1
        zb = (self.z >> b) & 1
        sign = 1 - 2 * (xa & xb & (za ^ zb))
        self.c = self.c * sign.to(_CD)
        self.z = self.z ^ (xb << a) ^ (xa << b)

    def u3(self, q, R):
        xq = (self.x >> q) & 1
        zq = (self.z >> q) & 1
        fac = xq + 2 * zq                      # 0=I,1=X,2=Z,3=Y
        isI = fac == 0
        outs = []
        if isI.any():
            outs.append((self.x[isI], self.z[isI], self.c[isI]))
        nonI = ~isI
        if nonI.any():
            xn = self.x[nonI]; zn = self.z[nonI]; cn = self.c[nonI]; fn = fac[nonI]
            xb0 = xn & ~(1 << q); zb0 = zn & ~(1 << q)
            j = torch.where(fn == 1, 0, torch.where(fn == 2, 2, 1))  # src X=0,Z=2,Y=1
            for i, (xt, zt) in enumerate([(1, 0), (1, 1), (0, 1)]):   # tgt X,Y,Z
                coef = R[i][j]
                nz = coef != 0
                if not bool(nz.any()):
                    continue
                outs.append((xb0[nz] | (xt << q),
                             zb0[nz] | (zt << q),
                             cn[nz] * coef[nz].to(_CD)))
        self.x = torch.cat([o[0] for o in outs])
        self.z = torch.cat([o[1] for o in outs])
        self.c = torch.cat([o[2] for o in outs])

    def expect0(self):
        m = self.x == 0
        return complex(self.c[m].sum().item()) if bool(m.any()) else 0j


def evolve(cgates, obs_qubits, K=2_000_000, merge_every=40, log=None, deadline=None):
    """cgates: compiled gates (u3 with R tensor). obs_qubits: list -> product Z_S observable."""
    ps = PS.z_string(obs_qubits)
    t0 = time.time(); L = len(cgates); sub = 0
    for gi in range(L - 1, -1, -1):
        g = cgates[gi]
        if g[0] == 'cz':
            ps.cz(g[1], g[2])
        else:
            ps.u3(g[1], g[2]); sub += 1
            if ps.size() > K * 3 // 2:
                ps.merge(); ps.prune(K)
            if sub % merge_every == 0:
                ps.merge(); ps.prune(K)
        if deadline and (gi & 255) == 0 and time.time() > deadline:
            if log: log(f"  deadline @gate {gi}")
            break
    ps.merge()
    return ps.expect0()
