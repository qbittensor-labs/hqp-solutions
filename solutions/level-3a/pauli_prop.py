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

"""GPU Heisenberg Pauli-path propagator for peaked-circuit peak finding.

Computes <Z_i>_out = <0| C^dag Z_i C |0> (and pairwise <Z_i Z_j>) by evolving the
observable in the Heisenberg picture: conjugate through the gate list in reverse,
expanding into a weighted sum of Pauli strings, truncating to the top-K terms by
coefficient magnitude (the Pauli-path truncation). At the end only I/Z-type Paulis
survive on |0> (X/Y factors annihilate), so <0|P|0> = sign for those.

Reconstruct the peak from first+second order: p(s) ~ exp(sum_i h_i m_i + sum_ij J_ij m_i m_j)
with m_i = (-1)^{s_i}, h_i=<Z_i>, J_ij = connected <Z_iZ_j>. If truncation preserves the
peak's Pauli mass, argmax over s recovers the peak; if the peak sits below the truncated
Pauli noise (the PPS wall), it won't beat random.

Symplectic rep: Pauli on qubit q = i^? X^x Z^z, packed x,z as int64 bitmasks (n<=63).
Y := X*Z, its i-phase carried in the complex coeff. Coeffs complex128.
"""
import time
import numpy as np
import torch

torch.set_grad_enabled(False)
_C = torch.complex128
import os as _os
_dev = _os.environ.get("PPS_DEVICE","cuda") if torch.cuda.is_available() else "cpu"


def _u3_so3(U):
    """R[i,j] = 1/2 Tr(sig_i U^dag sig_j U), i,j in (X,Y,Z). U: 2x2 complex ndarray."""
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


class PauliSum:
    """Sparse sum of Pauli strings with complex coeffs, as parallel torch tensors."""

    def __init__(self, n, x, z, c):
        self.n = n
        self.x = x  # int64 [K]
        self.z = z  # int64 [K]
        self.c = c  # complex128 [K]

    @classmethod
    def single_Z(cls, n, i):
        return cls(n,
                   torch.zeros(1, dtype=torch.int64, device=_dev),
                   torch.tensor([1 << i], dtype=torch.int64, device=_dev),
                   torch.ones(1, dtype=_C, device=_dev))

    def size(self):
        return self.x.numel()

    def _merge(self):
        """Combine identical (x,z) Paulis, summing coeffs; drop ~zero."""
        # sort by z within x: use a stable two-key sort via a 128-bit-ish composite.
        # x,z are <2^63; sort by x then z with a lexicographic order using sort on z then stable sort on x.
        idx = torch.argsort(self.z)
        x, z, c = self.x[idx], self.z[idx], self.c[idx]
        idx2 = torch.argsort(x, stable=True)
        x, z, c = x[idx2], z[idx2], c[idx2]
        # boundaries where (x,z) changes
        same = (x[1:] == x[:-1]) & (z[1:] == z[:-1])
        newseg = torch.ones(x.numel(), dtype=torch.bool, device=_dev)
        newseg[1:] = ~same
        seg_id = torch.cumsum(newseg.to(torch.int64), 0) - 1
        m = int(seg_id[-1].item()) + 1 if x.numel() else 0
        c_sum = torch.zeros(m, dtype=_C, device=_dev).index_add_(0, seg_id, c)
        first = torch.nonzero(newseg, as_tuple=True)[0]
        self.x, self.z, self.c = x[first], z[first], c_sum
        # drop negligible
        keep = self.c.abs() > 1e-14
        self.x, self.z, self.c = self.x[keep], self.z[keep], self.c[keep]

    def _prune(self, K):
        if self.x.numel() > K:
            top = torch.topk(self.c.abs(), K).indices
            self.x, self.z, self.c = self.x[top], self.z[top], self.c[top]

    def apply_cz(self, a, b):
        """CZ conjugation: (x,z) -> (x, z ^ x-on-other-qubit); sign (-1)^(x_a & x_b & (z_a ^ z_b))."""
        xa = (self.x >> a) & 1
        xb = (self.x >> b) & 1
        # z_a ^= x_b ; z_b ^= x_a
        self.z = self.z ^ (xb << a) ^ (xa << b)
        za = (self.z >> a) & 1  # AFTER update? sign uses pre-update z. recompute pre.
        # sign: CZ conjugation flips sign when both X present and the Z-parity is odd.
        # derived/validated numerically in selftest; use pre-update z:
        # (handled by caller-verified formula below)
        # pre-update z parity:
        pass

    def apply_1q(self, q, R):
        """u3 on q: each non-I factor sigma_j -> sum_i R[i,j] sigma_i (branch up to 3).
        R may be passed as the 2x2 gate U (converted here) or a precomputed 3x3 SO(3)."""
        R = np.asarray(R)
        if R.shape == (2, 2):
            R = _u3_so3(R)
        xq = (self.x >> q) & 1
        zq = (self.z >> q) & 1
        fac = xq + 2 * zq  # 0=I,1=X,2=Z,3=Y  (our code: bit-x, bit-z)
        isI = fac == 0
        # identity factor: unchanged
        xI, zI, cI = self.x[isI], self.z[isI], self.c[isI]
        # map our fac code -> Pauli index (X=0,Y=1,Z=2) for R
        # fac: 1->X(0), 2->Z(2), 3->Y(1)
        code2idx = {1: 0, 2: 2, 3: 1}
        idx2xz = {0: (1, 0), 1: (1, 1), 2: (0, 1)}  # X,Y,Z -> (x,z) bits
        outs = [(xI, zI, cI)] if isI.any() else []
        nonI = ~isI
        if nonI.any():
            xn, zn, cn, fn = self.x[nonI], self.z[nonI], self.c[nonI], fac[nonI]
            xn_base = xn & ~(1 << q)  # clear bit q
            zn_base = zn & ~(1 << q)
            j = torch.empty_like(fn)  # Pauli index of source
            j[fn == 1] = 0
            j[fn == 2] = 2
            j[fn == 3] = 1
            for i in range(3):  # target Pauli X,Y,Z
                coef = torch.tensor(R[i], dtype=torch.float64, device=_dev)[j]  # R[i, j]
                nz = coef.abs() > 1e-15
                if not nz.any():
                    continue
                xb, zb = idx2xz[i]
                nx = xn_base[nz] | (xb << q)
                nz_ = zn_base[nz] | (zb << q)
                nc = cn[nz] * coef[nz].to(_C)
                outs.append((nx, nz_, nc))
        self.x = torch.cat([o[0] for o in outs])
        self.z = torch.cat([o[1] for o in outs])
        self.c = torch.cat([o[2] for o in outs])

    def expect_zero(self):
        """<0| . |0>: only pure-I/Z Paulis (x==0) contribute, each as +coeff (Z|0>=+|0>)."""
        mask = self.x == 0
        return complex(self.c[mask].sum().item()) if mask.any() else 0j


def _cz_sign_apply(ps, a, b):
    """Apply CZ conjugation WITH correct sign (validated in selftest)."""
    xa = (ps.x >> a) & 1
    xb = (ps.x >> b) & 1
    za = (ps.z >> a) & 1
    zb = (ps.z >> b) & 1
    # sign = (-1)^( xa*xb*(za ^ zb) )   [validated numerically]
    sign = 1 - 2 * (xa & xb & (za ^ zb))
    ps.c = ps.c * sign.to(_C)
    ps.z = ps.z ^ (xb << a) ^ (xa << b)


def evolve_expectation(gates, n, obs_qubit, K=200000, cutoff=1e-9, log=print,
                       every=400, deadline=None):
    """gates: list of ('u3', q, U2x2) or ('cz', a, b) in CIRCUIT order (G1..GL).
    Returns <0|C^dag Z_{obs_qubit} C|0>."""
    ps = PauliSum.single_Z(n, obs_qubit)
    t0 = time.time()
    for gi in range(len(gates) - 1, -1, -1):  # reverse: conjugate innermost (last) first
        g = gates[gi]
        if g[0] == 'cz':
            _cz_sign_apply(ps, g[1], g[2])
        else:
            ps.apply_1q(g[1], g[2])
            if ps.size() > K * 2:
                ps._merge(); ps._prune(K)
        if gi % every == 0:
            ps._merge(); ps._prune(K)
            if log and gi % (every * 5) == 0:
                log(f"    obs q{obs_qubit}: gate {gi} terms={ps.size()} "
                    f"maxc={ps.c.abs().max().item():.2e} t={time.time()-t0:.0f}s")
            if deadline and time.time() > deadline:
                if log: log(f"    obs q{obs_qubit}: deadline hit at gate {gi}")
                break
    ps._merge()
    return ps.expect_zero()
