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

"""Fast torch TEBD MPS for peaked circuits: eigh-trick truncation (~10x faster
than gesvd at chi>=512), cumulative truncation-fidelity tracking, direct
bitstring amplitude, canonical beam top-k. Port of peaked_solution/mps_gpu.py.
"""
import time
import numpy as np
import torch

torch.set_grad_enabled(False)


class MPS:
    def __init__(self, n, dtype=torch.complex64, device="cuda", perm=None):
        self.n = n
        self.dtype = dtype
        self.dev = device
        z = torch.zeros((1, 2, 1), dtype=dtype, device=device)
        z[0, 0, 0] = 1.0
        self.A = [z.clone() for _ in range(n)]
        self.center = 0
        self.logF = 0.0        # cumulative log(kept mass) -> fidelity estimate
        self.min_keep = 1.0    # worst single-truncation kept fraction
        if perm is None:
            self.pos = list(range(n))
            self.qubit_at = list(range(n))
        else:
            self.pos = list(perm)
            self.qubit_at = [0] * n
            for q, p in enumerate(perm):
                self.qubit_at[p] = q

    def _move_right(self):
        p = self.center
        Dl, d, Dr = self.A[p].shape
        M = self.A[p].reshape(Dl * d, Dr)
        Q, R = torch.linalg.qr(M, mode="reduced")
        k = Q.shape[1]
        self.A[p] = Q.reshape(Dl, d, k)
        self.A[p + 1] = torch.tensordot(R, self.A[p + 1], dims=([1], [0]))
        self.center = p + 1

    def _move_left(self):
        p = self.center
        Dl, d, Dr = self.A[p].shape
        M = self.A[p].reshape(Dl, d * Dr)
        Q, R = torch.linalg.qr(M.mH, mode="reduced")
        k = Q.shape[1]
        self.A[p] = Q.mH.reshape(k, d, Dr)
        self.A[p - 1] = torch.tensordot(self.A[p - 1], R.mH, dims=([2], [0]))
        self.center = p - 1

    def move_center_to(self, target):
        while self.center < target:
            self._move_right()
        while self.center > target:
            self._move_left()

    def apply_two_site(self, p, G4, chi, cutoff):
        A1, A2 = self.A[p], self.A[p + 1]
        Dl = A1.shape[0]; Dr = A2.shape[2]
        theta = torch.tensordot(A1, A2, dims=([2], [0]))          # (Dl,2,2,Dr)
        theta = torch.einsum("IJij,aijc->aIJc", G4, theta)
        M = theta.reshape(Dl * 2, 2 * Dr)
        a, b = M.shape
        # eigh-trick truncated SVD: eigendecompose the smaller Gram matrix.
        if a >= b:
            H = M.mH @ M                                          # (b,b)
            # identity-shift regularization: eigenvectors are exactly unchanged,
            # but cusolver eigh silently NaNs on rank-deficient c64 Grams without it
            eps = float(H.diagonal().real.sum()) / b * 1e-6 + 1e-30
            H = H + eps * torch.eye(b, dtype=H.dtype, device=H.device)
            try:
                w, V = torch.linalg.eigh(H)                       # ascending
                w = w - eps
                if not bool(torch.isfinite(V.real).all()):
                    raise RuntimeError("eigh nan")
            except Exception:
                U_, s_, Vh_ = torch.linalg.svd(M, full_matrices=False)
                w = (s_ * s_).flip(0); V = Vh_.mH.flip(1)
            w = w.flip(0); V = V.flip(1)
            total = float(w.sum().real)
            wc = torch.clamp(w.real, min=0.0)
            s = torch.sqrt(wc)
            if cutoff and s.numel():
                tol = cutoff * float(s[0])
                kc = int((s > tol).sum())
            else:
                kc = s.numel()
            k = max(1, min(chi, kc))
            s = s[:k]; V = V[:, :k]
            U = M @ V
            U = U / torch.clamp(s, min=1e-30)
            Vh = V.mH
        else:
            H = M @ M.mH                                          # (a,a)
            eps = float(H.diagonal().real.sum()) / a * 1e-6 + 1e-30
            H = H + eps * torch.eye(a, dtype=H.dtype, device=H.device)
            try:
                w, U = torch.linalg.eigh(H)
                w = w - eps
                if not bool(torch.isfinite(U.real).all()):
                    raise RuntimeError("eigh nan")
            except Exception:
                U_, s_, Vh_ = torch.linalg.svd(M, full_matrices=False)
                w = (s_ * s_).flip(0); U = U_.flip(1)
            w = w.flip(0); U = U.flip(1)
            total = float(w.sum().real)
            wc = torch.clamp(w.real, min=0.0)
            s = torch.sqrt(wc)
            if cutoff and s.numel():
                tol = cutoff * float(s[0])
                kc = int((s > tol).sum())
            else:
                kc = s.numel()
            k = max(1, min(chi, kc))
            s = s[:k]; U = U[:, :k]
            Vh = (U.mH @ M) / torch.clamp(s.unsqueeze(1), min=1e-30)
        # NaN GUARD. The eigh trick checks only the eigenvectors; a non-finite eigenVALUE (or a 0/0 in the division
        # above) slips through, poisons the whole state, and the run "finishes" with margin nan / F 1.0 (MEASURED:
        # ensemble member, chain seed 102, d3_s2). Recompute this one split with a plain SVD in complex128; if even
        # that is not finite the state was already bad -> fail loudly instead of returning garbage.
        if not (bool(torch.isfinite(s).all()) and bool(torch.isfinite(U.real).all()) and bool(torch.isfinite(U.imag).all())
                and bool(torch.isfinite(Vh.real).all()) and bool(torch.isfinite(Vh.imag).all())):
            if not bool(torch.isfinite(M.real).all() and torch.isfinite(M.imag).all()):
                raise FloatingPointError("mps_torch: non-finite two-site tensor (state already corrupted)")
            U_, s_, Vh_ = torch.linalg.svd(M.to(torch.complex128), full_matrices=False)
            total = float((s_ * s_).sum())
            if cutoff and s_.numel():
                kc = int((s_ > cutoff * float(s_[0])).sum())
            else:
                kc = s_.numel()
            k = max(1, min(chi, kc))
            s = s_[:k].to(torch.float32 if self.dtype == torch.complex64 else torch.float64)
            U = U_[:, :k].to(self.dtype); Vh = Vh_[:k].to(self.dtype)
            self.nan_rescues = getattr(self, "nan_rescues", 0) + 1
        kept = float((s * s).sum())
        if total > 0:
            frac = min(1.0, kept / total)
            if frac > 0:
                self.logF += np.log(frac)
                self.min_keep = min(self.min_keep, frac)
        nrm = float(torch.sqrt((s * s).sum()))
        if nrm > 0:
            s = s / nrm
        self.A[p] = U.reshape(Dl, 2, k)
        self.A[p + 1] = (s.unsqueeze(1).to(self.dtype) * Vh).reshape(k, 2, Dr)
        self.center = p + 1

    def apply_1q(self, q, U2):
        p = self.pos[q]
        self.A[p] = torch.einsum("ij,ajb->aib", U2, self.A[p])

    def _swap_adjacent(self, p, chi, cutoff, SWAP4):
        self.move_center_to(p)
        self.apply_two_site(p, SWAP4, chi, cutoff)
        qa, qb = self.qubit_at[p], self.qubit_at[p + 1]
        self.qubit_at[p], self.qubit_at[p + 1] = qb, qa
        self.pos[qa], self.pos[qb] = p + 1, p

    def apply_2q(self, q1, q2, G4, chi, cutoff, SWAP4):
        p1, p2 = self.pos[q1], self.pos[q2]
        if p1 > p2:
            p1, p2 = p2, p1
            q1, q2 = q2, q1
            G4 = G4.permute(1, 0, 3, 2)
        while p2 > p1 + 1:
            self._swap_adjacent(p2 - 1, chi, cutoff, SWAP4)
            p2 -= 1
        self.move_center_to(p1)
        self.apply_two_site(p1, G4, chi, cutoff)

    def max_bond(self):
        return max(a.shape[2] for a in self.A)

    def amp(self, logical_bits):
        """<logical_bits|psi> given chain layout; logical_bits[i] = qubit i."""
        v = torch.ones((1, 1), dtype=self.dtype, device=self.dev)
        for p in range(self.n):
            b = int(logical_bits[self.qubit_at[p]])
            v = v @ self.A[p][:, b, :]
        return complex(v[0, 0])


def evolve(qc, chi, cutoff=1e-10, dtype=torch.complex64, device="cuda",
           perm=None, log=print, log_every=1000, truth=None):
    index = {qb: i for i, qb in enumerate(qc.qubits)}
    SWAP = torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype, device=device).reshape(2, 2, 2, 2)
    mps = MPS(qc.num_qubits, dtype=dtype, device=device, perm=perm)
    t0 = time.time()
    ng = 0
    for inst in qc.data:
        op = inst.operation
        if op.name in ("barrier", "measure"):
            continue
        qs = [index[qb] for qb in inst.qubits]
        M = torch.tensor(np.asarray(op.to_matrix()), dtype=dtype, device=device)
        if len(qs) == 1:
            mps.apply_1q(qs[0], M)
        elif len(qs) == 2:
            mps.apply_2q(qs[0], qs[1], M.reshape(2, 2, 2, 2), chi, cutoff, SWAP)
        else:
            raise ValueError("3q+ gate")
        ng += 1
        if log_every and ng % log_every == 0:
            torch.cuda.synchronize()
            extra = ""
            if truth is not None:
                a = mps.amp(truth)
                extra = f" amp_true={abs(a):.3e}"
            log(f"    [torch] {ng} gates {time.time()-t0:.0f}s chi={mps.max_bond()} "
                f"F_est={np.exp(mps.logF):.3e}{extra}")
    torch.cuda.synchronize()
    log(f"    [torch] done {ng} gates in {time.time()-t0:.1f}s reached chi={mps.max_bond()} "
        f"F_est={np.exp(mps.logF):.3e} min_keep={mps.min_keep:.4f}")
    return mps, time.time() - t0


def topk(mps, beam=512, k=8):
    mps.move_center_to(0)
    beams = [("", torch.ones((1, 1), dtype=mps.dtype, device=mps.dev))]
    for a in mps.A:
        cand = []
        for bits, vec in beams:
            for b in (0, 1):
                nv = vec @ a[:, b, :]
                w = float((nv.conj() * nv).sum().real)
                cand.append((w, bits + str(b), nv))
        cand.sort(key=lambda t: t[0], reverse=True)
        beams = [(bits, nv) for (_, bits, nv) in cand[:beam]]
    res = []
    for bits, nv in beams:
        logical = ["0"] * mps.n
        for chain_pos, ch in enumerate(bits):
            logical[mps.qubit_at[chain_pos]] = ch
        res.append(("".join(logical), float((nv.conj() * nv).sum().real)))
    res.sort(key=lambda t: t[1], reverse=True)
    return res[:k]
