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

"""TEBD matrix-product-state simulator for peaked circuits — torch/CUDA port.

Faithful port of the proven milestone-1 approach (custom MPS, mixed-canonical
SVD truncation, swap network for all-to-all CZ, canonical beam-search argmax) to
torch so it runs on the validator GPU via the CUDA libs torch already bundles —
no cupy/libcublas dependency. Algorithm validated against exact statevector.
"""
import time
import torch


class MPS:
    def __init__(self, n, dtype, dev, perm=None):
        self.n = n
        self.dtype = dtype
        self.dev = dev
        z = torch.zeros((1, 2, 1), dtype=dtype, device=dev)
        z[0, 0, 0] = 1.0
        self.A = [z.clone() for _ in range(n)]
        self.center = 0
        if perm is None:
            self.pos = list(range(n))
            self.qubit_at = list(range(n))
        else:
            assert sorted(perm) == list(range(n))
            self.pos = list(perm)
            self.qubit_at = [0] * n
            for q, p in enumerate(perm):
                self.qubit_at[p] = q

    def _move_right(self):
        p = self.center
        Dl, d, Dr = self.A[p].shape
        M = self.A[p].reshape(Dl * d, Dr)
        Q, R = torch.linalg.qr(M)
        k = Q.shape[1]
        self.A[p] = Q.reshape(Dl, d, k)
        self.A[p + 1] = torch.tensordot(R, self.A[p + 1], dims=([1], [0]))
        self.center = p + 1

    def _move_left(self):
        p = self.center
        Dl, d, Dr = self.A[p].shape
        M = self.A[p].reshape(Dl, d * Dr)
        Q, R = torch.linalg.qr(M.conj().T)
        Qr = Q.conj().T
        L = R.conj().T
        k = Qr.shape[0]
        self.A[p] = Qr.reshape(k, d, Dr)
        self.A[p - 1] = torch.tensordot(self.A[p - 1], L, dims=([2], [0]))
        self.center = p - 1

    def move_center_to(self, target):
        while self.center < target:
            self._move_right()
        while self.center > target:
            self._move_left()

    def apply_two_site(self, p, G4, chi, cutoff):
        A1, A2 = self.A[p], self.A[p + 1]
        Dl = A1.shape[0]; Dr = A2.shape[2]
        theta = torch.tensordot(A1, A2, dims=([2], [0]))           # (Dl,2,2,Dr)
        theta = torch.einsum("IJij,aijc->aIJc", G4, theta)
        M = theta.reshape(Dl * 2, 2 * Dr)
        U, s, Vh = torch.linalg.svd(M, full_matrices=False)
        if cutoff and s.numel():
            tol = cutoff * float(s[0])
            kc = int((s > tol).sum().item())
        else:
            kc = s.numel()
        k = max(1, min(chi, kc))
        U = U[:, :k]; s = s[:k]; Vh = Vh[:k, :]
        nrm = float(torch.sqrt((s * s).sum()).item())
        if nrm > 0:
            s = s / nrm
        self.A[p] = U.reshape(Dl, 2, k)
        self.A[p + 1] = (s[:, None] * Vh).reshape(k, 2, Dr)
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


def _gates(qc):
    import numpy as np
    index = {qb: i for i, qb in enumerate(qc.qubits)}
    for inst in qc.data:
        op = inst.operation
        if op.name in ("barrier", "measure"):
            continue
        qs = [index[qb] for qb in inst.qubits]
        yield np.asarray(op.to_matrix()), qs


def evolve(qc, chi, cutoff=1e-10, dtype=None, dev=None, perm=None, log=print):
    dev = dev or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = dtype or torch.complex128  # double precision (accuracy-first)
    SWAP = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                        dtype=dtype, device=dev).reshape(2, 2, 2, 2)
    mps = MPS(qc.num_qubits, dtype, dev, perm=perm)
    t0 = time.time()
    ng = 0
    for mat, qs in _gates(qc):
        M = torch.as_tensor(mat, dtype=dtype, device=dev)
        if len(qs) == 1:
            mps.apply_1q(qs[0], M)
        elif len(qs) == 2:
            mps.apply_2q(qs[0], qs[1], M.reshape(2, 2, 2, 2), chi, cutoff, SWAP)
        else:
            raise ValueError(f"{len(qs)}-qubit gate unsupported")
        ng += 1
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.time() - t0
    log(f"    [tebd_mps] {ng} gates, evolve={dt:.1f}s reachedχ={mps.max_bond()}")
    return mps, dt


def topk(mps, beam=512, k=8):
    """Canonical beam search. Returns LOGICAL-order [(bits, prob), ...] top-k."""
    mps.move_center_to(0)
    A = mps.A
    ones = torch.ones((1, 1), dtype=A[0].dtype, device=A[0].device)
    beams = [("", ones)]
    for a in A:
        cand = []
        for bits, vec in beams:
            for b in (0, 1):
                nv = vec @ a[:, b, :]
                w = float((nv.conj() * nv).real.sum().item())
                cand.append((w, bits + str(b), nv))
        cand.sort(key=lambda t: t[0], reverse=True)
        beams = [(bits, nv) for (_, bits, nv) in cand[:beam]]
    res = []
    for bits, nv in beams:
        logical = ["0"] * mps.n
        for chain_pos, ch in enumerate(bits):
            logical[mps.qubit_at[chain_pos]] = ch
        res.append(("".join(logical), float((nv.conj() * nv).real.sum().item())))
    res.sort(key=lambda t: t[1], reverse=True)
    return res[:k]
