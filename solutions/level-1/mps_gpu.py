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

"""Custom GPU TEBD matrix-product-state simulator for peaked circuits.

Maintains a mixed-canonical MPS, applies long-range 2-qubit gates via a swap
network, truncates each bond by (max bond chi, singular-value cutoff) using the
orthogonality center for OPTIMAL local truncation.  Returns top-K bitstrings by
MPS probability via canonical beam search.

Design goals: correctness (matches a reference at low chi) + speed (GPU SVD at
high chi) + reliability (no opaque backend hangs).
"""
import time
import numpy as np

try:
    import cupy as _cp
    _HAVE_CP = True
except Exception:
    _cp = None
    _HAVE_CP = False


class MPS:
    def __init__(self, n, xp, dtype, perm=None):
        self.n = n
        self.xp = xp
        self.dtype = dtype
        # |0...0>: each site (1,2,1) = [[1],[0]]
        z = xp.zeros((1, 2, 1), dtype=dtype)
        z[0, 0, 0] = 1.0
        self.A = [z.copy() for _ in range(n)]
        self.center = 0
        # perm[q] = initial chain position of logical qubit q (identity if None).
        # A different ordering yields independent truncation -> cross-check.
        if perm is None:
            self.pos = list(range(n))
            self.qubit_at = list(range(n))
        else:
            assert sorted(perm) == list(range(n))
            self.pos = list(perm)
            self.qubit_at = [0] * n
            for q, p in enumerate(perm):
                self.qubit_at[p] = q

    # --- canonical center movement (exact, no truncation) ---
    def _move_right(self):
        xp = self.xp
        p = self.center
        Dl, d, Dr = self.A[p].shape
        M = self.A[p].reshape(Dl * d, Dr)
        Q, R = xp.linalg.qr(M)
        k = Q.shape[1]
        self.A[p] = Q.reshape(Dl, d, k)
        self.A[p + 1] = xp.tensordot(R, self.A[p + 1], axes=(1, 0))  # (k,2,Dr2)
        self.center = p + 1

    def _move_left(self):
        xp = self.xp
        p = self.center
        Dl, d, Dr = self.A[p].shape
        M = self.A[p].reshape(Dl, d * Dr)
        Q, R = xp.linalg.qr(M.conj().T)     # M^H = Q R
        Qr = Q.conj().T                      # (k, d*Dr) orthonormal rows
        L = R.conj().T                       # (Dl, k)
        k = Qr.shape[0]
        self.A[p] = Qr.reshape(k, d, Dr)
        self.A[p - 1] = xp.tensordot(self.A[p - 1], L, axes=(2, 0))  # (Dl0,2,k)
        self.center = p - 1

    def move_center_to(self, target):
        while self.center < target:
            self._move_right()
        while self.center > target:
            self._move_left()

    # --- two-site gate at bond (p, p+1); center must be at p ---
    def apply_two_site(self, p, G4, chi, cutoff):
        xp = self.xp
        A1, A2 = self.A[p], self.A[p + 1]
        Dl = A1.shape[0]; Dr = A2.shape[2]
        theta = xp.tensordot(A1, A2, axes=(2, 0))        # (Dl,2,2,Dr)
        # apply gate: G4[I,J,i,j] * theta[Dl,i,j,Dr] -> (Dl,I,J,Dr)
        theta = xp.einsum("IJij,aijc->aIJc", G4, theta, optimize=True)
        M = theta.reshape(Dl * 2, 2 * Dr)
        U, s, Vh = xp.linalg.svd(M, full_matrices=False)
        # truncation
        if cutoff and s.size:
            tol = cutoff * float(s[0])
            kc = int((s > tol).sum())
        else:
            kc = s.size
        k = max(1, min(chi, kc))
        U = U[:, :k]; s = s[:k]; Vh = Vh[:k, :]
        nrm = float(xp.sqrt((s * s).sum()))
        if nrm > 0:
            s = s / nrm
        self.A[p] = U.reshape(Dl, 2, k)
        self.A[p + 1] = (s[:, None] * Vh).reshape(k, 2, Dr)
        self.center = p + 1

    def apply_1q(self, q, U2):
        xp = self.xp
        p = self.pos[q]
        self.A[p] = xp.einsum("ij,ajb->aib", U2, self.A[p], optimize=True)

    def _swap_adjacent(self, p, chi, cutoff, SWAP4):
        self.move_center_to(p)
        self.apply_two_site(p, SWAP4, chi, cutoff)
        # update permutation bookkeeping
        qa, qb = self.qubit_at[p], self.qubit_at[p + 1]
        self.qubit_at[p], self.qubit_at[p + 1] = qb, qa
        self.pos[qa], self.pos[qb] = p + 1, p

    def apply_2q(self, q1, q2, G4, chi, cutoff, SWAP4):
        p1, p2 = self.pos[q1], self.pos[q2]
        if p1 > p2:
            p1, p2 = p2, p1
            q1, q2 = q2, q1
            # swapping the two logical qubits => swap the gate's qubit legs too
            # (no-op for symmetric gates like CZ; correct for asymmetric ones)
            G4 = G4.transpose(1, 0, 3, 2)
        # bring q2 (at p2) down to p1+1 via adjacent swaps
        while p2 > p1 + 1:
            self._swap_adjacent(p2 - 1, chi, cutoff, SWAP4)
            p2 -= 1
        self.move_center_to(p1)
        self.apply_two_site(p1, G4, chi, cutoff)

    def max_bond(self):
        return max(a.shape[2] for a in self.A)


def _gates(qc):
    """Yield (matrix, [logical qubits]) for each instruction via qiskit."""
    index = {qb: i for i, qb in enumerate(qc.qubits)}
    for inst in qc.data:
        op = inst.operation
        if op.name in ("barrier", "measure"):
            continue
        qs = [index[qb] for qb in inst.qubits]
        yield np.asarray(op.to_matrix()), qs


def evolve(qc, chi, cutoff=1e-10, gpu=0, dtype=None, perm=None, log=print):
    if _HAVE_CP:
        xp = _cp
        ctx = _cp.cuda.Device(gpu)
    else:
        xp = np
        ctx = _nullctx()
    dtype = dtype or xp.complex64
    with ctx:
        SWAP = xp.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                          dtype=dtype).reshape(2, 2, 2, 2)
        mps = MPS(qc.num_qubits, xp, dtype, perm=perm)
        t0 = time.time()
        ng = 0
        for mat, qs in _gates(qc):
            M = xp.asarray(mat, dtype=dtype)
            if len(qs) == 1:
                mps.apply_1q(qs[0], M)
            elif len(qs) == 2:
                G4 = M.reshape(2, 2, 2, 2)
                mps.apply_2q(qs[0], qs[1], G4, chi, cutoff, SWAP)
            else:
                raise ValueError(f"{len(qs)}-qubit gate unsupported")
            ng += 1
        xp.cuda.Stream.null.synchronize() if _HAVE_CP else None
        dt = time.time() - t0
    log(f"    [mps_gpu] {ng} gates, evolve={dt:.1f}s reachedχ={mps.max_bond()}")
    return mps, dt


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---- top-K extraction (canonical beam search), returns LOGICAL-order bitstrings ----
def topk(mps, beam=256, k=None):
    xp = mps.xp
    # right-canonicalize so left partial norm^2 == prefix marginal
    mps.move_center_to(0)
    A = mps.A
    if k is None:
        k = beam
    beams = [("", xp.ones((1, 1), dtype=A[0].dtype))]
    for a in A:
        cand = []
        for bits, vec in beams:
            for b in (0, 1):
                nv = vec @ a[:, b, :]
                w = float(xp.vdot(nv, nv).real)
                cand.append((w, bits + str(b), nv))
        cand.sort(key=lambda t: t[0], reverse=True)
        beams = [(bits, nv) for (_, bits, nv) in cand[:beam]]
    res = []
    for bits, nv in beams:
        # bits are in CHAIN order; remap to logical order
        logical = ["0"] * mps.n
        for chain_pos, ch in enumerate(bits):
            logical[mps.qubit_at[chain_pos]] = ch
        res.append(("".join(logical), float(xp.vdot(nv, nv).real)))
    res.sort(key=lambda t: t[1], reverse=True)
    return res[:k]


if __name__ == "__main__":
    import sys, json
    from qload import load_qiskit
    qasm = sys.argv[1]; meta = sys.argv[2] if len(sys.argv) > 2 else None
    chi = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    beam = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    qc = load_qiskit(qasm)
    t0 = time.time()
    mps, dt = evolve(qc, chi)
    cands = topk(mps, beam=beam, k=8)
    pred = cands[0][0]
    print(f"top1: {pred}  w={cands[0][1]:.4g}  (total {time.time()-t0:.1f}s χ={chi})")
    print("top5:", [(b[:12]+'..', round(w,5)) for b, w in cands[:5]])
    if meta:
        m = json.load(open(meta)); exp = m["peaked_state"]
        def match(b): return b == exp or b == exp[::-1]
        ranks = [i for i, (b, _) in enumerate(cands) if match(b)]
        print("expected:", exp)
        print(f"TOP1 MATCH: {match(pred)} | peak in top{len(cands)} at rank: {ranks}")
