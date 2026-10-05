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
import math
import time
from dataclasses import dataclass, field
import numpy as np
from src.rt.source_ledger import parse_qasm2
SWAP4 = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex).reshape(2, 2, 2, 2)

def u_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    c, s = (math.cos(theta / 2), math.sin(theta / 2))
    return np.asarray([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def gate_matrix(name: str, params) -> np.ndarray:
    if name == 'u':
        return u_matrix(*params)
    if name == 'cz':
        return np.diag([1, 1, 1, -1]).astype(complex)
    raise ValueError('unsupported gate ' + name)

@dataclass
class Gate:
    index: int
    name: str
    params: tuple
    qubits: tuple

    def matrix(self) -> np.ndarray:
        return gate_matrix(self.name, self.params)

def load_gates(path, expected_sha256=None) -> tuple[int, list[Gate]]:
    n, source = parse_qasm2(path, expected_sha256)
    return (n, [Gate(g.raw_id, g.name, tuple(g.params), tuple(g.qubits)) for g in source])

def inverse_gates(gates: list[Gate]) -> list[Gate]:
    out = []
    for g in reversed(gates):
        if g.name == 'u':
            theta, phi, lam = g.params
            out.append(Gate(g.index, 'u', (-theta, -lam, -phi), g.qubits))
        else:
            out.append(g)
    return out

@dataclass
class Telemetry:
    gates_applied: int = 0
    two_site_ops: int = 0
    swaps: int = 0
    log10_kept: float = 0.0
    max_bond: int = 1
    cap_hits: int = 0
    mpo_ops: int = 0
    seconds: float = 0.0
    bond_trace: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {'gates_applied': self.gates_applied, 'two_site_ops': self.two_site_ops, 'swaps': self.swaps, 'log10_kept': self.log10_kept, 'max_bond': self.max_bond, 'cap_hits': self.cap_hits, 'mpo_ops': self.mpo_ops, 'seconds': self.seconds, 'bond_trace': list(self.bond_trace)}

def _perm(x, axes):
    return x.permute(*axes) if hasattr(x, 'permute') else np.transpose(x, axes)

class MPS:

    def __init__(self, n: int, chi: int, *, xp=np, dtype=np.complex128, perm=None, cutoff: float=0.0, product_bits: str | None=None, svd_method: str='svd'):
        self.n, self.chi, self.xp, self.dtype, self.cutoff = (n, chi, xp, dtype, cutoff)
        if svd_method not in ('svd', 'gram'):
            raise ValueError('svd_method must be svd or gram')
        self.svd_method = svd_method
        if perm is None:
            perm = list(range(n))
        if sorted(perm) != list(range(n)):
            raise ValueError('perm must be a permutation of range(n)')
        self.pos = list(perm)
        self.qubit_at = [0] * n
        for q, p in enumerate(perm):
            self.qubit_at[p] = q
        bits = product_bits or '0' * n
        if len(bits) != n or set(bits) - {'0', '1'}:
            raise ValueError('product_bits must be n binary characters')
        self.A = []
        for p in range(n):
            t = xp.zeros((1, 2, 1), dtype=dtype)
            t[0, int(bits[self.qubit_at[p]]), 0] = 1.0
            self.A.append(t)
        self.center = 0
        self.log_norm = 0.0
        self.telemetry = Telemetry()
        self._swap = xp.asarray(SWAP4, dtype=dtype)

    def _move_right(self):
        xp, p = (self.xp, self.center)
        Dl, d, Dr = self.A[p].shape
        Q, R = xp.linalg.qr(self.A[p].reshape(Dl * d, Dr))
        self.A[p] = Q.reshape(Dl, d, Q.shape[1])
        self.A[p + 1] = xp.tensordot(R, self.A[p + 1], axes=(1, 0))
        self.center = p + 1

    def _move_left(self):
        xp, p = (self.xp, self.center)
        Dl, d, Dr = self.A[p].shape
        Q, R = xp.linalg.qr(self.A[p].reshape(Dl, d * Dr).conj().T)
        self.A[p] = Q.conj().T.reshape(Q.shape[1], d, Dr)
        self.A[p - 1] = xp.tensordot(self.A[p - 1], R.conj().T, axes=(2, 0))
        self.center = p - 1

    def move_center_to(self, target: int):
        while self.center < target:
            self._move_right()
        while self.center > target:
            self._move_left()

    def apply_two_site(self, p: int, G4):
        xp = self.xp
        self.move_center_to(p)
        A1, A2 = (self.A[p], self.A[p + 1])
        Dl, Dr = (A1.shape[0], A2.shape[2])
        theta = xp.tensordot(A1, A2, axes=(2, 0))
        theta = xp.einsum('IJij,aijc->aIJc', G4, theta)
        matrix = theta.reshape(Dl * 2, 2 * Dr)
        if self.svd_method == 'gram':
            U, s, Vh = gram_svd(matrix, xp, self.chi)
        else:
            U, s, Vh = robust_svd(matrix, xp)
        total = float((s * s).sum())
        if self.cutoff and s.shape[0]:
            kc = int((s > self.cutoff * float(s[0])).sum())
        else:
            kc = int(s.shape[0])
        k = max(1, min(self.chi, kc))
        kept = float((s[:k] * s[:k]).sum())
        tel = self.telemetry
        if kept < total:
            tel.log10_kept += math.log10(kept / total)
        if k == self.chi and kc > self.chi:
            tel.cap_hits += 1
        scale = math.sqrt(kept)
        self.log_norm += math.log(scale)
        s = s[:k] / scale
        self.A[p] = U[:, :k].reshape(Dl, 2, k)
        self.A[p + 1] = (s[:, None] * Vh[:k, :]).reshape(k, 2, Dr)
        self.center = p + 1
        tel.two_site_ops += 1
        tel.max_bond = max(tel.max_bond, k)

    def compress(self):
        xp, tel = (self.xp, self.telemetry)
        for q in range(self.n - 1):
            Dl, d, Dr = self.A[q].shape
            Q, R = xp.linalg.qr(self.A[q].reshape(Dl * d, Dr))
            self.A[q] = Q.reshape(Dl, d, -1)
            self.A[q + 1] = xp.tensordot(R, self.A[q + 1], axes=(1, 0))
        for q in range(self.n - 1, 0, -1):
            Dl, d, Dr = self.A[q].shape
            matrix = self.A[q].reshape(Dl, d * Dr)
            if self.svd_method == 'gram':
                U, s, Vh = gram_svd(matrix, xp, self.chi)
            else:
                U, s, Vh = robust_svd(matrix, xp)
            total = float((s * s).sum())
            if self.cutoff and s.shape[0]:
                kc = int((s > self.cutoff * float(s[0])).sum())
            else:
                kc = int(s.shape[0])
            k = max(1, min(self.chi, kc))
            kept = float((s[:k] * s[:k]).sum())
            if kept < total:
                tel.log10_kept += math.log10(kept / total)
            if k == self.chi and kc > self.chi:
                tel.cap_hits += 1
            scale = math.sqrt(kept)
            self.log_norm += math.log(scale)
            sk = s[:k] / scale
            self.A[q] = Vh[:k, :].reshape(k, d, Dr)
            self.A[q - 1] = xp.tensordot(self.A[q - 1], U[:, :k] * sk[None, :], axes=(2, 0))
            tel.max_bond = max(tel.max_bond, k)
        self.center = 0
        return self

    def apply_mpo_zipup(self, W):
        xp, tel = (self.xp, self.telemetry)
        if len(W) != self.n:
            raise ValueError(f'MPO has {len(W)} sites, state has {self.n}')
        self.compress()
        carry = xp.asarray(np.ones((1, 1, 1)), dtype=self.dtype)
        out = []
        for p, w in enumerate(W):
            w = xp.asarray(w, dtype=self.dtype)
            A = self.A[p]
            k_, a_, w_ = carry.shape
            _, i_, b_ = A.shape
            _, o_, _, y_ = w.shape
            A_ = A if A.dtype == carry.dtype else xp.asarray(A, dtype=carry.dtype)
            w_c = w if w.dtype == carry.dtype else xp.asarray(w, dtype=carry.dtype)
            kwib = (_perm(carry, (0, 2, 1)).reshape(k_ * w_, a_) @ A_.reshape(a_, i_ * b_)).reshape(k_, w_, i_, b_)
            kb_wi = _perm(kwib, (0, 3, 1, 2)).reshape(k_ * b_, w_ * i_)
            wi_oy = _perm(w_c, (0, 2, 1, 3)).reshape(w_ * i_, o_ * y_)
            theta = _perm((kb_wi @ wi_oy).reshape(k_, b_, o_, y_), (0, 2, 1, 3))
            del kwib, kb_wi, wi_oy
            if hasattr(xp, 'empty_cache'):
                xp.empty_cache()
            k, d, Dr, Wr = theta.shape
            matrix = theta.reshape(k * d, Dr * Wr)
            if self.svd_method == 'gram':
                U, sv, Vh = gram_svd(matrix, xp, self.chi)
            else:
                U, sv, Vh = robust_svd(matrix, xp)
            total = float((sv * sv).sum())
            if self.cutoff and sv.shape[0]:
                kc = int((sv > self.cutoff * float(sv[0])).sum())
            else:
                kc = int(sv.shape[0])
            kk = max(1, min(self.chi, kc))
            kept = float((sv[:kk] * sv[:kk]).sum())
            if kept < total:
                tel.log10_kept += math.log10(kept / total)
            if kk == self.chi and kc > self.chi:
                tel.cap_hits += 1
            scale = math.sqrt(kept)
            self.log_norm += math.log(scale)
            out.append(U[:, :kk].reshape(k, d, kk))
            if p % 8 == 7 or p == self.n - 1:
                print(f'{{"zipup_site": {p + 1}, "of": {self.n}, "bond": {int(kk)}}}', flush=True)
            carry = ((sv[:kk] / scale)[:, None] * Vh[:kk, :]).reshape(kk, Dr, Wr)
            tel.max_bond = max(tel.max_bond, kk)
        tail = float(xp.asarray(carry).reshape(-1)[0].real) if carry.reshape(-1).shape[0] == 1 else None
        if tail is None:
            last = out[-1]
            out[-1] = xp.einsum('aib,bxy->aib', last, carry.reshape(carry.shape[0], 1, 1)[:last.shape[2]])
        else:
            out[-1] = out[-1] * tail
        self.A = out
        self.center = None
        tel.mpo_ops += 1
        return self.compress()

    def apply_mpo(self, W, *, compress: bool=True):
        xp = self.xp
        if len(W) != self.n:
            raise ValueError(f'MPO has {len(W)} sites, state has {self.n}')
        out = []
        for A, w in zip(self.A, W):
            w = xp.asarray(w, dtype=self.dtype)
            Dl, d, Dr = A.shape
            Wl, do, di, Wr = w.shape
            if di != d:
                raise ValueError(f'MPO input leg {di} does not match physical dimension {d}')
            T = xp.einsum('aib,xoiy->axoby', A, w)
            out.append(T.reshape(Dl * Wl, do, Dr * Wr))
        self.A = out
        self.center = None
        self.telemetry.mpo_ops += 1
        return self.compress() if compress else self

    def apply_1q(self, q: int, U2):
        p = self.pos[q]
        self.A[p] = self.xp.einsum('ij,ajb->aib', U2, self.A[p])

    def swap_adjacent(self, p: int):
        self.apply_two_site(p, self._swap)
        qa, qb = (self.qubit_at[p], self.qubit_at[p + 1])
        self.qubit_at[p], self.qubit_at[p + 1] = (qb, qa)
        self.pos[qa], self.pos[qb] = (p + 1, p)
        self.telemetry.swaps += 1

    def apply_2q(self, q1: int, q2: int, G4):
        p1, p2 = (self.pos[q1], self.pos[q2])
        if p1 > p2:
            p1, p2 = (p2, p1)
            G4 = _perm(G4, (1, 0, 3, 2))
        while p2 > p1 + 1:
            self.swap_adjacent(p2 - 1)
            p2 -= 1
        self.apply_two_site(p1, G4)

    def apply_gates(self, gates, *, trace_every: int=0, progress=None):
        xp, tel = (self.xp, self.telemetry)
        t0 = time.time()
        for g in gates:
            M = xp.asarray(g.matrix(), dtype=self.dtype)
            if len(g.qubits) == 1:
                self.apply_1q(g.qubits[0], M)
            else:
                self.apply_2q(g.qubits[0], g.qubits[1], M.reshape(2, 2, 2, 2))
            tel.gates_applied += 1
            if trace_every and tel.gates_applied % trace_every == 0:
                tel.bond_trace.append((g.index, self.max_bond(), tel.log10_kept))
                if progress is not None:
                    progress(g.index, self.max_bond(), tel.log10_kept, tel.seconds + time.time() - t0)
        tel.seconds += time.time() - t0
        return self

    def restore_order(self, target_qubit_at=None):
        target = list(range(self.n)) if target_qubit_at is None else list(target_qubit_at)
        for p_target in range(self.n):
            q = target[p_target]
            p = self.pos[q]
            while p > p_target:
                self.swap_adjacent(p - 1)
                p -= 1
        assert self.qubit_at == target
        return self

    def permute_logical(self, pi):
        new_pos = [0] * self.n
        for q, p in enumerate(self.pos):
            new_pos[pi[q]] = p
        self.pos = new_pos
        for q, p in enumerate(new_pos):
            self.qubit_at[p] = q
        return self

    def max_bond(self) -> int:
        return max((int(a.shape[2]) for a in self.A))

    def bond_profile(self) -> list[int]:
        return [int(a.shape[2]) for a in self.A[:-1]]

    def norm(self) -> float:
        self.move_center_to(0)
        a = self.A[0]
        return float(self.xp.sqrt(self.xp.vdot(a, a).real))

    def to_numpy(self):
        get = getattr(self.xp, 'asnumpy', np.asarray)
        return [get(a) for a in self.A]

def robust_svd(matrix, xp):
    try:
        return xp.linalg.svd(matrix, full_matrices=False)
    except Exception:
        if xp is not np:
            raise
    try:
        import scipy.linalg
        return scipy.linalg.svd(matrix, full_matrices=False, lapack_driver='gesvd')
    except Exception:
        pass
    m, k = matrix.shape
    if m <= k:
        w, U = np.linalg.eigh(matrix @ matrix.conj().T)
        order = np.argsort(w)[::-1]
        w, U = (np.clip(w[order], 0, None), U[:, order])
        s = np.sqrt(w)
        Vh = U.conj().T @ matrix / np.where(s > 0, s, 1)[:, None]
        return (U, s, Vh)
    w, V = np.linalg.eigh(matrix.conj().T @ matrix)
    order = np.argsort(w)[::-1]
    w, V = (np.clip(w[order], 0, None), V[:, order])
    s = np.sqrt(w)
    U = matrix @ V / np.where(s > 0, s, 1)[None, :]
    return (U, s, V.conj().T)

def gram_svd(matrix, xp, k):
    m, n = matrix.shape
    try:
        if m <= n:
            G = matrix @ matrix.conj().T
            w, U = xp.linalg.eigh((G + G.conj().T) * 0.5)
            order = xp.argsort(-w)
            w, U = (w[order], U[:, order])
            s = xp.sqrt(xp.maximum(w, 0))
            kk = int(min(k, s.shape[0]))
            Uk, sk = (U[:, :kk], s[:kk])
            Vh = Uk.conj().T @ matrix / xp.where(sk > 0, sk, 1)[:, None]
        else:
            G = matrix.conj().T @ matrix
            w, V = xp.linalg.eigh((G + G.conj().T) * 0.5)
            order = xp.argsort(-w)
            w, V = (w[order], V[:, order])
            s = xp.sqrt(xp.maximum(w, 0))
            kk = int(min(k, s.shape[0]))
            Vk, sk = (V[:, :kk], s[:kk])
            Uk = matrix @ Vk / xp.where(sk > 0, sk, 1)[None, :]
            Vh = Vk.conj().T
        if bool(xp.isfinite(Uk).all()) and bool(xp.isfinite(Vh).all()) and bool(xp.isfinite(s).all()):
            return (Uk, s, Vh)
    except Exception:
        pass
    return robust_svd(matrix, xp)

def overlap(a: MPS, b: MPS) -> complex:
    if a.qubit_at != b.qubit_at:
        raise ValueError('chain orders differ; restore_order first')
    xp = a.xp
    env = xp.ones((1, 1), dtype=a.dtype)
    for A, B in zip(a.A, b.A):
        env = xp.einsum('ab,aic,bid->cd', env, A.conj(), B)
    return complex(env[0, 0])

def dense_state(mps: MPS) -> np.ndarray:
    arrays = mps.to_numpy()
    psi = arrays[0][0]
    for a in arrays[1:]:
        psi = np.tensordot(psi, a, axes=(psi.ndim - 1, 0))
    psi = psi[..., 0]
    axes = [mps.pos[q] for q in range(mps.n)]
    psi = np.transpose(psi, axes)
    return psi.reshape(-1) * math.exp(mps.log_norm)

def fiedler_order(n: int, gates) -> list[int]:
    W = np.zeros((n, n))
    for g in gates:
        if len(g.qubits) == 2:
            a, b = g.qubits
            W[a, b] += 1
            W[b, a] += 1
    L = np.diag(W.sum(1)) - W
    vals, vecs = np.linalg.eigh(L)
    fiedler = vecs[:, 1]
    order = np.argsort(fiedler)
    perm = [0] * n
    for p, q in enumerate(order):
        perm[int(q)] = p
    return perm

def random_order(n: int, seed: int) -> list[int]:
    return [int(x) for x in np.random.default_rng(seed).permutation(n)]
