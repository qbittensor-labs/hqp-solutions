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
import numpy as np

def _perm(x, axes):
    return x.permute(*axes) if hasattr(x, 'permute') else np.transpose(x, axes)

def _matmul(a, b):
    return a @ b

def _conj_t(m):
    return m.mH if hasattr(m, 'mH') else m.conj().T

def _flip(w, xp):
    return w[::-1] if xp is np else xp.flip(w, (0,))

def _flip_cols(v, xp):
    return v[:, ::-1] if xp is np else xp.flip(v, (1,))

def _sqrt_clamped(w, xp):
    return np.sqrt(np.clip(w, 0.0, None)) if xp is np else xp.sqrt(xp.clamp(w, min=0.0))

class LocalChain:

    def __init__(self, arrays, *, xp=np, cutoff: float=0.0, max_bond: int | None=None, record=None, decomp: str='svd'):
        self.a = list(arrays)
        self.n = len(self.a)
        self.xp = xp
        self.cutoff = float(cutoff)
        self.max_bond = max_bond
        self.center: int | None = None
        self.log_retained_ln = 0.0
        self.truncations = 0
        self.record = record
        if decomp not in ('svd', 'gram'):
            raise ValueError(f"decomp must be 'svd' or 'gram', not {decomp!r}")
        self.decomp = decomp
        self.gram_fallbacks = 0
        self._last_gram_error = None

    def _qr(self, matrix):
        return self.xp.linalg.qr(matrix)

    def _left_step(self, i):
        xp = self.xp
        A, B = (self.a[i], self.a[i + 1])
        dl, dr = (A.shape[0], A.shape[1])
        q, r = self._qr(xp.permute(A, (0, 2, 3, 1)).reshape(dl * 4, dr) if hasattr(xp, 'permute') else np.transpose(A, (0, 2, 3, 1)).reshape(dl * 4, dr))
        k = q.shape[1]
        qa = q.reshape(dl, 2, 2, k)
        self.a[i] = qa.permute(0, 3, 1, 2) if hasattr(qa, 'permute') else np.transpose(qa, (0, 3, 1, 2))
        k2, dr2 = (r.shape[0], B.shape[1])
        self.a[i + 1] = _matmul(r, B.reshape(B.shape[0], dr2 * 4)).reshape(k2, dr2, 2, 2)

    def _right_step(self, i):
        xp = self.xp
        A, B = (self.a[i - 1], self.a[i])
        dl, dr = (B.shape[0], B.shape[1])
        m = B.permute(1, 2, 3, 0) if hasattr(B, 'permute') else np.transpose(B, (1, 2, 3, 0))
        q, r = self._qr(m.reshape(dr * 4, dl))
        k = q.shape[1]
        qb = q.reshape(dr, 2, 2, k)
        self.a[i] = qb.permute(3, 0, 1, 2) if hasattr(qb, 'permute') else np.transpose(qb, (3, 0, 1, 2))
        Am = _perm(A, (0, 2, 3, 1)).reshape(A.shape[0] * 4, A.shape[1])
        self.a[i - 1] = _perm(_matmul(Am, _perm(r, (1, 0))).reshape(A.shape[0], 2, 2, r.shape[0]), (0, 3, 1, 2))

    def canonicalize(self, center: int=0):
        for i in range(center):
            self._left_step(i)
        for i in range(self.n - 1, center, -1):
            self._right_step(i)
        self.center = center

    def move_center(self, target: int):
        if self.center is None:
            self.canonicalize(target)
            return
        while self.center < target:
            self._left_step(self.center)
            self.center += 1
        while self.center > target:
            self._right_step(self.center)
            self.center -= 1

    def _keep(self, s, record: bool=True) -> int:
        xp = self.xp
        weights = s * s
        if xp is np:
            weights = np.sort(weights)[::-1]
        else:
            weights = xp.sort(weights, descending=True).values
        total = float(weights.sum())
        if not total > 0.0 or not math.isfinite(total):
            raise FloatingPointError(f'singular spectrum total {total!r} at a split: the chain has lost its norm or holds NaNs')
        keep = int(s.shape[0])
        if total > 0 and self.cutoff > 0:
            if xp is np:
                tail = np.cumsum(weights[::-1])[::-1] / total
            else:
                tail = xp.flip(xp.cumsum(xp.flip(weights, (0,)), 0), (0,)) / total
            keep = int((tail > self.cutoff).sum())
        if self.max_bond is not None:
            keep = min(keep, int(self.max_bond))
        keep = max(1, keep)
        if record and keep < int(s.shape[0]) and (total > 0):
            kept = float(weights[:keep].sum())
            self.log_retained_ln += math.log(max(kept / total, 1e-300))
            self.truncations += 1
            if self.record is not None:
                self.record(total, total - kept)
        return keep

    def _right_svd_step(self, i):
        xp = self.xp
        A, B = (self.a[i - 1], self.a[i])
        dl, dr = (B.shape[0], B.shape[1])
        u, s, vh = xp.linalg.svd(_perm(B, (1, 2, 3, 0)).reshape(dr * 4, dl), full_matrices=False)
        k = self._keep(s)
        u, r = (u[:, :k], s[:k, None] * vh[:k])
        self.a[i] = _perm(u.reshape(dr, 2, 2, k), (3, 0, 1, 2))
        Am = _perm(A, (0, 2, 3, 1)).reshape(A.shape[0] * 4, A.shape[1])
        self.a[i - 1] = _perm(_matmul(Am, _perm(r, (1, 0))).reshape(A.shape[0], 2, 2, k), (0, 3, 1, 2))
        self.center = i - 1

    def apply_1q(self, site: int, gate, side: str):
        xp = self.xp
        t = self.a[site]
        if side == 'right':
            self.a[site] = _matmul(t, gate)
        else:
            self.a[site] = _matmul(gate, t)

    def apply_2q_adjacent(self, i: int, gate, side: str):
        xp = self.xp
        self.move_center(i)
        L, R = (self.a[i], self.a[i + 1])
        dl, dr = (L.shape[0], R.shape[1])
        dm = L.shape[1]
        Lm = _perm(L, (0, 2, 3, 1)).reshape(dl * 4, dm)
        Rm = _perm(R, (0, 2, 3, 1)).reshape(dm, 4 * dr)
        theta = _matmul(Lm, Rm).reshape(dl, 2, 2, 2, 2, dr)
        theta = _perm(theta, (0, 1, 3, 2, 4, 5)).reshape(dl, 4, 4, dr)
        if side == 'right':
            theta = _perm(_matmul(_perm(theta, (0, 3, 1, 2)), gate), (0, 2, 3, 1))
        else:
            theta = _perm(_matmul(gate, _perm(theta, (0, 3, 1, 2))), (0, 2, 3, 1))
        theta = _perm(theta.reshape(dl, 2, 2, 2, 2, dr), (0, 1, 3, 5, 2, 4))
        matrix = theta.reshape(4 * dl, 4 * dr)
        self._split(i, matrix, dl, dr)

    def _split(self, i: int, matrix, dl: int, dr: int):
        if not self._finite(matrix):
            raise FloatingPointError(f'pair matrix at bond {i} is not finite before the split (shape {tuple(matrix.shape)})')
        if self.decomp == 'gram':
            try:
                self._split_gram(i, matrix, dl, dr)
                if self._finite(self.a[i]) and self._finite(self.a[i + 1]):
                    return
                raise FloatingPointError('gram split produced non-finite site tensors')
            except Exception as ex:
                self.gram_fallbacks += 1
                self._last_gram_error = f'{type(ex).__name__}: {str(ex)[:120]}'
        self._split_svd(i, matrix, dl, dr)
        if not (self._finite(self.a[i]) and self._finite(self.a[i + 1])):
            raise FloatingPointError(f'svd split at bond {i} produced non-finite site tensors (shape {tuple(matrix.shape)})')

    def _finite(self, t) -> bool:
        xp = self.xp
        return bool(np.isfinite(t).all()) if xp is np else bool(xp.isfinite(t).all())

    def _split_svd(self, i: int, matrix, dl: int, dr: int):
        xp = self.xp
        u, s, vh = xp.linalg.svd(matrix, full_matrices=False)
        keep = self._keep(s)
        u, s, vh = (u[:, :keep], s[:keep], vh[:keep])
        self.a[i] = _perm(u.reshape(dl, 2, 2, keep), (0, 3, 1, 2))
        self.a[i + 1] = (s[:, None] * vh).reshape(keep, dr, 2, 2)
        self.center = i + 1

    def _split_gram(self, i: int, matrix, dl: int, dr: int):
        xp = self.xp
        mh = _conj_t(matrix)
        if matrix.shape[0] <= matrix.shape[1]:
            w, vec = xp.linalg.eigh(_matmul(matrix, mh))
            w, vec = (_flip(w, xp), _flip_cols(vec, xp))
            keep = self._keep(_sqrt_clamped(w, xp))
            uk = vec[:, :keep]
            self.a[i] = _perm(uk.reshape(dl, 2, 2, keep), (0, 3, 1, 2))
            self.a[i + 1] = _matmul(_conj_t(uk), matrix).reshape(keep, dr, 2, 2)
            self.center = i + 1
        else:
            w, vec = xp.linalg.eigh(_matmul(mh, matrix))
            w, vec = (_flip(w, xp), _flip_cols(vec, xp))
            keep = self._keep(_sqrt_clamped(w, xp))
            vk = vec[:, :keep]
            self.a[i] = _perm(_matmul(matrix, vk).reshape(dl, 2, 2, keep), (0, 3, 1, 2))
            self.a[i + 1] = _conj_t(vk).reshape(keep, dr, 2, 2)
            self.center = i
    _SWAP_AXES = {'none': (0, 1, 2, 3, 4, 5), 'left': (0, 1, 4, 3, 2, 5), 'right': (0, 3, 2, 1, 4, 5), 'both': (0, 3, 4, 1, 2, 5)}

    def _pair_theta(self, i: int):
        L, R = (self.a[i], self.a[i + 1])
        dl, dm, dr = (L.shape[0], L.shape[1], R.shape[1])
        Lm = _perm(L, (0, 2, 3, 1)).reshape(dl * 4, dm)
        Rm = _perm(R, (0, 2, 3, 1)).reshape(dm, 4 * dr)
        return (_matmul(Lm, Rm).reshape(dl, 2, 2, 2, 2, dr), dl, dr)

    def _svdvals(self, matrix):
        xp = self.xp
        if self.decomp == 'gram':
            mh = _conj_t(matrix)
            g = _matmul(matrix, mh) if matrix.shape[0] <= matrix.shape[1] else _matmul(mh, matrix)
            try:
                return _sqrt_clamped(xp.linalg.eigvalsh(g), xp)
            except Exception as ex:
                self.gram_fallbacks += 1
                self._last_gram_error = f'{type(ex).__name__}: {str(ex)[:120]}'
        if xp is np:
            return np.linalg.svd(matrix, compute_uv=False)
        return xp.linalg.svdvals(matrix)

    def local_unswap(self, hows=('left', 'right', 'both'), max_sweeps: int=12, on_swap=None, progress=None):
        if self.center is None:
            self.canonicalize(0)
        moves = sweeps = 0
        for sweep in range(int(max_sweeps)):
            applied = 0
            order = range(self.n - 1) if sweep % 2 == 0 else range(self.n - 2, -1, -1)
            for i in order:
                if progress is not None:
                    progress(sweep, i)
                self.move_center(i)
                theta, dl, dr = self._pair_theta(i)
                best, best_k = ('none', None)
                for how in ('none',) + tuple((h for h in hows if h != 'none')):
                    m = _perm(_perm(theta, self._SWAP_AXES[how]), (0, 1, 2, 5, 3, 4)).reshape(4 * dl, 4 * dr)
                    k = self._keep(self._svdvals(m), record=False)
                    if best_k is None or k < best_k:
                        best, best_k = (how, k)
                self._split(i, _perm(_perm(theta, self._SWAP_AXES[best]), (0, 1, 2, 5, 3, 4)).reshape(4 * dl, 4 * dr), dl, dr)
                if best != 'none':
                    applied += 1
                    if on_swap is not None:
                        on_swap(i, best)
            moves += applied
            sweeps += 1
            if applied == 0:
                break
        return (moves, sweeps)

    def apply_2q_long(self, i: int, j: int, gate, side: str):
        xp = self.xp
        self.move_center(i)
        g = _perm(gate.reshape(2, 2, 2, 2), (0, 2, 1, 3)).reshape(4, 4)
        u, s, vh = xp.linalg.svd(g, full_matrices=False)
        r = max(1, int((s > s[0] * 1e-13).sum()))
        A = (u[:, :r] * s[:r]).T.reshape(r, 2, 2)
        B = vh[:r].reshape(r, 2, 2)
        T = self.a[i]
        dl, dm = (T.shape[0], T.shape[1])
        T = _matmul(T[:, :, None], A[None, None]) if side == 'right' else _matmul(A[None, None], T[:, :, None])
        self.a[i] = T.reshape(dl, dm * r, 2, 2)
        eye = xp.eye(r, dtype=gate.dtype)
        if xp is not np:
            eye = eye.to(gate.device)
        for k in range(i + 1, j):
            T = self.a[k]
            dl, dm = (T.shape[0], T.shape[1])
            self.a[k] = (T[:, None, :, None] * eye[None, :, None, :, None, None]).reshape(dl * r, dm * r, 2, 2)
        T = self.a[j]
        dl, dm = (T.shape[0], T.shape[1])
        T = _matmul(T[:, None], B[None, :, None]) if side == 'right' else _matmul(B[None, :, None], T[:, None])
        self.a[j] = T.reshape(dl * r, dm, 2, 2)
        for k in range(i, j):
            self._left_step(k)
        self.center = j
        for k in range(j, i, -1):
            self._right_svd_step(k)

    def apply_circuit(self, circuit, side: str, to_matrix, progress=None):
        ops = []
        for inst in circuit.data:
            name = inst.operation.name
            if name in ('barrier', 'measure', 'delay'):
                continue
            qs = [circuit.find_bit(q).index for q in inst.qubits]
            ops.append((inst.operation, qs))
        seq = ops if side == 'left' else list(reversed(ops))
        for k, (op, qs) in enumerate(seq):
            if progress is not None:
                progress(k, len(seq), op.name)
            if len(qs) == 1:
                self.apply_1q(qs[0], to_matrix(op, 1), side)
                continue
            a, b = qs
            m = to_matrix(op, 2)
            lo = min(a, b)
            if a == lo:
                m = m.reshape(2, 2, 2, 2).permute(1, 0, 3, 2).reshape(4, 4) if hasattr(m, 'permute') else np.transpose(m.reshape(2, 2, 2, 2), (1, 0, 3, 2)).reshape(4, 4)
            if abs(a - b) == 1:
                self.apply_2q_adjacent(lo, m, side)
            else:
                self.apply_2q_long(lo, max(a, b), m, side)

    @staticmethod
    def legs_from_mpo(mpo):
        n = mpo.L
        up_prefix = mpo.upper_ind_id.split('{')[0]
        lo_prefix = mpo.lower_ind_id.split('{')[0]
        upper, lower = ([], [])
        for i in range(n):
            t = mpo[i]
            neighbours = set()
            if i > 0:
                neighbours |= set(mpo[i - 1].inds)
            if i < n - 1:
                neighbours |= set(mpo[i + 1].inds)
            phys = [ix for ix in t.inds if ix not in neighbours]
            ups = [ix for ix in phys if ix.startswith(up_prefix)]
            los = [ix for ix in phys if ix.startswith(lo_prefix)]
            if len(ups) != 1 or len(los) != 1:
                raise ValueError(f'position {i}: physical legs {phys} are not one upper ({up_prefix}*) and one lower ({lo_prefix}*)')
            upper.append(int(ups[0][len(up_prefix):]))
            lower.append(int(los[0][len(lo_prefix):]))
        return (upper, lower)

    @staticmethod
    def arrays_from_mpo(mpo):
        n = mpo.L
        upper, lower = LocalChain.legs_from_mpo(mpo)
        up_prefix = mpo.upper_ind_id.split('{')[0]
        lo_prefix = mpo.lower_ind_id.split('{')[0]
        out = []
        for i in range(n):
            t = mpo[i]
            bl = [ix for ix in t.inds if i > 0 and ix in mpo[i - 1].inds]
            br = [ix for ix in t.inds if i < n - 1 and ix in mpo[i + 1].inds]
            data = t.transpose(*bl + br + [f'{up_prefix}{upper[i]}', f'{lo_prefix}{lower[i]}']).data
            shape = list(data.shape)
            if i == 0:
                shape = [1] + shape
            if i == n - 1:
                shape = shape[:-2] + [1] + shape[-2:]
            out.append(data.reshape(shape))
        return out

    @staticmethod
    def mpo_from_arrays(arrays, like=None):
        import quimb.tensor as qtn
        n = len(arrays)
        tensors = []
        for i, a in enumerate(arrays):
            if i == 0:
                a = a.reshape(a.shape[1:])
            if i == n - 1:
                a = a.reshape(a.shape[:-3] + a.shape[-2:])
            tensors.append(a)
        kwargs = {}
        if like is not None:
            kwargs = dict(upper_ind_id=like.upper_ind_id, lower_ind_id=like.lower_ind_id, site_tag_id=like.site_tag_id)
        mpo = qtn.MatrixProductOperator(tensors, shape='lrud', **kwargs)
        if like is not None and getattr(like, 'exponent', 0.0):
            mpo.exponent = like.exponent
        return mpo
