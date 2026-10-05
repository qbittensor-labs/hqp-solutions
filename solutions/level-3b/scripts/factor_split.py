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
import itertools
import sys
import numpy as np
EXHAUSTIVE_MAX = 8

def _try_split(T: np.ndarray, m: int, thr: float):
    max_size = m // 2 if m <= EXHAUSTIVE_MAX else min(2, m // 2)
    for size in range(1, max_size + 1):
        for ao in itertools.combinations(range(m), size):
            for ai in itertools.combinations(range(m), size):
                bo = [i for i in range(m) if i not in ao]
                bi = [i for i in range(m) if i not in ai]
                perm = list(ao) + [m + i for i in ai] + bo + [m + i for i in bi]
                M = np.transpose(T, perm).reshape(4 ** size, 4 ** (m - size))
                ev = np.linalg.eigvalsh(M @ M.conj().T)[::-1]
                tot = float(ev.sum())
                if tot > 0 and float(max(ev[1:].sum(), 0.0)) / tot < thr:
                    U, s, Vh = np.linalg.svd(M, full_matrices=False)
                    A = (U[:, 0] * np.sqrt(s[0])).reshape([2] * (2 * size))
                    B = (Vh[0] * np.sqrt(s[0])).reshape([2] * (2 * (m - size)))
                    return (list(ao), list(ai), A, bo, bi, B)
    return None

def split_factor(G: np.ndarray, k: int, thr: float=1e-09):

    def rec(outs, ins, T):
        m = len(outs)
        if m <= 1:
            return [(outs, ins, T.reshape(2 ** m, 2 ** m))]
        r = _try_split(T, m, thr)
        if r is None:
            return [(outs, ins, T.reshape(2 ** m, 2 ** m))]
        ao, ai, A, bo, bi, B = r
        return rec([outs[i] for i in ao], [ins[i] for i in ai], A) + rec([outs[i] for i in bo], [ins[i] for i in bi], B)
    pieces = rec(list(range(k)), list(range(k)), G.reshape([2] * (2 * k)))
    out = []
    for outs, ins, U in pieces:
        m = len(outs)
        u, s, vh = np.linalg.svd(U)
        out.append((outs, ins, u @ vh, float(s.mean())))
    return out

def rebuild(pieces, k: int) -> np.ndarray:
    T = np.ones([1])
    order_out, order_in = ([], [])
    for outs, ins, U, sc in pieces:
        m = len(outs)
        T = np.multiply.outer(T, (sc * U).reshape([2] * (2 * m)))
        order_out += outs
        order_in += ins
    T = T.reshape([2] * (2 * k))
    axes, pos = ([], 0)
    cur = []
    for outs, ins, U, sc in pieces:
        cur += [('o', o) for o in outs] + [('i', i) for i in ins]
    target = [('o', i) for i in range(k)] + [('i', i) for i in range(k)]
    perm = [cur.index(t) for t in target]
    return np.transpose(T, perm).reshape(2 ** k, 2 ** k)

def _selftest():
    from scipy.stats import unitary_group
    rng = np.random.default_rng(3)
    k = 6
    G3, G2, G1 = (unitary_group.rvs(8, random_state=1), unitary_group.rvs(4, random_state=2), unitary_group.rvs(2, random_state=4))
    T = np.multiply.outer(np.multiply.outer(G3.reshape([2] * 6), G2.reshape([2] * 4)), G1.reshape([2] * 2)).reshape([2] * 12)
    cur = [('o', 1), ('o', 3), ('o', 4), ('i', 1), ('i', 3), ('i', 4), ('o', 0), ('o', 5), ('i', 0), ('i', 5), ('o', 2), ('i', 2)]
    target = [('o', i) for i in range(k)] + [('i', i) for i in range(k)]
    G = np.transpose(T, [cur.index(t) for t in target]).reshape(2 ** k, 2 ** k)
    P = np.eye(2 ** k)[:, rng.permutation(2 ** k)]
    wire = [5, 0, 1, 2, 3, 4]
    Tw = np.transpose(G.reshape([2] * 12), wire + list(range(6, 12))).reshape(2 ** k, 2 ** k)
    ok = True
    for name, M, want in (('3+2+1 interleaved', G, [1, 2, 3]), ('same with a wire permutation', Tw, [1, 2, 3]), ('dense random 6-qubit', unitary_group.rvs(64, random_state=9), [6])):
        pcs = split_factor(M, k)
        sizes = sorted((len(p[0]) for p in pcs))
        err = np.linalg.norm(rebuild(pcs, k) - M) / np.linalg.norm(M)
        good = sizes == want and err < 1e-10
        ok &= good
        print(f"{('PASS' if good else 'FAIL')}  {name}: pieces {sizes}, rebuild error {err:.1e}")
    return ok
if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] == '--selftest':
        sys.exit(0 if _selftest() else 1)
