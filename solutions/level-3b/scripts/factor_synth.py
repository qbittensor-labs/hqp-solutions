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
import sys
import numpy as np

def _embed(g: np.ndarray, i: int, k: int) -> np.ndarray:
    return np.kron(np.kron(np.eye(2 ** i), g), np.eye(2 ** (k - i - 2)))

def _pairs(k: int, depth: int) -> list[int]:
    out = []
    for d in range(depth):
        out += list(range(d % 2, k - 1, 2))
    return out

def rebuild_brickwork(gates, k: int) -> np.ndarray:
    C = np.eye(2 ** k, dtype=complex)
    for i, _j, g in gates:
        C = _embed(g, i, k) @ C
    return C

def _err(U, C) -> float:
    t = np.trace(U.conj().T @ C)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(U * ph - C) / np.sqrt(U.shape[0]))

def _fit(U: np.ndarray, k: int, pos: list[int], sweeps: int, rng) -> tuple[list[np.ndarray], float]:
    D = 2 ** k
    gs = []
    for _ in pos:
        X = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        q, _ = np.linalg.qr(np.eye(4) + 0.3 * X)
        gs.append(q)
    Ud = U.conj().T
    last = None
    for sw in range(sweeps):
        order = range(len(pos)) if sw % 2 == 0 else reversed(range(len(pos)))
        for m in order:
            A = np.eye(D, dtype=complex)
            for t in range(len(pos) - 1, m, -1):
                A = A @ _embed(gs[t], pos[t], k)
            B = np.eye(D, dtype=complex)
            for t in range(m - 1, -1, -1):
                B = B @ _embed(gs[t], pos[t], k)
            M = B @ Ud @ A
            i = pos[m]
            T = M.reshape([2] * (2 * k))
            rest = [s for s in range(k) if s not in (i, i + 1)]
            T = np.transpose(T, [i, i + 1] + rest + [k + i, k + i + 1] + [k + s for s in rest])
            T = T.reshape(4, 2 ** (k - 2), 4, 2 ** (k - 2))
            E = np.einsum('arbr->ab', T)
            W, _s, Vh = np.linalg.svd(E)
            gs[m] = (W @ Vh).conj().T
        C = rebuild_brickwork([(pos[t], pos[t] + 1, gs[t]) for t in range(len(pos))], k)
        e = _err(U, C)
        if e < 1e-09 or (last is not None and abs(last - e) < 1e-12):
            break
        last = e
    return (gs, e)

def synth_brickwork(U: np.ndarray, k: int, tol: float=0.0003, max_depth: int | None=None, restarts: int=3, sweeps: int=400, seed: int=0):
    if k < 3:
        return None
    U = np.asarray(U, dtype=complex)
    max_depth = max_depth or 2 * k
    rng = np.random.default_rng(seed)
    for depth in range(1, max_depth + 1):
        pos = _pairs(k, depth)
        if 3 * len(pos) >= _generic_cz(k):
            return None
        for _r in range(restarts):
            gs, e = _fit(U, k, pos, sweeps, rng)
            if e < tol:
                gates = [(pos[t], pos[t] + 1, gs[t]) for t in range(len(pos))]
                C = rebuild_brickwork(gates, k)
                t = np.trace(U.conj().T @ C)
                ph = np.conj(t / abs(t))
                i0, j0, g0 = gates[0]
                gates[0] = (i0, j0, g0 * ph)
                return gates
    return None

def _generic_cz(k: int) -> int:
    return {1: 0, 2: 3, 3: 20, 4: 100, 5: 444, 6: 1800}.get(k, 10 ** 5)

def _selftest() -> bool:
    from scipy.stats import unitary_group
    rng = np.random.default_rng(5)
    ok = True
    for k, depth in ((3, 2), (4, 3), (5, 3), (5, 4)):
        pos = _pairs(k, depth)
        gates = [(i, i + 1, unitary_group.rvs(4, random_state=int(rng.integers(1 << 30)))) for i in pos]
        U = rebuild_brickwork(gates, k) * np.exp(0.7j)
        got = synth_brickwork(U, k, tol=1e-08, max_depth=depth, restarts=4)
        e = _err(U, rebuild_brickwork(got, k)) if got else None
        exact = got is not None and np.linalg.norm(rebuild_brickwork(got, k) - U) / np.sqrt(2 ** k) < 1e-06
        good = got is not None and len(got) <= len(pos) and exact
        ok &= good
        print(f"{('PASS' if good else 'FAIL')}  k={k} brickwork depth {depth}: recovered {(None if got is None else len(got))} gates, error {e}")
    from scipy.stats import unitary_group as ug
    R = ug.rvs(32, random_state=3)
    got = synth_brickwork(R, 5, tol=0.0003, restarts=1, sweeps=100)
    good = got is None
    ok &= good
    print(f"{('PASS' if good else 'FAIL')}  dense random 5-qubit rejected ({('None' if got is None else len(got))})")
    return ok
if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] == '--selftest':
        sys.exit(0 if _selftest() else 1)
