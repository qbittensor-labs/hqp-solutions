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

"""Chain orders for ensemble members: good AND different.

A random chain order decorrelates truncation errors between members but is a poor order (more swap
work, more entanglement across each cut, lower fidelity per member). A spectral order followed by a
seeded local search gives orders that are all good and all different: the search is stochastic, so
each seed lands in its own local optimum of the same minimum-linear-arrangement objective.
"""
import numpy as np


def interaction_matrix(qc):
    n = qc.num_qubits
    idx = {qb: i for i, qb in enumerate(qc.qubits)}
    W = np.zeros((n, n))
    for inst in qc.data:
        if len(inst.qubits) == 2:
            a, b = idx[inst.qubits[0]], idx[inst.qubits[1]]
            W[a, b] += 1
            W[b, a] += 1
    return W


def _cost(W, pos):
    d = np.abs(pos[:, None] - pos[None, :])
    return float((W * d).sum()) / 2.0


def good_order(qc, seed=0, iters=6000, temperature=0.02):
    """Returns perm with perm[q] = chain position of qubit q."""
    W = interaction_matrix(qc)
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    L = np.diag(W.sum(1)) - W
    _, v = np.linalg.eigh(L)
    fied = v[:, 1] + 1e-3 * rng.standard_normal(n)        # seed-dependent tie-breaking
    if rng.random() < 0.5:
        fied = -fied                                       # chain direction is a free choice
    order = np.argsort(fied)                               # order[p] = qubit at position p
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(n)
    cur = _cost(W, pos)
    T0 = temperature * cur / max(n, 1)
    for it in range(iters):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        pos[i], pos[j] = pos[j], pos[i]
        c = _cost(W, pos)
        T = T0 * (1.0 - it / iters)
        if c <= cur or (T > 0 and rng.random() < np.exp(-(c - cur) / T)):
            cur = c
        else:
            pos[i], pos[j] = pos[j], pos[i]
    return [int(x) for x in pos], cur


def random_order(n, seed):
    return [int(x) for x in np.random.default_rng(seed).permutation(n)]
