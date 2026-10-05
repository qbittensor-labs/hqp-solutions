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

"""Pauli-path consistency gate for a candidate answer.

Heisenberg-picture <Z_q> of the reduced circuit, propagated with a truncated Pauli sum. It is a
DIFFERENT error structure from the truncated-state readouts (no bond, no chain order, no smear onto
neighbouring strings), so it catches what they cannot see in themselves: MEASURED on the hard circuit
h0, where the v9 state readout returned a string 6 bits wrong, the Pauli-path string is within 2 bits of
the truth and the agreement score S(x) = sum_q <Z_q>(1-2x_q) puts the truth at 10.71 of a possible
10.86 while 2000 random 6-bit-wrong strings average 8.0 and none reach it.

It is NOT an arbiter: a marginal is delta*s_q + (1-delta)*m_q, and where the non-peak part m_q is
biased the sign is wrong even when computed exactly (reduced d3_s1: 43/48, identical 5 wrong bits at
3x the paths, one of them at +0.118). Use the RATIO S(answer)/S_max as a gate, not the signs.
"""
import time

import numpy as np

import pauli_prop as pp


def marginals(red_gates, n, K=2_000_000, budget_s=1500.0, log=None):
    """red_gates: [('cz',(a,b),None) | ('u',(q,),(theta,phi,lam))] in circuit order -> z[n], done."""
    gates = []
    for t, q, p in red_gates:
        if t == "cz":
            gates.append(("cz", q[0], q[1]))
        else:
            th, ph, la = p
            c, s = np.cos(th / 2), np.sin(th / 2)
            U = np.array([[c, -np.exp(1j * la) * s],
                          [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]], complex)
            gates.append(("u3", q[0], U))
    t0 = time.time()
    z = np.zeros(n)
    done = 0
    for q in range(n):
        left = budget_s - (time.time() - t0)
        if left <= 0:
            break
        per = left / (n - q) + 30.0
        z[q] = pp.evolve_expectation(gates, n, q, K=K, log=None, deadline=time.time() + per).real
        done += 1
    if log:
        log(f"  [pps] {done}/{n} marginals in {time.time()-t0:.0f}s, mean|<Z>| {np.abs(z[:done]).mean():.3f}, "
            f"max {np.abs(z[:done]).max():.3f}")
    return z, done


def agreement(z, bits):
    """S(x)/S_max in [-1, 1] and the number of sign disagreements among the 10 strongest marginals."""
    sgn = np.array([1.0 if b == "0" else -1.0 for b in bits])
    smax = float(np.abs(z).sum())
    ratio = float((z * sgn).sum()) / smax if smax > 0 else 0.0
    top = np.argsort(-np.abs(z))[:10]
    strong_wrong = int(sum(1 for i in top if np.sign(z[i]) != sgn[i] and z[i] != 0))
    ham = int(sum(1 for i in range(len(bits)) if z[i] != 0 and np.sign(z[i]) != sgn[i]))
    return {"ratio": ratio, "strong_wrong": strong_wrong, "sign_disagreements": ham}
