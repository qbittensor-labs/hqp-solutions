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

"""Pauli-path opinions on the reduced circuit (pp2 engine): marginals, agreement ratio, and PAIR VOTES.

PAIR VOTES. A marginal is <Z_i> = d*s_i + (1-d)*m_i: where the non-peak part m_i is biased the sign of the
marginal is wrong even when it is computed exactly (reduced d3_s1: q38 at +0.118, identical at 3x the paths).
The CONNECTED correlator with an anchor qubit a does not have that problem:
    C_ia = <Z_i Z_a> - <Z_i><Z_a> = d(1-d)(s_i - m_i)(s_a - m_a) + (1-d)*cov_nonpeak(i, a)
and |m| < 1 always, so sign(C_ia) = s_i*s_a whenever the non-peak covariance is small. With s_a known (anchors =
strongest marginals on which all candidates agree) every anchor casts an independent vote on bit i. Used ONLY on
the few bits where the state readouts' candidates differ -- exactly the low-margin rivals a truncated state
cannot separate -- so it costs a handful of propagations, not 1128.
"""
import os
import time

import numpy as np

import pp2

K_DEFAULT = int(os.environ.get("V12_PPS_K", "2000000"))


def compile_reduced(red_gates):
    gates = []
    for t, q, p in red_gates:
        if t == "cz":
            gates.append(("cz", q[0], q[1]))
        else:
            th, ph, la = p
            c, s = np.cos(th / 2), np.sin(th / 2)
            U = np.array([[c, -np.exp(1j * la) * s], [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]], complex)
            gates.append(("u3", q[0], U))
    return pp2.compile_gates(gates)


def marginals(cg, n, K=None, budget_s=1200.0, log=None):
    K = K or K_DEFAULT
    t0 = time.time(); z = np.zeros(n); done = 0
    for q in range(n):
        left = budget_s - (time.time() - t0)
        if left <= 0:
            break
        z[q] = pp2.evolve(cg, [q], K=K, deadline=time.time() + left / (n - q) + 20.0).real
        done += 1
    if log:
        log(f"  [pps2] {done}/{n} marginals in {time.time()-t0:.0f}s, mean|<Z>| {np.abs(z[:done]).mean():.3f}, max {np.abs(z[:done]).max():.3f}")
    return z, done


def agreement(z, bits):
    sgn = np.array([1.0 if b == "0" else -1.0 for b in bits])
    smax = float(np.abs(z).sum())
    ratio = float((z * sgn).sum()) / smax if smax > 0 else 0.0
    top = np.argsort(-np.abs(z))[:10]
    return {"ratio": ratio, "strong_wrong": int(sum(1 for i in top if z[i] != 0 and np.sign(z[i]) != sgn[i])),
            "sign_disagreements": int(sum(1 for i in range(len(bits)) if z[i] != 0 and np.sign(z[i]) != sgn[i]))}


def pair_votes(cg, n, z, cands_red, K=None, budget_s=900.0, n_anchor=6, log=None):
    """cands_red: candidate strings in the REDUCED frame (first = current leader). Returns
    {qubit: {"votes0": k, "votes1": k, "C": [...]}} for every qubit on which the candidates differ."""
    K = K or K_DEFAULT
    lead = cands_red[0]
    disputed = sorted({i for c in cands_red[1:] for i in range(n) if c[i] != lead[i]})
    if not disputed:
        return {}
    agreed = [a for a in range(n) if a not in disputed]
    anchors = sorted(agreed, key=lambda a: -abs(z[a]))[:n_anchor]
    t0 = time.time(); out = {}
    total = len(disputed) * len(anchors); k = 0
    for i in disputed:
        rec = {"votes0": 0, "votes1": 0, "C": []}
        for a in anchors:
            left = budget_s - (time.time() - t0)
            if left <= 5:
                break
            zz = pp2.evolve(cg, [i, a], K=K, deadline=time.time() + left / max(total - k, 1) + 15.0).real
            k += 1
            C = zz - z[i] * z[a]
            s_a = 1.0 if lead[a] == "0" else -1.0
            s_i = np.sign(C) * s_a
            rec["C"].append(round(float(C), 4))
            if s_i > 0:
                rec["votes0"] += 1
            elif s_i < 0:
                rec["votes1"] += 1
        out[i] = rec
        if log:
            log(f"  [pps2] disputed qubit {i}: marginal {z[i]:+.4f}, pair votes 0:{rec['votes0']} 1:{rec['votes1']} C={rec['C']}")
    return out
