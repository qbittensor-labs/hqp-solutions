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

"""Counters (c) and the loose / lowered-gate census: the winner census with options (run inside peakpilot's env; hqpgen
never imports it).

  --directed   DIRECTED tau matching (counter c, below)
  --tol T      twin tolerance of the centre scan and of tau discovery (winner: 1e-6; H3 detuning moves twins by
               1e-3..1e-1)
  EXC_MIN_FRAC (env) the twin-fraction gate of the centre scan (winner 0.25; counter d lowers it)

Directed matching (counter c):

The winner's discover_tau builds M[a, b] = u3 twins between left wire a and right wire b and then solves a maximum-
weight PERFECT MATCHING of the symmetrised M + M^T (an involution: wires paired two by two). A module whose core
permutation is not an involution (3-cycles, a random derangement, one n-cycle) mirrors wire a onto tau(a) != tau^-1(a),
so the symmetrised matching returns a wrong tau. Here tau = linear_sum_assignment on M itself (a general permutation:
left wire a -> right wire tau[a]); the CZ-consistency refinement (gencz) is kept with its directed count C (not
C + C^T). The core-count evidence (gencore) is undirected and is dropped. Everything else (centre scan, _choose_tau,
thresholds) is the winner's code, unchanged. For an involution the directed and symmetric answers agree.

usage: uv run --project <peakpilot> python _census_x.py <qasm> <out.json> [--directed] [--tol T] [--centres c1,c2]
       (env WINNER_SRC = winner src dir)
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.environ["WINNER_SRC"])
import excise as X  # noqa: E402
import excision_solver as ES  # noqa: E402
import v13_solver as V  # noqa: E402
from qiskit import qasm2  # noqa: E402
from qiskit.circuit.library import UGate  # noqa: E402


def _assign_directed(S):
    S = S.copy()
    np.fill_diagonal(S, -1e9)
    r, col = linear_sum_assignment(-S)
    tau = list(range(S.shape[0]))
    for a, b in zip(r, col):
        tau[a] = int(b)
    return tau


def discover_tau_directed(g, n, c, half=400, tol=1e-6, mode="all"):
    M = X._twin_counts(g, n, c, half, tol)
    if mode == "all":
        tau = _assign_directed(M)
    else:
        S0 = X._twin_counts(g, n, c, half, tol, generic_only=True)
        tau = _assign_directed(S0)
        if mode in ("gencz", "gencore"):
            lam = float(os.environ.get("EXC_TAU_CZ_LAMBDA", "1.0"))
            for _ in range(int(os.environ.get("EXC_TAU_CZ_ITERS", "6"))):
                C = X._cz_consistency(g, n, c, half, tau)
                t2 = _assign_directed(S0 + lam * C)
                if t2 == tau:
                    break
                tau = t2
    score = sum(M[a, tau[a]] for a in range(n))
    lo, _hi = X._window(g, c, half)
    nleft = sum(1 for i in range(lo, c) if g[i][0] == "u")
    return tau, float(score), float(score) / max(nleft, 1)


args = sys.argv[1:]
directed = "--directed" in args
tol = float(args[args.index("--tol") + 1]) if "--tol" in args else 1e-6
q, out = args[0], args[1]                      # positional first: <qasm> <out.json> [--directed] [--tol T]
_orig_discover, _orig_choose = X.discover_tau, V._choose_tau
if directed:
    X.discover_tau = discover_tau_directed

    def _choose_tau_directed(gates, n, c, centres, log, tol=1e-6):
        """The winner's _choose_tau scores disagreeing candidates by the synchronous cut, whose pair closure assumes
        an involution (a correct directed tau scored 0 mirror pairs there, MEASURED 09-30 on s1 H4_tau medium); here
        the candidate with the largest twin fraction wins (windows 600 / 900 / 1200 x modes all / generic / gencz)."""
        best = None
        for h in [int(x) for x in os.environ.get("V13_TAU_HALVES", "600,900,1200").split(",") if x.strip()]:
            for mode in ("all", "generic", "gencz"):
                tau, _, fr = discover_tau_directed(gates, n, c, h, tol=tol, mode=mode)
                if best is None or fr > best[1]:
                    best = (tau, fr)
            # the identity candidate (the matching forbids fixed points): after every module is excised the outer
            # mirror is the identity (the winner keeps a 'fixed' candidate for the same reason)
            M = X._twin_counts(gates, n, c, h, tol)
            lo, _hi = X._window(gates, c, h)
            nleft = sum(1 for i in range(lo, c) if gates[i][0] == "u")
            fr_id = float(sum(M[a, a] for a in range(n))) / max(nleft, 1)
            if fr_id > best[1]:
                best = (list(range(n)), fr_id)
        log(f"directed: block {c}: best twin fraction {best[1]:.3f}")
        return best

    def _choose_tau_hybrid(gates, n, c, centres, log, tol=1e-6):
        """Directed tau, unless it is (almost) an involution: then the winner's symmetric matching, which enforces
        tau(tau(w)) = w and so repairs single mis-paired wires (MEASURED 09-30, s1 48 q baseline seed 2: directed
        46/48 wires with a 3-cycle error, symmetric 48/48; p12 52 q seed 3: directed 48/52 with 44/52 involutive wires).
        Share of involutive wires >= 0.5 -> symmetric (3-cycles and n-cycles have none, a random derangement few)."""
        tau_d, fr_d = _choose_tau_directed(gates, n, c, centres, log, tol=tol)
        inv_share = sum(1 for w in range(n) if tau_d[tau_d[w]] == w) / n
        if inv_share < 0.5:
            return tau_d, fr_d
        X.discover_tau = _orig_discover
        try:
            return _orig_choose(gates, n, c, centres, log, tol=tol)
        finally:
            X.discover_tau = discover_tau_directed

    V._choose_tau = _choose_tau_hybrid
t0 = time.time()
circ = qasm2.load(q, custom_instructions=[qasm2.CustomInstruction("u", 3, 1, lambda t, p, l: UGate(t, p, l), builtin=True)])
n = circ.num_qubits
g = ES._circ_to_gates(circ)
if "--centres" in args:                       # centres given (e.g. found at another tolerance): tau only
    found = [(int(c), None, 0.0) for c in args[args.index("--centres") + 1].split(",") if c]
else:
    found = X.discover_centres(g, n, tol=tol, log=lambda m: print(m[:2000], flush=True))
res = []
for c, _, fr in found:
    tau, fr2 = V._choose_tau(g, n, int(c), [int(c2) for c2, _, _ in found], lambda m: print(m, flush=True), tol=tol)
    res.append({"centre": int(c), "centre_frac_pos": c / len(g), "frac": fr if fr else fr2, "tau": tau,
                "moved": sum(1 for a in range(n) if tau[a] != a),
                "involution": all(tau[tau[a]] == a for a in range(n))})
    print(f"block {c}: frac {fr:.3f} involution {res[-1]['involution']}", flush=True)
json.dump({"qasm": q, "n": n, "gates": len(g), "blocks": res, "secs": time.time() - t0,
           "method": {"directed": directed, "tol": tol, "min_frac": os.environ.get("EXC_MIN_FRAC", "0.25")}},
          open(out, "w"), indent=1)
