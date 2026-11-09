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

"""(level-4 submission copy of peakpilot research/priv/hard3/mkred_multi.py; only change: budget from MKRED_BUDGET)
Excise several blocks (winner measure_D) and write the reduced circuit. usage: mkred_multi.py <qasm> <census.json> <out_prefix> c1[:win] c2[:win] ..."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.environ.setdefault("EXC_MIN_EVIDENCE", "3"); os.environ.setdefault("EXC_PEEL_MODE", "close"); os.environ["HQP_D_ENGINE"] = "sync_only"; os.environ["EXC_EXTEND"] = "1"
import excise as X, excision_solver as ES, sectioned as SEC
from qiskit import qasm2
from qiskit.circuit.library import UGate
q, cj, out = sys.argv[1:4]; specs = sys.argv[4:]
circ = qasm2.load(q, custom_instructions=[qasm2.CustomInstruction('u', 3, 1, lambda t, p, l: UGate(t, p, l), builtin=True)])
n = circ.num_qubits; g = ES._circ_to_gates(circ)
blocks = json.load(open(cj))["blocks"]; X.EXC_MAX_DIST = int(os.environ.get("EXC_MAX_DIST", "1500")); X.EXC_TWIN_TOL = 1e-3
cuts, corrs, infos = [], [], []
# trivial-cut guard (2026-10-01, from the core_scan ladder validation): a decode is accepted only if its cut is
# non-empty and holds both legs of >= MKRED_MIN_OWN (default 0.25) of the module's own mirror pairs (excise.cz_twins).
# Negative controls before the guard: a random circuit 'excised' 2 CZ / 1 of 56 pairs at model error 0; an off-centre
# hard_3 centre 'excised' an empty cut. MKRED_MIN_OWN=0 restores the old model-error-only acceptance.
MIN_OWN = float(os.environ.get("MKRED_MIN_OWN", "0.25"))
def own_inside(cut, c, tau):
    W, _ = X.wire_ops(g, n)
    S = {W[x][k] for x in range(n) for k in range(cut.s[x], cut.e[x] + 1) if 0 <= k < len(W[x])}
    own = X.cz_twins(g, n, c, tau)
    return S, sum(1 for r, l in own.items() if r in S and l in S), len(own)
for sp in specs:
    c, _, w = sp.partition(":"); c = int(c); wins = [int(w)] if w else [1200, 1800]
    tau = next(b["tau"] for b in blocks if b["centre"] == c); best = None
    for win in wins:
        cut, corr, info = ES.measure_D(g, n, c, tau, win, None, lambda m: None, budget=float(os.environ.get("MKRED_BUDGET", "900")), cutoff=6e-4, seed=123)
        if cut is not None and MIN_OWN > 0:
            S, ins, tot = own_inside(cut, c, tau)
            if not S or ins < MIN_OWN * max(1, tot):
                print(f"centre {c} win {win}: TRIVIAL cut rejected ({len(S)} gates, {ins}/{tot} own pairs inside, model_err {info.get('model_err')})", flush=True)
                cut = None
        print(f"centre {c} win {win}: {'EXCISED' if cut is not None else 'REJECTED'} model_err {info.get('model_err')} cz {info.get('cz_excised')} resid_gates {info.get('resid_gates')}", flush=True)
        if cut is not None and info["model_err"] < float(os.environ.get("MKRED_MAX_ERR", "0.1")) and (best is None or info["model_err"] < best[2]["model_err"]): best = (cut, corr, info, win)
    assert best, f"centre {c} did not decode"
    cuts.append(best[0]); corrs.append(best[1]); infos.append({"centre": c, "win": best[3], "model_err": best[2]["model_err"], "cz": best[2]["cz_excised"]})
secs, Pinv = SEC.sectioned(g, n, cuts, corrections=corrs)
red = [x for s in secs for x in s]
print(f"reduced: gates {len(g)} -> {len(red)}, CZ {sum(t=='cz' for t,_,_ in g)} -> {sum(t=='cz' for t,_,_ in red)}; sections {[len(s) for s in secs]}")
L = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"] + [f"cz q[{qq[0]}],q[{qq[1]}];" if t == "cz" else f"u({p[0]!r},{p[1]!r},{p[2]!r}) q[{qq[0]}];" for t, qq, p in red]
open(out + ".qasm", "w").write("\n".join(L) + "\n"); json.dump({"Pinv": list(Pinv), "blocks": infos}, open(out + ".json", "w"), indent=1)
