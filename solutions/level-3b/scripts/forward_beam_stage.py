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
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'scripts'))
from peaked.mps import MPS, fiedler_order, random_order
from checkpoint_forward_pass import beam_bitstrings

class G:
    __slots__ = ('index', 'qubits', 'M')

    def __init__(self, index, qubits, M):
        self.index, self.qubits, self.M = (index, tuple(qubits), M)

    def matrix(self):
        return self.M

def load(qasm_path: str):
    from hqp_structure import load_circuit
    from qiskit.quantum_info import Operator
    qc = load_circuit(qasm_path)
    gates = []
    for i, inst in enumerate(qc.data):
        if inst.operation.name in ('barrier', 'delay'):
            continue
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if len(qs) > 2:
            raise ValueError(f'gate {inst.operation.name} on {len(qs)} qubits: not a 1q/2q circuit')
        M = np.asarray(Operator(inst.operation).data, dtype=complex)
        if len(qs) == 2:
            M = M.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
        gates.append(G(i, qs, M))
    return (qc.num_qubits, gates)

def orderings(n: int, gates, k: int, seed: int):
    out = [('native', list(range(n)))]
    if k >= 2:
        try:
            f = fiedler_order(n, gates)
            out.append(('fiedler', f))
            if k >= 4:
                out.append(('fiedler_rev', [n - 1 - p for p in f]))
        except Exception:
            out.append(('random0', random_order(n, seed)))
    for j in range(len(out), k):
        out.append((f'random{j}', random_order(n, seed + j)))
    return out

def log_amplitude(s: MPS, logical_bits: str) -> float:
    xp = s.xp
    v = xp.ones((1,), dtype=s.dtype)
    for p in range(s.n):
        b = int(logical_bits[s.qubit_at[p]])
        v = v @ s.A[p][:, b, :]
        nv = float(np.linalg.norm(np.asarray(v)))
        if nv == 0.0:
            return float('-inf')
        v = v / nv
        if p == 0:
            acc = np.log(nv)
        else:
            acc += np.log(nv)
    return float(acc + s.log_norm)

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('qasm')
    ap.add_argument('--chi', type=int, default=64)
    ap.add_argument('--beam', type=int, default=4096)
    ap.add_argument('--orderings', type=int, default=5)
    ap.add_argument('--pool', type=int, default=16)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--dtype', default='c64', choices=('c64', 'c128'))
    ap.add_argument('--emit-bits', default='')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    n, gates = load(a.qasm)
    dt = np.complex64 if a.dtype == 'c64' else np.complex128
    print(f'[fb] {Path(a.qasm).name}: {n} qubits, {len(gates)} gates ({sum((1 for g in gates if len(g.qubits) == 2))} two-qubit), chi {a.chi}, {a.orderings} orderings', flush=True)
    runs = []
    for name, perm in orderings(n, gates, a.orderings, a.seed):
        t0 = time.time()
        s = MPS(n, a.chi, xp=np, dtype=dt, cutoff=1e-06 if dt is np.complex64 else 1e-10, perm=perm, svd_method='svd')
        s.apply_gates(gates)
        cands = beam_bitstrings(s, a.beam)
        order = list(s.qubit_at)
        rows = []
        for logw, bits in cands[:64]:
            wire = [''] * n
            for site, ch in enumerate(bits):
                wire[order[site]] = ch
            rows.append(''.join(wire))
        w = [float(np.exp(2 * (lw - cands[0][0]))) for lw, _ in cands[:64]]
        ratio = 1.0 / w[1] if len(w) > 1 and w[1] > 0 else float('inf')
        h = hashlib.sha256(rows[0].encode()).hexdigest()[:12] if rows else '-'
        print(f'[fb] {name:8s}: bond {s.max_bond()} swaps {s.telemetry.swaps} beam {len(cands)} top1/top2 {ratio:.3f} rank1 sha {h} ({time.time() - t0:.0f} s)', flush=True)
        runs.append(dict(name=name, rows=rows, w=w, ratio=ratio, mps=s))
    pool = []
    for r in runs:
        for x in r['rows'][:a.pool]:
            if x not in pool:
                pool.append(x)
    la = {x: [log_amplitude(r['mps'], x) for r in runs] for x in pool}
    score = {x: sum(la[x]) for x in pool}
    ranked = sorted(pool, key=lambda x: -score[x])
    best_bits = ranked[0] if ranked else None
    stable = bool(ranked) and all((max(pool, key=lambda x: score[x] - la[x][k]) == best_bits for k in range(len(runs))))
    gap = score[ranked[0]] - score[ranked[1]] if len(ranked) > 1 else float('inf')
    agree = sum((1 for r in runs if r['rows'] and r['rows'][0] == best_bits))
    votes = {}
    for r in runs:
        if r['rows']:
            votes.setdefault(r['rows'][0], []).append(r)
    cons_bits, cons = max(votes.items(), key=lambda kv: (len(kv[1]), max((x['ratio'] for x in kv[1])))) if votes else (None, [])
    sha = lambda x: hashlib.sha256(x.encode()).hexdigest()[:12] if x else '-'
    print(f'[fb] product score over {len(pool)} pooled strings: winner sha {sha(best_bits)} is rank-1 in {agree}/{len(runs)} orderings; log-amplitude gap to the runner-up {gap:.2f} (summed over orderings); leave-one-out stable {stable}; plain rank-1 consensus {len(cons)}/{len(runs)} on sha {sha(cons_bits)}', flush=True)
    summary = dict(qasm=str(a.qasm), n=n, gates=len(gates), chi=a.chi, orderings=[dict(name=r['name'], ratio=r['ratio'], rank1_sha=hashlib.sha256(r['rows'][0].encode()).hexdigest()[:16] if r['rows'] else None) for r in runs], agree=agree, gap=gap, stable=stable, consensus_agree=len(cons), consensus_is_winner=cons_bits == best_bits)
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1) + '\n')
    if a.emit_bits and best_bits is not None:
        K = max(1, len(runs))
        wts = [float(np.exp(2 * (score[x] - score[ranked[0]]) / K)) for x in ranked[:64]]
        Path(a.emit_bits).write_text(json.dumps({'bits': ranked[:64], 'weights': wts, 'agree': agree, 'gap': gap, 'stable': stable, 'consensus_agree': len(cons), 'consensus_is_winner': cons_bits == best_bits}))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
