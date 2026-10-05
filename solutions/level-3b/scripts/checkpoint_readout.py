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
import re
import io
import json
import math
import pickle
import sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'scripts'))
from peaked.core_mpo import apply_core, portable_to_chain
from peaked.mps import MPS, dense_state
from checkpoint_forward_pass import beam_bitstrings
from checkpoint_to_reduced_qasm import layer_gates, load_layers

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('--chi', type=int, default=1024)
    ap.add_argument('--cutoff', type=float, default=1e-10)
    ap.add_argument('--beam', type=int, default=4096)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--manifest', default=str(_ROOT / 'dispatch/c3/inputs/truth_manifest.json'))
    ap.add_argument('--instance', default='')
    ap.add_argument('--dense-qasm', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--mpo-gram', type=int, default=1)
    ap.add_argument('--emit-bits', default='')
    a = ap.parse_args()
    pay = pickle.load(open(a.checkpoint, 'rb'))
    c = pay.get('counters') or {}
    il, ir = (int(pay['ii_left']), int(pay['ii_right']))
    ll, lr = (load_layers(pay['layers_left']), load_layers(pay['layers_right']))
    fr = [int(v) for v in pay['frame_right']]
    n = len(fr)
    chain = portable_to_chain(pay)
    labels = list(portable_to_chain.last_leg_labels)
    W = [np.asarray(t, dtype=complex) for t in chain]
    mb = max((t.shape[0] for t in W))
    print(f"[ro] {Path(a.checkpoint).name}: work {c.get('work_ops_absorbed_total')}/{c.get('work_ops_total')}, cycles {c.get('unswap_cycles')}, MPO bond {mb}, pending layers {len(ll) - il} left / {len(lr) - ir} right", flush=True)
    fwd_left, fwd_right = ([], [])
    for lay in reversed(ll[il:]):
        fwd_left += layer_gates(lay.inverse())
    for lay in lr[ir:]:
        fwd_right += layer_gates(lay)
    nl = sum((1 for k, _, _ in fwd_left if k == 'gate'))
    nr = sum((1 for k, _, _ in fwd_right if k == 'gate'))
    print(f'[ro] pending blocks: {nl} left, {nr} right', flush=True)
    if a.device.startswith('cuda'):
        import torch
        from peaked.torch_np import TorchNP
        xp, dt = (TorchNP(a.device), torch.complex128)
    else:
        xp, dt = (np, np.complex128)
    s = MPS(n, a.chi, xp=xp, dtype=dt, cutoff=a.cutoff, svd_method='svd')

    def run(gates):
        for kind, ws, m in gates:
            if kind == 'swap':
                pi = list(range(n))
                pi[ws[0]], pi[ws[1]] = (ws[1], ws[0])
                s.permute_logical(pi)
            else:
                G4 = xp.asarray(np.ascontiguousarray(m.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2)), dtype=dt)
                s.apply_2q(ws[0], ws[1], G4)
    run(fwd_left)
    print(f'[ro] after pending left: bond {s.max_bond()}', flush=True)
    FL = [0] * n
    FR = [0] * n
    for i, (up, lo) in enumerate(labels):
        FL[lo] = i
        FR[up] = i
    s.svd_method = 'gram' if a.mpo_gram else 'svd'
    cost = apply_core(s, W, FL, FR)
    s.svd_method = 'svd'
    print(f'[ro] after MPO: bond {s.max_bond()} (frame swaps {cost}, log10 norm {s.log_norm / math.log(10):.4f})', flush=True)
    run(fwd_right)
    print(f'[ro] after pending right: bond {s.max_bond()}', flush=True)
    summary = {'checkpoint': str(a.checkpoint), 'counters': c, 'mpo_bond': mb, 'pending': [nl, nr], 'chi': a.chi, 'final_bond': s.max_bond()}
    if a.dense_qasm:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        oc = QuantumCircuit.from_qasm_file(a.dense_qasm)
        oc.remove_final_measurements(inplace=True)
        psi_o = np.asarray(Statevector(oc).data)
        psi = dense_state(s).reshape([2] * n)
        psi_l = np.transpose(psi, [fr[q] for q in range(n)])
        f_big = abs(np.vdot(psi_l.reshape(-1), psi_o[::-1] if False else np.transpose(psi_o.reshape([2] * n), list(range(n))[::-1]).reshape(-1)))
        f_lit = abs(np.vdot(np.transpose(psi_l, list(range(n))[::-1]).reshape(-1), psi_o))
        nrm = np.linalg.norm(psi_l)
        print(f'[ro] DENSE: fidelity {max(f_big, f_lit) / nrm:.6f} (norm {nrm:.6f})', flush=True)
        summary['dense_fidelity'] = float(max(f_big, f_lit) / nrm)
    cands = beam_bitstrings(s, a.beam)
    order = list(s.qubit_at)
    top = [float(np.exp(2 * w)) for w, _ in cands[:8]]
    summary['beam'] = len(cands)
    summary['weights_top8_rel'] = [t / top[0] for t in top] if top else []
    print(f"[ro] beam {len(cands)}: rank1/rank2 weight ratio {(top[0] / top[1] if len(top) > 1 else float('nan')):.2f}", flush=True)
    if a.emit_bits:
        rows = []
        for logw, bits in cands[:64]:
            wire = [''] * n
            for site, ch in enumerate(bits):
                wire[order[site]] = ch
            rows.append(''.join((wire[fr[q]] for q in range(n))))
        Path(a.emit_bits).write_text(json.dumps({'bits': rows, 'weights': [float(np.exp(2 * w)) for w, _ in cands[:64]]}))
    if a.instance:
        man = json.load(open(a.manifest))
        truth_h = set(next((e for e in man['instances'] if e['qasm'] == a.instance))['truth_sha256'].values())
        hit = None
        for rank, (logw, bits) in enumerate(cands, 1):
            wire = [''] * n
            for site, ch in enumerate(bits):
                wire[order[site]] = ch
            logical = ''.join((wire[fr[q]] for q in range(n)))
            forms = {'logical': logical, 'logical_rev': logical[::-1], 'wire': ''.join(wire), 'wire_rev': ''.join(wire)[::-1], 'site': bits, 'site_rev': bits[::-1]}
            for name, f in forms.items():
                if hashlib.sha256(f.encode()).hexdigest() in truth_h:
                    hit = (rank, name)
                    break
            if hit:
                break
        summary['instance'] = a.instance
        summary['truth_rank'] = hit[0] if hit else None
        summary['truth_form'] = hit[1] if hit else None
        summary['solved'] = bool(hit and hit[0] == 1)
        print(f"[ro] GRADE {a.instance}: {('rank %d (%s)' % hit if hit else 'no match in beam')} -> solved {summary['solved']}", flush=True)
    if a.out:
        txt = json.dumps(summary, indent=1, default=float)
        assert not re.search('"[01]{%d,}"' % max(8, n // 2), txt)
        Path(a.out).write_text(txt + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
