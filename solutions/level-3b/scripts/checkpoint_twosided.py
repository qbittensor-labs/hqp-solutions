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
import math
import pickle
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT / 'scripts'))
from peaked.core_mpo import apply_core, portable_to_chain
from peaked.mps import MPS, overlap
from checkpoint_forward_pass import beam_bitstrings
from checkpoint_to_reduced_qasm import layer_gates, load_layers

def chain_gate(m, xp, dt, adjoint: bool=False):
    m = np.asarray(m, dtype=complex)
    if adjoint:
        m = m.conj().T
    return xp.asarray(np.ascontiguousarray(m.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2)), dtype=dt)

def run_gates(s: MPS, gates, xp, dt, *, inverse: bool=False) -> None:
    seq = reversed(gates) if inverse else gates
    for kind, ws, m in seq:
        if kind == 'swap':
            pi = list(range(s.n))
            pi[ws[0]], pi[ws[1]] = (ws[1], ws[0])
            s.permute_logical(pi)
        else:
            s.apply_2q(ws[0], ws[1], chain_gate(m, xp, dt, adjoint=inverse))

def backward_scores(L: MPS, gates_right, cands_wire: list[str], chi: int, xp, dt, cutoff: float=1e-10):
    layout = MPS(L.n, 2, xp=xp, dtype=dt, perm=list(L.pos))
    for kind, ws, _m in gates_right:
        if kind == 'swap':
            pi = list(range(L.n))
            pi[ws[0]], pi[ws[1]] = (ws[1], ws[0])
            layout.permute_logical(pi)
    out = []
    for bits in cands_wire:
        t = MPS(L.n, chi, xp=xp, dtype=dt, perm=list(layout.pos), product_bits=bits, cutoff=cutoff)
        run_gates(t, gates_right, xp, dt, inverse=True)
        t.restore_order(L.qubit_at)
        amp = abs(overlap(L, t)) * math.exp(L.log_norm + t.log_norm)
        out.append((amp, math.log10(amp) if amp > 0 else -math.inf))
    return out

def load_checkpoint(path: str, chi: int, xp, dt, cutoff: float, mpo_gram: bool):
    pay = pickle.load(open(path, 'rb'))
    il, ir = (int(pay['ii_left']), int(pay['ii_right']))
    ll, lr = (load_layers(pay['layers_left']), load_layers(pay['layers_right']))
    fr = [int(v) for v in pay['frame_right']]
    n = len(fr)
    chain = portable_to_chain(pay)
    labels = list(portable_to_chain.last_leg_labels)
    W = [np.asarray(t, dtype=complex) for t in chain]
    fwd_left, fwd_right = ([], [])
    for lay in reversed(ll[il:]):
        fwd_left += layer_gates(lay.inverse())
    for lay in lr[ir:]:
        fwd_right += layer_gates(lay)
    s = MPS(n, chi, xp=xp, dtype=dt, cutoff=cutoff, svd_method='svd')
    run_gates(s, fwd_left, xp, dt)
    FL = [0] * n
    FR = [0] * n
    for i, (up, lo) in enumerate(labels):
        FL[lo] = i
        FR[up] = i
    s.svd_method = 'gram' if mpo_gram else 'svd'
    apply_core(s, W, FL, FR)
    s.svd_method = 'svd'
    info = {'checkpoint': path, 'mpo_bond': max((t.shape[0] for t in W)), 'pending': [sum((1 for k, _, _ in g if k == 'gate')) for g in (fwd_left, fwd_right)], 'bond_after_mpo': s.max_bond()}
    return (s, fwd_right, fr, n, info)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--chi', type=int, default=256)
    ap.add_argument('--cutoff', type=float, default=1e-10)
    ap.add_argument('--bits', default='')
    ap.add_argument('--beam', type=int, default=4096)
    ap.add_argument('--top', type=int, default=16)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--mpo-gram', type=int, default=1)
    ap.add_argument('--out', default='')
    ap.add_argument('--manifest', default='')
    ap.add_argument('--instance', default='')
    ap.add_argument('--truth-hashes', default='')
    a = ap.parse_args()
    if a.device.startswith('cuda'):
        import torch
        from peaked.torch_np import TorchNP
        xp, dt = (TorchNP(a.device), torch.complex128)
    else:
        xp, dt = (np, np.complex128)
    L, gates_right, fr, n, info = load_checkpoint(a.checkpoint, a.chi, xp, dt, a.cutoff, bool(a.mpo_gram))
    print(f"[ts] {Path(a.checkpoint).name}: MPO bond {info['mpo_bond']}, pending {info['pending']}, bond after MPO {info['bond_after_mpo']}", flush=True)
    inv_fr = [0] * n
    for q, w in enumerate(fr):
        inv_fr[w] = q
    if a.bits:
        d = json.load(open(a.bits))
        rows = d['bits'][:a.top]
        weights = d.get('weights', [])[:a.top]
    else:
        import copy
        s = copy.deepcopy(L)
        run_gates(s, gates_right, xp, dt)
        cands = beam_bitstrings(s, a.beam)[:a.top]
        order = list(s.qubit_at)
        rows, weights = ([], [])
        for logw, bits in cands:
            wire = [''] * n
            for site, ch in enumerate(bits):
                wire[order[site]] = ch
            rows.append(''.join((wire[fr[q]] for q in range(n))))
            weights.append(float(np.exp(2 * logw)))
    cands_wire = [''.join((x[inv_fr[w]] for w in range(n))) for x in rows]
    import time
    t0 = time.time()
    sc = backward_scores(L, gates_right, cands_wire, a.chi, xp, dt, a.cutoff)
    dt_s = time.time() - t0
    order_ts = sorted(range(len(rows)), key=lambda i: -sc[i][0])
    top_ts = sc[order_ts[0]][0]
    second = sc[order_ts[1]][0] if len(order_ts) > 1 else 0.0
    res = {'info': info, 'chi': a.chi, 'n_cands': len(rows), 'seconds': dt_s, 'beam_ratio': weights[0] / weights[1] if len(weights) > 1 and weights[1] > 0 else None, 'twosided_ratio': top_ts / second if second > 0 else None, 'twosided_rank_of_beam_rank1': order_ts.index(0) + 1 if rows else None, 'scores_rel': [sc[i][0] / top_ts if top_ts > 0 else None for i in range(len(rows))]}
    truth = None
    if a.instance and a.manifest:
        man = json.load(open(a.manifest))
        truth = set(next((e for e in man['instances'] if e['qasm'] == a.instance))['truth_sha256'].values())
    elif a.truth_hashes:
        t = json.load(open(a.truth_hashes))
        inst = t['instances'][0] if 'instances' in t else t
        truth = set((inst.get('truth_sha256') or {}).values())
    if truth is not None:
        h = lambda s_: hashlib.sha256(s_.encode()).hexdigest()
        hit = [i for i, x in enumerate(rows) if h(x) in truth or h(x[::-1]) in truth]
        res['truth_beam_rank'] = hit[0] + 1 if hit else None
        res['truth_twosided_rank'] = order_ts.index(hit[0]) + 1 if hit else None
    print(f"[ts] {len(rows)} candidates in {dt_s:.0f} s: beam ratio {res['beam_ratio']}, two-sided ratio {res['twosided_ratio']}, beam rank-1 is two-sided rank {res['twosided_rank_of_beam_rank1']}, truth beam/two-sided rank {res.get('truth_beam_rank')}/{res.get('truth_twosided_rank')}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
