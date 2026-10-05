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

import argparse
import hashlib
import io
import os
import pickle
import re
import sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'src'))
from peaked.mps import MPS, overlap
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

def load_layers(blob):
    from qiskit import qpy
    return list(qpy.load(io.BytesIO(blob)))

def layer_ops(layer):
    from qiskit.quantum_info import Operator
    out = []
    for inst in layer.data:
        qs = tuple((layer.find_bit(q).index for q in inst.qubits))
        if len(qs) != 2:
            continue
        m = SWAP if inst.operation.name == 'swap' else np.asarray(Operator(inst.operation).data, dtype=complex)
        out.append((qs, m.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2)))
    return out

def beam_bitstrings(mps, beam):
    xp = mps.xp
    mps.move_center_to(0)
    A = [np.asarray(t) if xp is np else t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(xp.asnumpy(t)) for t in mps.A]
    logw = np.zeros(1)
    bits = ['']
    env = np.ones((1, 1), dtype=complex)
    for t in A:
        V = np.stack([env @ t[:, 0, :], env @ t[:, 1, :]], axis=1)
        nv = np.linalg.norm(V, axis=2)
        ok = (nv > 0).reshape(-1)
        with np.errstate(divide='ignore'):
            sc = (logw[:, None] + np.log(nv)).reshape(-1)
        idx = np.flatnonzero(ok)
        idx = idx[np.argsort(-sc[idx], kind='stable')][:beam]
        par, b = (idx // 2, idx % 2)
        env = V[par, b] / nv[par, b][:, None]
        logw = sc[idx]
        bits = [bits[p] + str(int(q)) for p, q in zip(par, b)]
    return [(float(w), s_) for w, s_ in zip(logw, bits)]

def candidate_digests(private_dir):
    curated, broad = ({}, set())
    p = Path(private_dir)
    if not p.is_dir():
        return (curated, broad)
    for f in sorted(p.iterdir()):
        if not f.is_file() or f.stat().st_size > 50000000:
            continue
        try:
            text = f.read_text(errors='ignore')
        except OSError:
            continue
        hits = re.findall('[01]{40,}', text)
        for i, m in enumerate(hits):
            d = hashlib.sha256(m.encode()).hexdigest()
            broad.add(d)
            if f.name.startswith('s2_cands') and f.suffix == '.txt':
                curated.setdefault(d, f'{f.name}#{i}')
    return (curated, broad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--chi', type=int, default=512)
    ap.add_argument('--beam', type=int, default=4096)
    ap.add_argument('--cutoff', type=float, default=0.0001)
    ap.add_argument('--device', default='auto', choices=('auto', 'cuda', 'cpu'))
    ap.add_argument('--private', default=os.path.expanduser('/tmp/private'))
    ap.add_argument('--max-layers', type=int, default=0)
    a = ap.parse_args()
    with open(a.checkpoint, 'rb') as fh:
        p = pickle.load(fh)
    c = p['counters']
    il, ir = (int(p['ii_left']), int(p['ii_right']))
    ll, lr = (load_layers(p['layers_left']), load_layers(p['layers_right']))
    pend_l, pend_r = (list(reversed(ll[il:])), list(lr[ir:]))
    if a.max_layers:
        pend_l, pend_r = (pend_l[-a.max_layers:], pend_r[:a.max_layers])
    xp, dtype, dev = (np, np.complex128, 'cpu')
    if a.device != 'cpu':
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
            xp, dtype, dev = (cupy, cupy.complex64, 'cuda(cupy)')
        except Exception as exc:
            print(f'[fp] no cupy GPU path ({type(exc).__name__}), falling back to numpy', flush=True)
    from peaked.core_mpo import load_core
    W, _, _ = load_core(a.checkpoint)
    nq = len(W)
    mb = max((int(np.asarray(t).shape[0]) for t in W))
    print(f"[fp] {Path(a.checkpoint).name}: cycles {c['unswap_cycles']}, layers {c['layers_absorbed']}, work {c['work_ops_absorbed_total']}/{c['work_ops_total']}, MPO bond {mb}", flush=True)
    print(f'[fp] pending: {len(pend_l)} left + {len(pend_r)} right layers, chi {a.chi}, device {dev}', flush=True)
    s = MPS(nq, a.chi, xp=xp, dtype=dtype, cutoff=a.cutoff, product_bits='0' * nq, svd_method='gram')
    for tag, seq in (('pre', pend_l),):
        for i, lay in enumerate(seq):
            for qs, g in layer_ops(lay):
                s.apply_2q(qs[0], qs[1], xp.asarray(g, dtype=dtype) if xp is not np else g)
            if i % 50 == 0:
                print(f'[fp] {tag} layer {i}/{len(seq)}  bond {s.max_bond()}', flush=True)
    s.apply_mpo(W)
    print(f'[fp] MPO applied, bond {s.max_bond()}', flush=True)
    for i, lay in enumerate(pend_r):
        for qs, g in layer_ops(lay):
            s.apply_2q(qs[0], qs[1], xp.asarray(g, dtype=dtype) if xp is not np else g)
        if i % 50 == 0:
            print(f'[fp] post layer {i}/{len(pend_r)}  bond {s.max_bond()}', flush=True)
    order = list(s.qubit_at)
    fperm = None
    try:
        fm = load_layers(p['final_meas'])
        if fm:
            fperm = [fm[-1].find_bit(g.qubits[0]).index for g in fm[-1].data]
    except Exception as exc:
        print(f'[fp] final_meas not decodable ({type(exc).__name__})', flush=True)
    print(f"[fp] final_meas permutation: {('present, %d entries' % len(fperm) if fperm else 'none')}", flush=True)
    print(f'[fp] beam search start (beam {a.beam}, bond {s.max_bond()})', flush=True)
    try:
        cands = beam_bitstrings(s, a.beam)
    except BaseException as exc:
        import traceback
        print(f'[fp] beam FAILED: {type(exc).__name__}: {exc}', flush=True)
        traceback.print_exc()
        return 3
    print(f'[fp] beam done: {len(cands)} candidates', flush=True)
    try:
        curated, broad = candidate_digests(a.private)
    except BaseException as exc:
        import traceback
        print(f'[fp] digest load FAILED: {type(exc).__name__}: {exc}', flush=True)
        traceback.print_exc()
        return 3
    print(f'[fp] beam {len(cands)}; {len(curated)} curated candidate digests, {len(broad)} seen-before digests', flush=True)

    def views(bits):
        base = [None] * nq
        for site, ch in enumerate(bits):
            base[order[site]] = ch
        v = ''.join((x or '0' for x in base))
        out = {'qubit_at': v, 'site': bits}
        if fperm and len(fperm) == nq:
            pv = [None] * nq
            for i, ch in enumerate(v):
                pv[fperm[i]] = ch
            out['final_meas'] = ''.join((x or '0' for x in pv))
            iv = [None] * nq
            for i, ch in enumerate(v):
                iv[i] = v[fperm[i]]
            out['final_meas_inv'] = ''.join(iv)
        for k in list(out):
            out[k + '_rev'] = out[k][::-1]
        return out
    hit = None
    for rank, (logw, bits) in enumerate(cands, 1):
        for name, form in views(bits).items():
            d = hashlib.sha256(form.encode()).hexdigest()
            if d in curated:
                hit = (rank, logw, name, curated[d], True)
                break
            if d in broad:
                hit = (rank, logw, name, 'seen-before', False)
                break
        if hit:
            break
    if hit:
        kind = 'CURATED CANDIDATE' if hit[4] else 'previously-seen string'
        print(f'[fp] *** MATCH ({kind}) *** rank {hit[0]} of {len(cands)}  log-weight {hit[1]:.4f}  readout={hit[2]}  source={hit[3]}', flush=True)
    else:
        top = cands[0][0] if cands else float('nan')
        print(f'[fp] no match in beam (top log-weight {top:.4f})', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
