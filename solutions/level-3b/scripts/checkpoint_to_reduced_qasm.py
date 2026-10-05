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
import io
import json
import pickle
import sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
from peaked.core_mpo import portable_to_chain

def spectra(chain):
    A = [np.asarray(a).reshape(a.shape[0], a.shape[1] * a.shape[2], a.shape[3]) for a in chain]
    L = len(A)
    for i in range(L - 1):
        dl, dp, dr = A[i].shape
        Q, R = np.linalg.qr(A[i].reshape(dl * dp, dr))
        A[i] = Q.reshape(dl, dp, -1)
        A[i + 1] = np.tensordot(R, A[i + 1], axes=(1, 0))
    out = [None] * (L - 1)
    for i in range(L - 1, 0, -1):
        dl, dp, dr = A[i].shape
        U, s, Vh = np.linalg.svd(A[i].reshape(dl, dp * dr), full_matrices=False)
        out[i - 1] = s
        A[i] = Vh.reshape(-1, dp, dr)
        A[i - 1] = np.tensordot(A[i - 1], U * s, axes=(2, 0))
    return out

def mpo_overlap(X, Y):
    E = np.ones((1, 1), dtype=complex)
    for a, b in zip(X, Y):
        t = np.tensordot(E, np.asarray(a).conj(), axes=(0, 0))
        E = np.tensordot(t, np.asarray(b), axes=([0, 1, 2], [0, 1, 2]))
    return complex(E.reshape(-1)[0])

def regions(strong, n):
    out, cur = ([], [0])
    for c in range(n - 1):
        if c in strong:
            cur.append(c + 1)
        else:
            out.append(cur)
            cur = [c + 1]
    out.append(cur)
    return out

def region_gate(chain, sites):
    T = np.asarray(chain[sites[0]])
    for s in sites[1:]:
        T = np.tensordot(T, np.asarray(chain[s]), axes=(T.ndim - 1, 0))
    k = len(sites)
    assert T.shape[0] == 1 and T.shape[-1] == 1, 'region does not have unit outer bonds'
    T = T.reshape([2, 2] * k)
    return np.transpose(T, list(range(0, 2 * k, 2)) + list(range(1, 2 * k, 2))).reshape(2 ** k, 2 ** k)

def gate_to_chain(G, k):
    T = G.reshape([2] * k + [2] * k)
    order = []
    for i in range(k):
        order += [i, k + i]
    rest = np.transpose(T, order).reshape([1] + [2, 2] * k + [1])
    parts, left = ([], 1)
    for i in range(k - 1):
        m = rest.reshape(left * 4, -1)
        U, s, Vh = np.linalg.svd(m, full_matrices=False)
        keep = int((s > 1e-13).sum()) or 1
        U, s, Vh = (U[:, :keep], s[:keep], Vh[:keep])
        parts.append(U.reshape(left, 2, 2, keep))
        left = keep
        rest = (np.diag(s) @ Vh).reshape(keep, *[2, 2] * (k - i - 1), 1)
    parts.append(rest.reshape(left, 2, 2, 1))
    return parts
SPLIT_USED = [0]
REASSEMBLY_TOL = 0.01

def truncate_weak_cuts(chain, strong):
    shapes = [np.asarray(a).shape for a in chain]
    A = [np.asarray(a).reshape(sh[0], sh[1] * sh[2], sh[3]) for a, sh in zip(chain, shapes)]
    L = len(A)
    for i in range(L - 1):
        dl, dp, dr = A[i].shape
        Q, R = np.linalg.qr(A[i].reshape(dl * dp, dr))
        A[i] = Q.reshape(dl, dp, -1)
        A[i + 1] = np.tensordot(R, A[i + 1], axes=(1, 0))
    for i in range(L - 1, 0, -1):
        dl, dp, dr = A[i].shape
        U, sv, Vh = np.linalg.svd(A[i].reshape(dl, dp * dr), full_matrices=False)
        keep = 1 if i - 1 not in strong else max(1, int((sv > 1e-14 * sv[0]).sum()))
        A[i] = Vh[:keep].reshape(keep, dp, dr)
        A[i - 1] = np.tensordot(A[i - 1], U[:, :keep] * sv[:keep], axes=(2, 0))
    return [a.reshape(a.shape[0], sh[1], sh[2], a.shape[2]) for a, sh in zip(A, shapes)]

def factorise(chain, labels, strong_thr=1e-09, max_factor=8):
    n = len(chain)
    sp = spectra(chain)
    strong = set()
    for c, s in enumerate(sp):
        t = float((s ** 2).sum())
        if t > 0 and len(s) > 1 and (float((s[1:] ** 2).sum()) / t > strong_thr):
            strong.add(c)
    regs = regions(strong, n)
    chain = truncate_weak_cuts(chain, strong)
    from factor_split import split_factor, rebuild
    SPLIT_THR = 1e-06
    SPLIT_MAX = 12
    split_regions = {}
    SPLIT_USED[0] = 0
    for r in regs:
        if len(r) >= 3 and len(r) <= SPLIT_MAX:
            pcs = split_factor(region_gate(chain, r), len(r), SPLIT_THR)
            if max((len(pc[0]) for pc in pcs)) <= max_factor and len(pcs) > 1:
                split_regions[tuple(r)] = pcs
                SPLIT_USED[0] += 1
                print(f'[red] split a {len(r)}-site region into pieces {sorted((len(pc[0]) for pc in pcs), reverse=True)}', flush=True)
    big = [r for r in regs if len(r) > max_factor and tuple(r) not in split_regions]
    if big:
        raise SystemExit(f'largest irreducible factor has {max((len(r) for r in big))} sites (> {max_factor}); this object is not extractable as gates')
    factors, rebuilt = ([], [])
    for r in regs:
        if tuple(r) in split_regions:
            pcs = split_regions[tuple(r)]
            for outs, ins, Up, sc in pcs:
                factors.append({'sites': [int(r[i]) for i in outs], 'up': [int(labels[r[i]][0]) for i in outs], 'lo': [int(labels[r[i]][1]) for i in ins], 'unitary': Up, 'scale': sc, 'nonunitarity': 0.0, 'sv_min_over_max': 1.0})
            rebuilt += gate_to_chain(rebuild(pcs, len(r)), len(r))
            continue
        G = region_gate(chain, r)
        U, s, Vh = np.linalg.svd(G)
        Gu = U @ Vh
        scale = float(s.mean())
        dev = float(np.linalg.norm(G / scale - Gu) / np.sqrt(G.shape[0]))
        factors.append({'sites': list(map(int, r)), 'up': [int(labels[s_][0]) for s_ in r], 'lo': [int(labels[s_][1]) for s_ in r], 'unitary': Gu, 'scale': scale, 'nonunitarity': dev, 'sv_min_over_max': float(s.min() / s.max())})
        rebuilt += gate_to_chain(Gu * scale, len(r))
    n2 = mpo_overlap(chain, chain).real
    num = n2 + mpo_overlap(rebuilt, rebuilt).real - 2 * mpo_overlap(chain, rebuilt).real
    rel = float(max(num, 0.0) ** 0.5 / n2 ** 0.5)
    return (factors, rel, sp)

def load_layers(blob):
    from qiskit import qpy
    return list(qpy.load(io.BytesIO(blob)))

def layer_gates(circ):
    from qiskit.quantum_info import Operator
    out = []
    for inst in circ.data:
        name = inst.operation.name
        if name in ('barrier', 'measure'):
            continue
        ws = tuple((circ.find_bit(q).index for q in inst.qubits))
        if name == 'swap':
            out.append(('swap', ws, None))
        else:
            out.append(('gate', ws, np.asarray(Operator(inst.operation).data, dtype=complex)))
    return out

def original_blocks(qasm_path, n):
    from enigma_peaked.engine.generator import prepare_circuit
    from enigma_peaked.frames.schedule import FrameSchedule
    from qiskit.quantum_info import Operator
    sha = hashlib.sha256(Path(qasm_path).read_bytes()).hexdigest()
    circ = prepare_circuit(Path(qasm_path), FrameSchedule.identity(n, sha))
    blocks = []
    for inst in circ.data:
        if inst.operation.name in ('barrier', 'measure'):
            continue
        qs = tuple((circ.find_bit(q).index for q in inst.qubits))
        blocks.append((len(blocks), qs, np.asarray(Operator(inst.operation).data, dtype=complex)))
    return (blocks, sha)

def overlap_phase_invariant(A, B):
    return float(abs(np.vdot(A, B)) / (np.linalg.norm(A) * np.linalg.norm(B)))

def reorder_2q(U, qs_from, qs_to):
    if tuple(qs_from) == tuple(qs_to):
        return U
    assert set(qs_from) == set(qs_to)
    return U.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)

def match_block(qs, U, blocks_by_pair, used, tol):
    key = tuple(sorted(qs))
    best = (0.0, None)
    for idx, bqs, BU in blocks_by_pair.get(key, []):
        if idx in used:
            continue
        ov = overlap_phase_invariant(reorder_2q(U, qs, bqs), BU)
        if ov > best[0]:
            best = (ov, idx)
    if best[1] is not None and best[0] >= 1 - tol:
        return (best[1], best[0])
    return (None, best[0])

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('--qasm', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--window')
    ap.add_argument('--locate', default='')
    ap.add_argument('--match-tol', type=float, default=1e-05)
    ap.add_argument('--max-factor', type=int, default=8)
    ap.add_argument('--strong-thr', type=float, default=1e-09)
    ap.add_argument('--left-mode', choices=('inverse', 'asis'), default='inverse')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--no-transpile', action='store_true')
    ap.add_argument('--dense-check', action='store_true')
    ap.add_argument('--no-write', action='store_true')
    ap.add_argument('--absorbed-only', action='store_true')
    a = ap.parse_args()
    with open(a.checkpoint, 'rb') as fh:
        pay = pickle.load(fh)
    c = pay.get('counters') or {}
    il, ir = (int(pay['ii_left']), int(pay['ii_right']))
    ll, lr = (load_layers(pay['layers_left']), load_layers(pay['layers_right']))
    fl, fr = ([int(v) for v in pay['frame_left']], [int(v) for v in pay['frame_right']])
    n = len(fr)
    sha_ck = pay.get('input_qasm_sha256')
    print(f"[red] {Path(a.checkpoint).name}: layers {c.get('layers_absorbed')} cycles {c.get('unswap_cycles')} work {c.get('work_ops_absorbed_total')}/{c.get('work_ops_total')} (L {c.get('work_ops_absorbed_left')} R {c.get('work_ops_absorbed_right')}); pending layers {len(ll) - il} left, {len(lr) - ir} right", flush=True)
    blocks, sha = original_blocks(a.qasm, n)
    if sha_ck and sha != sha_ck:
        print(f'[red] WARNING: checkpoint sha {sha_ck[:12]} != qasm sha {sha[:12]}', flush=True)
    N = len(blocks)
    if a.window:
        WA, WB = (int(x) for x in a.window.split(','))
    else:
        w = (pay.get('engine_config') or {}).get('absorb_window')
        WA, WB = (int(w[0]), int(w[1])) if w else (0, N)
    print(f'[red] original: {N} blocks, window [{WA},{WB})', flush=True)
    chain = portable_to_chain(pay)
    labels = list(portable_to_chain.last_leg_labels)
    if a.absorbed_only:
        factors = [{'up': [u], 'lo': [l], 'sites': [i], 'nonunitarity': 0.0, 'unitary': None} for i, (u, l) in enumerate(labels)]
        reassembly, sp = (0.0, [])
    else:
        factors, reassembly, sp = factorise(chain, labels, strong_thr=a.strong_thr, max_factor=a.max_factor)
        from factor_synth import synth_brickwork, rebuild_brickwork, _err
        synth_err = 0.0
        for f in factors:
            k = len(f['sites'])
            if k >= 3:
                g = synth_brickwork(f['unitary'], k, tol=0.0003, max_depth=2 * k, restarts=2, sweeps=300)
                if g is not None:
                    f['synth'] = g
                    synth_err = max(synth_err, _err(f['unitary'], rebuild_brickwork(g, k)))
        n_syn = sum((1 for f in factors if 'synth' in f))
        print(f"[red] brickwork synthesis: {n_syn} of {sum((1 for f in factors if len(f['sites']) >= 3))} factors of >= 3 sites ({[len(f['synth']) for f in factors if 'synth' in f]} 2q gates), worst error {synth_err:.1e}", flush=True)
    up_all = sorted((u for f in factors for u in f['up']))
    lo_all = sorted((l for f in factors for l in f['lo']))
    assert up_all == list(range(n)) and lo_all == list(range(n)), 'leg labels are not a bijection'
    multi = [f for f in factors if len(f['sites']) > 1]
    worst_nonu = max((f['nonunitarity'] for f in factors))
    bits = 0.0
    for s in sp:
        p = s ** 2
        t = float(p.sum())
        if t > 0:
            p = p / t
            p = p[p > 1e-15]
            bits += float(-(p * np.log2(p)).sum())
    print(f"[red] object: {len(factors)} factors ({len(multi)} multi-site: {[len(f['sites']) for f in multi]}), reassembly {reassembly:.3e}, worst factor non-unitarity {worst_nonu:.3e}, {bits:.1f} bits in this frame", flush=True)
    fwd = []
    left_layers = ll[il:]
    if a.left_mode == 'inverse':
        for lay in reversed(left_layers):
            fwd += [(k, w, m, 'L') for k, w, m in layer_gates(lay.inverse())]
    else:
        for lay in left_layers:
            fwd += [(k, w, m, 'L') for k, w, m in layer_gates(lay)]
    fwd.append(('object', None, None, 'O'))
    for lay in lr[ir:]:
        fwd += [(k, w, m, 'R') for k, w, m in layer_gates(lay)]
    M = [0] * n
    for q, w in enumerate(fr):
        M[w] = q
    emitted = []
    for kind, ws, m, side in reversed(fwd):
        if kind == 'swap':
            x, y = ws
            M[x], M[y] = (M[y], M[x])
        elif kind == 'gate':
            emitted.append((tuple((M[w] for w in ws)), m, side))
        else:
            Mb = list(M)
            for f in factors:
                for u, l in zip(f['up'], f['lo']):
                    Mb[l] = M[u]
            for f in factors:
                qs = [Mb[l] for l in f['lo']]
                if 'synth' in f:
                    for i, j, g in reversed(f['synth']):
                        emitted.append(((qs[j], qs[i]), g, 'O'))
                    continue
                emitted.append((tuple(reversed(qs)), f['unitary'], 'O'))
            M = Mb
    M_left = list(M)
    inv_fl = [0] * n
    for q, w in enumerate(fl):
        inv_fl[w] = q
    Pi = [0] * n
    for w in range(n):
        Pi[inv_fl[w]] = M_left[w]
    pi_identity = Pi == list(range(n))
    print(f"[red] net logical permutation across the absorbed set: {('identity' if pi_identity else f'{sum((1 for q in range(n) if Pi[q] != q))} qubits moved')}", flush=True)
    gates = list(reversed(emitted))
    peeled = [(int(i), sd) for i, sd in pay.get('peeled') or []]
    if peeled:
        bmap = {idx: (qs, U) for idx, qs, U in blocks}
        io = next((k for k, g in enumerate(gates) if g[2] == 'O'))
        left_p = [(tuple((Pi[q] for q in bmap[i][0])), bmap[i][1], 'P') for i, sd in sorted(peeled) if sd == 'L']
        right_p = [(tuple(bmap[i][0]), bmap[i][1], 'P') for i, sd in sorted(peeled) if sd == 'R']
        io_end = max((k for k, g in enumerate(gates) if g[2] == 'O')) + 1
        gates = gates[:io] + left_p + gates[io:io_end] + right_p + gates[io_end:]
        print(f'[red] peeled blocks re-emitted next to the object: {len(left_p)} left, {len(right_p)} right', flush=True)
    by_pair = {}
    for idx, qs, U in blocks:
        by_pair.setdefault(tuple(sorted(qs)), []).append((idx, qs, U))
    inv_Pi = [0] * n
    for q, r in enumerate(Pi):
        inv_Pi[r] = q
    used = {}
    unmatched, worst = ([], 1.0)
    for gi, (qs, U, side) in enumerate(gates):
        if side == 'O':
            continue
        if side == 'P':
            continue
        if len(qs) != 2:
            unmatched.append((gi, side, qs, 'not-2q'))
            continue
        qs_orig = tuple((inv_Pi[q] for q in qs)) if side == 'L' else qs
        idx, ov = match_block(qs_orig, U, by_pair, used, a.match_tol)
        if idx is None:
            unmatched.append((gi, side, qs_orig, round(ov, 6)))
        else:
            used[idx] = gi
            worst = min(worst, ov)
    in_window = [idx for idx in used if WA <= idx < WB]
    outside = [idx for idx in used if not WA <= idx < WB]
    absorbed = sorted(set(range(WA, WB)) - set(used) - {i for i, _ in peeled})
    n_pending = sum((1 for g in gates if g[2] not in ('O', 'P')))
    print(f'[red] pending blocks: {n_pending}; matched {len(used)} (worst overlap {worst:.8f}); unmatched {len(unmatched)}' + (f' e.g. {unmatched[:5]}' if unmatched else ''), flush=True)
    print(f"[red] absorbed set: {len(absorbed)} blocks in [{WA},{WB}) not pending (engine says {c.get('work_ops_absorbed_total')}); range [{(absorbed[0] if absorbed else None)},{(absorbed[-1] + 1 if absorbed else None)}); pending matched outside window: {len(outside)}", flush=True)
    if a.absorbed_only:
        Path(str(a.out) + '.absorbed.json').write_text(json.dumps({'absorbed': [int(x) for x in absorbed], 'window': [WA, WB]}))
        return 0
    try:
        Path(str(a.out) + '.absorbed.json').write_text(json.dumps({'absorbed': [int(x) for x in absorbed], 'window': [WA, WB]}))
    except OSError:
        pass
    checks = {'reassembly_rel_err': reassembly, 'reassembly_ok': reassembly < REASSEMBLY_TOL, 'worst_factor_nonunitarity': worst_nonu, 'pending_all_matched': not unmatched, 'n_unmatched': len(unmatched), 'pending_matched': len(used), 'pending_emitted': n_pending, 'absorbed_count_matches_engine': c.get('work_ops_absorbed_total') is None or len(absorbed) + len(peeled) == int(c['work_ops_absorbed_total']), 'pending_outside_window': len(outside), 'net_permutation_identity': pi_identity}
    ok = checks['reassembly_ok'] and checks['pending_all_matched'] and checks['absorbed_count_matches_engine'] and (checks['pending_outside_window'] == 0)
    from qiskit import QuantumCircuit, qasm2, transpile
    from qiskit.circuit.library import UnitaryGate
    red = QuantumCircuit(n)
    pend_dev = [0.0]

    def ug(U):
        Uu, s_, Vh = np.linalg.svd(np.asarray(U, dtype=complex))
        W = Uu @ Vh
        pend_dev[0] = max(pend_dev[0], float(np.linalg.norm(U / s_.mean() - W) / np.sqrt(U.shape[0])))
        return UnitaryGate(W, check_input=False)
    pre = [b for b in blocks if b[0] < WA]
    for idx, qs, U in pre:
        red.append(ug(U), [Pi[q] for q in qs])
    obj_pos = None
    for gi, (qs, U, side) in enumerate(gates):
        if side == 'O' and obj_pos is None:
            obj_pos = len(red.data)
        red.append(ug(U), list(qs))
    post = [b for b in blocks if b[0] >= WB]
    for idx, qs, U in post:
        red.append(ug(U), list(qs))
    checks['worst_gate_reunitarisation'] = pend_dev[0]
    n_obj_gates = sum((1 for g in gates if g[2] == 'O'))
    print(f'[red] reduced logical circuit: {len(pre)} pre + {n_pending} pending + {n_obj_gates} object factors + {len(post)} post = {len(red.data)} gates (original {N} blocks)', flush=True)
    dense = None
    if a.dense_check:
        from qiskit.quantum_info import Operator
        oc = QuantumCircuit.from_qasm_file(a.qasm)
        oc.remove_final_measurements(inplace=True)
        Uo, Ur = (Operator(oc).data, Operator(red).data)
        D = Uo.conj().T @ Ur
        col_max = np.abs(D).max(axis=0)
        bij = len(set(np.abs(D).argmax(axis=0).tolist())) == D.shape[0]
        perm_dev = float(np.linalg.norm(np.abs(D) - (np.abs(D) > 0.5)) / np.sqrt(D.shape[0]))
        f0 = float(abs(np.vdot(Ur[:, 0], Uo[:, 0])))
        dense = {'min_col_max': float(col_max.min()), 'bijective': bij, 'perm_dev': perm_dev, 'vac_fidelity': f0}
        print(f'[red] DENSE: min column max {col_max.min():.6f} (1 = permutation x phases), bijective {bij}, deviation {perm_dev:.3e}, |<0|Ur^+ Uo|0>| = {f0:.6f}', flush=True)
    if a.no_write:
        out = Path(a.out)
        out.with_suffix(out.suffix + '.json').write_text(json.dumps({'checks': checks, 'dense': dense, 'ok': ok}, indent=1))
        return 0 if ok else 1
    red_t = QuantumCircuit(n)
    offsets = []
    for inst in red.data:
        offsets.append(len(red_t.data))
        qs_ = [red.find_bit(q).index for q in inst.qubits]
        if a.no_transpile:
            red_t.append(inst.operation, qs_)
            continue
        sub = QuantumCircuit(n)
        sub.append(inst.operation, qs_)
        sub = transpile(sub, basis_gates=['u', 'cz'], optimization_level=0)
        red_t.compose(sub, inplace=True)
    out = Path(a.out)
    side = {'checkpoint': str(a.checkpoint), 'qasm': str(a.qasm), 'qasm_sha256': sha, 'counters': c, 'window': [WA, WB], 'checks': checks, 'ok': ok, 'left_mode': a.left_mode, 'absorbed_blocks': absorbed, 'net_permutation_left_to_right': Pi, 'factors': [{'sites': f['sites'], 'up': f['up'], 'lo': f['lo'], 'nonunitarity': f['nonunitarity'], 'sv_min_over_max': f['sv_min_over_max']} for f in factors], 'object_bits_in_frame': bits, 'object_first_gate_index': obj_pos, 'gate_count': len(red_t.data), 'unmatched': unmatched[:50], 'dense': dense}
    if not ok and (not a.force):
        print(f'[red] CHECKS FAILED: {json.dumps({k: v for k, v in checks.items() if v is False or (isinstance(v, int) and v)})}; nothing written (use --force for diagnostics)', flush=True)
        out.with_suffix(out.suffix + '.json').write_text(json.dumps(side, indent=1))
        return 1
    qasm_text = qasm2.dumps(red_t)
    out.write_text(qasm_text)
    side['reduced_sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
    rblocks, _ = original_blocks(str(out), n)
    side['reduced_blocks'] = len(rblocks)
    rby = {}
    for idx, qs, U in rblocks:
        rby.setdefault(tuple(sorted(qs)), []).append((idx, qs, U))
    located = {}
    gate_index_of_block = {}
    for k, (idx, qs, U) in enumerate(pre):
        gate_index_of_block[idx] = k
    for idx, gi in used.items():
        gate_index_of_block[idx] = len(pre) + gi
    for k, (idx, qs, U) in enumerate(post):
        gate_index_of_block[idx] = len(pre) + len(gates) + k
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks

    def blocks_before(raw_cut):
        sub = QuantumCircuit(n)
        for inst in red_t.data[:raw_cut]:
            sub.append(inst.operation, [red_t.find_bit(q).index for q in inst.qubits])
        cc = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(sub)
        return sum((1 for inst in cc.data if len(inst.qubits) == 2))
    for tok in [t for t in a.locate.split(',') if t.strip()]:
        j = int(tok)
        if j in absorbed:
            located[j] = 'absorbed'
            continue
        gi = gate_index_of_block[j]
        raw = offsets[gi]
        located[j] = {'gate_index': gi, 'raw_offset': raw, 'reduced_block': blocks_before(raw)}
    obj_raw = offsets[obj_pos] if obj_pos is not None else None
    side['object_raw_offset'] = obj_raw
    side['object_reduced_block'] = blocks_before(obj_raw) if obj_raw is not None else None
    side['located'] = located
    out.with_suffix(out.suffix + '.json').write_text(json.dumps(side, indent=1))
    print(f"[red] wrote {out} ({len(red_t.data)} gates, {len(rblocks)} consolidated blocks, sha {side['reduced_sha256'][:12]}) + sidecar; located {located}", flush=True)
    return 0 if ok else 2
if __name__ == '__main__':
    raise SystemExit(main())
