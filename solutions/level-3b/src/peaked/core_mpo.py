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
import pickle
from pathlib import Path
import numpy as np

def _payload(obj):
    return obj['mpo'] if isinstance(obj, dict) and 'mpo' in obj else obj

def portable_to_chain(payload) -> list[np.ndarray]:
    p = _payload(payload)
    L, uid, lid = (int(p['L']), p['upper_ind_id'], p['lower_ind_id'])
    inds = [list(t['inds']) for t in p['tensors']]
    arrays = [np.asarray(a) for a in p['arrays']]
    if len(inds) != L or len(arrays) != L:
        raise ValueError(f'payload has {len(arrays)} tensors for L={L}')
    out = []
    up_pre, lo_pre = (uid.split('{}')[0], lid.split('{}')[0])
    leg_labels = []
    for i in range(L):
        u, l = (uid.format(i), lid.format(i))
        if u not in inds[i] or l not in inds[i]:
            us = [x for x in inds[i] if x.startswith(up_pre) and x[len(up_pre):].isdigit()]
            ls = [x for x in inds[i] if x.startswith(lo_pre) and x[len(lo_pre):].isdigit()]
            if len(us) != 1 or len(ls) != 1:
                raise ValueError(f'site {i} is missing its physical legs {u}/{l}')
            u, l = (us[0], ls[0])
        leg_labels.append((int(u[len(up_pre):]), int(l[len(lo_pre):])))
        left = set(inds[i]) & set(inds[i - 1]) if i > 0 else set()
        right = set(inds[i]) & set(inds[i + 1]) if i + 1 < L else set()
        left, right = (sorted(left - {u, l}), sorted(right - {u, l}))
        if len(left) > 1 or len(right) > 1:
            raise ValueError(f'site {i} shares more than one bond with a neighbour')
        order, shape = ([], [])
        for name in (left[0] if left else None, u, l, right[0] if right else None):
            if name is None:
                order.append(None)
            else:
                order.append(inds[i].index(name))
        a = arrays[i]
        perm = [ax for ax in order if ax is not None]
        a = np.transpose(a, perm)
        for slot, ax in enumerate(order):
            if ax is None:
                a = np.expand_dims(a, slot)
        out.append(a)
    portable_to_chain.last_leg_labels = leg_labels
    return out

def pinned_frames(frame_left, frame_right, leg_labels, centre_site_of):
    n = len(frame_left)
    pos_k = {lab: s for s, (lab, _) in enumerate(leg_labels)}
    pos_b = {lab: s for s, (_, lab) in enumerate(leg_labels)}
    fl = [pos_b[frame_left[centre_site_of[q]]] for q in range(n)]
    fr = [pos_k[frame_right[centre_site_of[q]]] for q in range(n)]
    return (fl, fr)

def load_core(path: str | Path):
    with open(path, 'rb') as fh:
        payload = pickle.load(fh)
    frames = (list(payload.get('frame_left') or []), list(payload.get('frame_right') or []))
    if 'mpo_chain' in payload:
        return ([np.asarray(w) for w in payload['mpo_chain']], frames[0], frames[1])
    chain = portable_to_chain(payload)
    pin = Path(str(path) + '.pin.json')
    if pin.exists():
        import json
        E = [int(v) for v in json.loads(pin.read_text())['pin_centre_site_of']]
        fl, fr = pinned_frames([int(v) for v in frames[0]], [int(v) for v in frames[1]], portable_to_chain.last_leg_labels, E)
        return (chain, fl, fr)
    labels = portable_to_chain.last_leg_labels
    if any((lab != (i, i) for i, lab in enumerate(labels))):
        raise ValueError('core has permuted leg labels but no <path>.pin.json with its centre mapping')
    return (chain, frames[0], frames[1])

def adjoint_chain(W):
    return [np.conj(np.transpose(np.asarray(w), (0, 2, 1, 3))) for w in W]

def apply_core(mps, W, frame_left, frame_right, *, compress: bool=True) -> int:
    n = mps.n
    if len(frame_left) != n or len(frame_right) != n:
        raise ValueError('frames must cover every qubit')
    target = [0] * n
    for q, site in enumerate(frame_left):
        target[site] = q
    before = mps.telemetry.swaps
    mps.restore_order(target)
    cost = mps.telemetry.swaps - before
    mps.apply_mpo_zipup(W) if compress else mps.apply_mpo(W, compress=False)
    for q, site in enumerate(frame_right):
        mps.pos[q] = site
        mps.qubit_at[site] = q
    return cost

def chain_to_dense(W) -> np.ndarray:
    acc = np.asarray(W[0])
    for w in W[1:]:
        acc = np.tensordot(acc, np.asarray(w), axes=(acc.ndim - 1, 0))
    acc = acc[0]
    acc = acc[..., 0] if acc.shape[-1] == 1 else acc
    L = len(W)
    perm = [2 * i for i in range(L)] + [2 * i + 1 for i in range(L)]
    return np.transpose(acc, perm).reshape(2 ** L, 2 ** L)

def _svd(xp, M):
    return xp.svd(M, full_matrices=False) if hasattr(xp, 'svd') else xp.linalg.svd(M, full_matrices=False)

def truncate_chain(W, maxbond, *, xp=np, cutoff: float=0.0, log=None):
    n = len(W)
    A = [xp.asarray(w) for w in W]
    for s in range(n - 1, 0, -1):
        Wl, o, i, Wr = A[s].shape
        M = A[s].reshape(Wl, o * i * Wr)
        Q, R = xp.linalg.qr(M.conj().T)
        L, Qm = (R.conj().T, Q.conj().T)
        A[s] = Qm.reshape(Qm.shape[0], o, i, Wr)
        A[s - 1] = xp.tensordot(A[s - 1], L, axes=(3, 0))
    out, carry = ([], None)
    for s in range(n):
        w = A[s] if carry is None else xp.tensordot(carry, A[s], axes=(1, 0))
        Wl, o, i, Wr = w.shape
        U, S, Vh = _svd(xp, w.reshape(Wl * o * i, Wr))
        keep = int(min(maxbond, S.shape[0]))
        if cutoff and S.shape[0]:
            keep = int(min(keep, max(1, int((S > cutoff * float(S[0])).sum()))))
        U, S, Vh = (U[:, :keep], S[:keep], Vh[:keep])
        out.append(U.reshape(Wl, o, i, keep))
        carry = S[:, None] * Vh
        if log and (s % 8 == 0 or s == n - 1):
            log({'truncate_site': s, 'bond': keep})
    out[-1] = xp.tensordot(out[-1], carry, axes=(3, 0))
    return [np.asarray(w) for w in out]

def compose_chain_zipup(B, A, chi, *, xp=np, cutoff: float=0.0, log=None):
    n = len(A)
    if len(B) != n:
        raise ValueError('operators must have the same length')
    T = xp.ones((1, 1, 1), dtype=A[0].dtype)
    out = []
    for s in range(n):
        a, b = (xp.asarray(A[s]), xp.asarray(B[s]))
        theta = xp.einsum('kab,amir,bomn->koirn', T, a, b)
        k, o, i, r, nn = theta.shape
        M = theta.reshape(k * o * i, r * nn)
        U, S, Vh = _svd(xp, M)
        keep = int(min(chi, S.shape[0]))
        if cutoff and S.shape[0]:
            keep = int(min(keep, max(1, int((S > cutoff * S[0]).sum()))))
        U, S, Vh = (U[:, :keep], S[:keep], Vh[:keep])
        out.append(U.reshape(k, o, i, keep))
        T = (S[:, None] * Vh).reshape(keep, r, nn)
        if log and (s % 8 == 0 or s == n - 1):
            log({'compose_site': s, 'bond': keep})
    out[-1] = out[-1] * T.reshape(-1)[0]
    return out

def mpo_swap_adjacent(W, p, chi, *, xp=np, cutoff: float=0.0):
    a, b = (xp.asarray(W[p]), xp.asarray(W[p + 1]))
    theta = xp.einsum('aoib,bpjc->apjoic', a, b)
    Wl, o2, i2, o1, i1, Wr = theta.shape
    U, S, Vh = _svd(xp, theta.reshape(Wl * o2 * i2, o1 * i1 * Wr))
    keep = int(min(chi, S.shape[0]))
    if cutoff and S.shape[0]:
        kc = int((S > cutoff * S[0]).sum())
        keep = max(1, min(keep, kc))
    W[p] = U[:, :keep].reshape(Wl, o2, i2, keep)
    W[p + 1] = (S[:keep, None] * Vh[:keep]).reshape(keep, o1, i1, Wr)

def mpo_permute_to_input_frame(W, frame_left, frame_right, target_left, chi, *, xp=np, log=None, cutoff: float=0.0):
    W = [xp.asarray(w) for w in W]
    n = len(W)
    in_at = [0] * n
    out_at = [0] * n
    for q in range(n):
        in_at[frame_left[q]] = q
        out_at[frame_right[q]] = q
    target_at = [0] * n
    for q in range(n):
        target_at[target_left[q]] = q
    swaps = 0
    for p_target in range(n):
        q = target_at[p_target]
        p = in_at.index(q)
        while p > p_target:
            mpo_swap_adjacent(W, p - 1, chi, xp=xp, cutoff=cutoff)
            in_at[p - 1], in_at[p] = (in_at[p], in_at[p - 1])
            out_at[p - 1], out_at[p] = (out_at[p], out_at[p - 1])
            p -= 1
            swaps += 1
            if log and swaps % 25 == 0:
                log({'swaps': swaps, 'max_bond': int(max((w.shape[3] for w in W[:-1])))})
    fl = [0] * n
    fr = [0] * n
    for s in range(n):
        fl[in_at[s]] = s
        fr[out_at[s]] = s
    return (W, fl, fr, swaps)
