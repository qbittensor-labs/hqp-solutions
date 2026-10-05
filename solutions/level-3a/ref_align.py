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

"""Bring a REFERENCE-absorber operator (two wire frames) into the single-frame SyncMPO form dresid needs.

The reference absorber (unswap.mpo_compress_unswap) routes the two halves of a block independently, so the
operator it returns has site i carrying OUTPUT wire siteR[i] (leg k{i}) and INPUT wire siteL[i] (leg b{i}), and
MEASURED on 12 saved operators (/sp/mpo) the two frames differ on 5-37 wires (3-153 adjacent inversions). The
residual extractor (dresid.build) needs ONE wire per site on both legs, so:

  1. from_TS: the (tensor, index-names) list becomes a syncd.SyncMPO (site tensors (l, o, i, r), the sqrt(2)
     per site that syncd.to_TS multiplies in is divided out, stray singleton indices squeezed);
  2. realign: one-leg SWAP gates (dresid.two_general on side 'L' = input legs, or 'R' = output legs) applied as
     an adjacent-transposition sort until the sorted leg's wire order equals the other leg's. Only INVERSIONS are
     swapped, so the permutation part of the operator never gains a crossing at any bond: its bond dimension is
     monotone non-increasing (the operator is near-identity in wire labels, so the realigned MPO is the
     near-product one). Every swap is one SVD at the extractor's cutoff/bond cap; the discarded weight is
     accumulated in the MPO's `lost` and reported, so a caller can refuse an unreliable realignment.

Env (excision_solver.measure_D reads them; defaults shown):
    HQP_D_RESID_REF=1            0 disables the reference-path residual correction altogether
    HQP_D_REALIGN_SIDE=L         which leg is sorted: L (input legs, frame = output wires), R, or auto (both,
                                 keep the one with the smaller final bond, then the smaller discarded weight)
    HQP_RESID_CUTOFF / HQP_RESID_MAX_BOND / HQP_RESID_MAX_LOST are shared with dresid (1e-6 / 512 / 1e-3).
"""
import copy
import math
import os
import time

import numpy as np

import syncd

SWAP4 = np.eye(4, dtype=complex)[[0, 2, 1, 3]]


def _np(d):
    if hasattr(d, "detach"):                                    # torch tensor (possibly on the GPU)
        d = d.detach().cpu().numpy()
    return np.asarray(d, dtype=complex)


def from_TS(TS, n, cutoff=None, max_bond=None):
    """SyncMPO from measure_D's TS = [(tensor, inds)] with legs k{i} (output) / b{i} (input) and one bond name
    shared with each neighbour. Inverts to_TS's normalisation (x sqrt2 per site) so a unitary has norm 1."""
    # HQP_D_REALIGN_CUTOFF: truncation while realigning; defaults to dresid's HQP_RESID_CUTOFF (1e-6). MEASURED on the
    # 12 saved operators the realignment discards 0-6e-6 of the weight at 1e-6 and the bond never grows past the
    # operator's own peak, so the extractor's cutoff is the natural choice; a tighter one costs nothing but time.
    if cutoff is None:
        cutoff = float(os.environ.get("HQP_D_REALIGN_CUTOFF", os.environ.get("HQP_RESID_CUTOFF", "1e-6")))
    max_bond = int(os.environ.get("HQP_RESID_MAX_BOND", "512")) if max_bond is None else int(max_bond)
    M = syncd.SyncMPO(n, cutoff=float(cutoff), max_bond=max_bond)
    T = []
    for i in range(n):
        d, inds = TS[i]
        d = _np(d)
        inds = list(inds)
        if d.ndim != len(inds):
            raise ValueError(f"site {i}: tensor rank {d.ndim} != {len(inds)} index names")
        left = [x for x in inds if i > 0 and x in TS[i - 1][1]]
        right = [x for x in inds if i < n - 1 and x in TS[i + 1][1]]
        ko, bi = f"k{i}", f"b{i}"
        if ko not in inds or bi not in inds:
            raise ValueError(f"site {i}: legs {ko}/{bi} missing from {inds}")
        other = [x for x in inds if x not in left and x not in right and x not in (ko, bi)]
        for x in other:                                          # e.g. a stray dim-1 'b' index (seen in a saved MPO)
            if d.shape[inds.index(x)] != 1:
                raise ValueError(f"site {i}: unexpected index {x} of dim {d.shape[inds.index(x)]}")
        perm = [inds.index(x) for x in left] + [inds.index(ko), inds.index(bi)] + [inds.index(x) for x in right] \
            + [inds.index(x) for x in other]
        A = d.transpose(perm)
        Dl = int(np.prod([A.shape[j] for j in range(len(left))])) if left else 1
        Dr = int(np.prod([A.shape[len(left) + 2 + j] for j in range(len(right))])) if right else 1
        T.append(np.ascontiguousarray(A.reshape(Dl, 2, 2, Dr)) / math.sqrt(2.0))
    for i in range(n - 1):
        if T[i].shape[3] != T[i + 1].shape[0]:
            raise ValueError(f"bond {i}: {T[i].shape[3]} vs {T[i + 1].shape[0]}")
    if T[0].shape[0] != 1 or T[-1].shape[3] != 1:
        raise ValueError("open boundary bonds are not 1-dimensional")
    M.T = T
    M.canonicalise()                                             # exact QR sweep, centre 0
    M.peak = max(M.bonds()) if n > 1 else 1
    return M


def norm2(M):
    """Squared Frobenius norm in SyncMPO normalisation (1 for a unitary)."""
    nrm = np.ones((1, 1), complex)
    for s in range(M.n):
        T = M.T[s]
        nrm = np.einsum("ab,aoic,boid->cd", nrm, T.conj(), T)
    return float(np.real(nrm[0, 0]))


def realign(M, siteR, siteL, side="L", log=None):
    """Sort one leg's wire order into the other's with adjacent one-leg SWAPs (inversions only).

    side 'L': the INPUT legs are permuted until siteL == siteR; the common frame is wire_at[s] = siteR[s].
    side 'R': the OUTPUT legs are permuted; wire_at[s] = siteL[s].
    Returns (M, wire_at, info) -- M is modified in place."""
    import dresid
    log = log or (lambda m: None)
    n = M.n
    t0 = time.time()
    if side == "L":
        target = [int(siteR[s]) for s in range(n)]
        cur = [int(siteL[s]) for s in range(n)]
    elif side == "R":
        target = [int(siteL[s]) for s in range(n)]
        cur = [int(siteR[s]) for s in range(n)]
    else:
        raise ValueError(side)
    if sorted(target) != list(range(n)) or sorted(cur) != list(range(n)):
        raise ValueError("frames are not permutations of the wires")
    tpos = {w: s for s, w in enumerate(target)}
    key = [tpos[w] for w in cur]                                 # target site of the wire now at each site
    inv0 = sum(1 for a in range(n) for b in range(a + 1, n) if key[a] > key[b])
    bonds0 = M.bonds()
    lost0 = M.lost
    nsw = 0
    peak = max(bonds0) if n > 1 else 1
    direction = 1
    # GUARDS (review 2026-09-24): a one-leg swap at the 512 bond cap is a 4.3 s SVD and n = 48 allows up to 1128
    # inversions, so a badly entangled operator could sort for over an hour before the discarded-weight refusal
    # fires. Abort on wall clock (HQP_D_REALIGN_MAX_S) or when a swap hits the bond cap: the caller's except then
    # keeps the ansatz-only result for that block, exactly the pre-change behaviour.
    max_s = float(os.environ.get("HQP_D_REALIGN_MAX_S", "120"))
    while True:
        changed = False
        rng_ = range(n - 1) if direction == 1 else range(n - 2, -1, -1)
        for s in rng_:
            if key[s] > key[s + 1]:
                if time.time() - t0 > max_s:
                    raise RuntimeError(f"realignment exceeded {max_s:g} s after {nsw} swaps (bond {max(M.bonds())})")
                k = dresid.two_general(M, s, SWAP4, side)
                peak = max(peak, k)
                if k >= M.max_bond:
                    raise RuntimeError(f"realignment hit the bond cap {M.max_bond} after {nsw + 1} swaps")
                key[s], key[s + 1] = key[s + 1], key[s]
                cur[s], cur[s + 1] = cur[s + 1], cur[s]
                nsw += 1
                changed = True
        if not changed:
            break
        direction = -direction                                   # cocktail sweeps: the centre is already at this end
    if cur != target:
        raise RuntimeError("realignment did not converge")
    info = {"side": side, "swaps": nsw, "inversions": inv0, "differ": sum(1 for s in range(n) if siteR[s] != siteL[s]),
            "bond_before": max(bonds0) if n > 1 else 1, "bond_after": max(M.bonds()) if n > 1 else 1, "peak": peak,
            "lost": M.lost - lost0, "secs": time.time() - t0}
    log(f"  realign[{side}]: {info['differ']} wires differ between the legs, {inv0} inversions -> {nsw} one-leg swaps, "
        f"bond {info['bond_before']} -> {info['bond_after']} (peak {peak}), discarded {info['lost']:.2e} "
        f"[{info['secs']:.1f}s]")
    return M, list(target), info


def from_frames(TS, n, siteR, siteL, side=None, log=None, cutoff=None, max_bond=None):
    """TS + the two frames -> (SyncMPO in one frame, wire_at, info). side None reads HQP_D_REALIGN_SIDE (L);
    cutoff None reads HQP_D_REALIGN_CUTOFF (1e-6) -- D's own cutoff (6e-4) is NOT a sensible realignment cutoff."""
    log = log or (lambda m: None)
    side = (os.environ.get("HQP_D_REALIGN_SIDE", "L") if side is None else side).strip().upper()
    M0 = from_TS(TS, n, cutoff=cutoff, max_bond=max_bond)
    if side in ("L", "R"):
        return realign(M0, siteR, siteL, side=side, log=log)
    if side != "AUTO":
        raise ValueError(f"HQP_D_REALIGN_SIDE={side!r}")
    best = None
    for sd in ("L", "R"):
        Ms, wa, inf = realign(copy.deepcopy(M0), siteR, siteL, side=sd, log=log)
        key_ = (inf["bond_after"], inf["lost"])
        if best is None or key_ < best[0]:
            best = (key_, Ms, wa, inf)
    _k, Ms, wa, inf = best
    inf = dict(inf, chosen=inf["side"])
    return Ms, wa, inf
