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

"""SYNCHRONOUS two-sided absorber for a mirror block (CPU, numpy complex128).

The reference absorber (unswap.py) routes the two halves of a block INDEPENDENTLY onto the MPO chain (SabreSwap per
side) and then searches for leg swaps that undo the damage: MEASURED 2026-09-19 (d3_s2, production rehearsal) 65 % of
a D measurement is spent in those unswap rounds, the operator hits bond 256 every ~15 unitaries although what is
being absorbed cancels to a near-product operator, and one block takes 2000-3000 s.

Here both legs of the operator always share ONE layout:
  * M starts as the explicit SWAP_tau layer with tau-partners on neighbouring sites (bond 4 on alternate bonds);
  * the gates of the two halves are scheduled in LOCKSTEP from the mirror surface outward: a CZ of the L half and
    its twin of the R half (same wires after the tau relabelling) are applied back to back, so at matched depth M
    is the near-product operator D-so-far;
  * ONE routing of that joint schedule (SabreSwap on a line); every routing swap is applied to BOTH legs, which is
    a mere site exchange of M (exact, and cheap while M is near-product) -- nothing to un-swap afterwards.
The result is D = Pi_tau . E in a known frame (site s carries wire `wire_at[s]` on both legs).

Everything is exact except the SVD truncation (relative cutoff, as in the reference).
"""
import collections
import math
import os
import time

import numpy as np


def u3m(th, ph, la):
    c, s = math.cos(th / 2.0), math.sin(th / 2.0)
    return np.array([[c, -np.exp(1j * la) * s], [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]], complex)


class TooBig(RuntimeError):
    pass


class SyncMPO:
    """MPO with site tensors T[s][l, o, i, r] (o = output/ket leg, i = input/bra leg), normalised so that a unitary
    has Frobenius norm 1, kept in mixed-canonical form around `centre`."""

    def __init__(self, n, cutoff=6e-4, max_bond=1024):
        self.n, self.cutoff, self.max_bond = n, float(cutoff), int(max_bond)
        self.mode = os.environ.get("SYNCD_CUTOFF_MODE", "rsum2").strip().lower()
        self.T = [(np.eye(2, dtype=complex) / math.sqrt(2.0)).reshape(1, 2, 2, 1) for _ in range(n)]
        self.centre = 0
        self.peak = 1
        self.nsvd = 0
        self.lost = 0.0            # accumulated discarded weight (relative to the running norm)

    # ---- canonical form ----
    def _shift_right(self):
        s = self.centre
        A = self.T[s]; Dl, _, _, Dr = A.shape
        Q, R = np.linalg.qr(A.reshape(Dl * 4, Dr))
        self.T[s] = Q.reshape(Dl, 2, 2, Q.shape[1])
        self.T[s + 1] = np.tensordot(R, self.T[s + 1], axes=(1, 0))
        self.centre = s + 1

    def _shift_left(self):
        s = self.centre
        A = self.T[s]; Dl, _, _, Dr = A.shape
        Q, R = np.linalg.qr(A.reshape(Dl, 4 * Dr).T)
        self.T[s] = Q.T.reshape(Q.shape[1], 2, 2, Dr)
        self.T[s - 1] = np.tensordot(self.T[s - 1], R.T, axes=(3, 0))
        self.centre = s - 1

    def move(self, s):
        while self.centre < s:
            self._shift_right()
        while self.centre > s:
            self._shift_left()

    def canonicalise(self):
        """(re)build the canonical form from scratch: right-to-left LQ sweep, centre = 0."""
        self.centre = self.n - 1
        self.move(0)

    # ---- operations ----
    def one(self, s, u, side):
        if side == "R":                                    # g . M : acts on the output leg
            self.T[s] = np.einsum("po,loir->lpir", u, self.T[s])
        else:                                              # M . g : acts on the input leg
            self.T[s] = np.einsum("loir,ip->lopr", self.T[s], u)

    def two(self, s, kind, side=None, prefer=None):
        """kind: 'cz' (side 'L'/'R') or 'swap' (both legs) on sites s, s+1."""
        if self.centre <= s:
            self.move(s)
        else:
            self.move(s + 1)
        A, B = self.T[s], self.T[s + 1]
        Dl, Dr = A.shape[0], B.shape[3]
        th = np.tensordot(A, B, axes=(3, 0))               # l o1 i1 o2 i2 r
        if kind == "cz":
            th = th.copy()
            if side == "R":
                th[:, 1, :, 1, :, :] *= -1.0
            else:
                th[:, :, 1, :, 1, :] *= -1.0
        elif kind == "swap":
            th = th.transpose(0, 3, 4, 1, 2, 5)
        else:
            raise ValueError(kind)
        mat = np.ascontiguousarray(th).reshape(Dl * 4, 4 * Dr)
        try:
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        except np.linalg.LinAlgError:
            import scipy.linalg as sla
            U, S, Vh = sla.svd(mat, full_matrices=False, lapack_driver="gesvd")
        self.nsvd += 1
        if self.mode == "rel":
            k = int(np.count_nonzero(S > self.cutoff * S[0])) if S[0] > 0 else 1
        else:
            # "rsum2" = quimb's default, i.e. what the reference absorber does with the same number: drop the smallest
            # singular values while the DISCARDED weight stays below cutoff x total. MEASURED 2026-09-19: with "rel"
            # every imperfectly cancelled (retrained, ~1e-3) mirror pair leaves a link just above the threshold and
            # the operator reaches bond 512 after 130 CZ; the reference discards exactly those.
            s2 = S ** 2
            tail = np.cumsum(s2[::-1])[::-1]                   # tail[j] = sum_{i>=j} s_i^2
            keep = np.nonzero(tail > self.cutoff * s2.sum())[0]
            k = int(keep[-1]) + 1 if keep.size else 1
        k = max(1, min(k, self.max_bond))
        tot = float((S ** 2).sum())
        if tot > 0:
            self.lost += float((S[k:] ** 2).sum()) / tot
        self.peak = max(self.peak, k)
        if (prefer if prefer is not None else s + 1) == s:
            self.T[s] = (U[:, :k] * S[:k]).reshape(Dl, 2, 2, k)
            self.T[s + 1] = Vh[:k].reshape(k, 2, 2, Dr)
            self.centre = s
        else:
            self.T[s] = U[:, :k].reshape(Dl, 2, 2, k)
            self.T[s + 1] = (S[:k, None] * Vh[:k]).reshape(k, 2, 2, Dr)
            self.centre = s + 1
        return k

    def bonds(self):
        return [self.T[s].shape[3] for s in range(self.n - 1)]

    def log10_norm2(self):
        return float(np.log10(max(float((np.abs(self.T[self.centre]) ** 2).sum()), 1e-300)))

    def dense(self):
        """2^n x 2^n matrix in SITE order (tests only), de-normalised."""
        n = self.n
        M = self.T[0]
        for s in range(1, n):
            M = np.tensordot(M, self.T[s], axes=(M.ndim - 1, 0))
        M = M.reshape([2] * (2 * n))                      # o0 i0 o1 i1 ...
        M = M.transpose([2 * s for s in range(n)] + [2 * s + 1 for s in range(n)])
        return M.reshape(2 ** n, 2 ** n) * (2.0 ** (n / 2.0))


def _lcs(a, b):
    n1, n2 = len(a), len(b)
    D = np.zeros((n1 + 1, n2 + 1), np.int32)
    for i in range(n1):
        for j in range(n2):
            D[i + 1, j + 1] = D[i, j] + 1 if a[i] == b[j] else max(D[i, j + 1], D[i + 1, j])
    i, j, m = n1, n2, []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and D[i, j] == D[i - 1, j - 1] + 1:
            m.append((i - 1, j - 1)); i -= 1; j -= 1
        elif D[i - 1, j] >= D[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return m[::-1]


def _lcs_w(a, b, wfn):
    """Maximum-WEIGHT common subsequence: like _lcs, but a match (i, j) with equal labels is worth wfn(i, j) > 0
    instead of 1.  Used to prefer alignments whose CZ pairs carry exact-twin u3 neighbours (mirror evidence) over
    alignments through inserted, evidence-free CZs of the same labels."""
    n1, n2 = len(a), len(b)
    D = np.zeros((n1 + 1, n2 + 1))
    Wm = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            if a[i] == b[j]:
                Wm[i, j] = wfn(i, j)
                D[i + 1, j + 1] = max(D[i, j] + Wm[i, j], D[i, j + 1], D[i + 1, j])
            else:
                D[i + 1, j + 1] = max(D[i, j + 1], D[i + 1, j])
    i, j, m = n1, n2, []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and abs(D[i, j] - (D[i - 1, j - 1] + Wm[i - 1, j - 1])) < 1e-9:
            m.append((i - 1, j - 1)); i -= 1; j -= 1
        elif D[i - 1, j] >= D[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return m[::-1]


def schedule(gates, n, tau, Eidx, split, twins=None, log=None):
    """Joint lockstep schedule. Returns (ops, stats); ops = [(side, kind, wires, mat, raw_index)], side 'L' (applied as
    M.g, walking BACKWARDS in time from the surface) or 'R' (g.M, forwards, wires relabelled by tau).

    MEASURED structure of a real block, per wire from the surface outward (d3_s2): the two sides carry the SAME
    sequence of mirror gates (equal relabelled wires), plus one-sided INSERTIONS -- circuit identities used as
    obfuscation: cz(a,z) G cz(a,z) around a core CZ or a mirror gate, nested versions with their compensating CZ,
    extra CZ(x, tau x). So:
      1. mirror pairs = per-wire LCS alignment of the two sides (both wires must agree), relative to THIS split
         (the index-based twin map misses every pair whose members lie on one side of the scan centre);
      2. the schedule is driven by the mirror pairs, shallowest first; whatever blocks a pair on either side is
         executed JUST IN TIME (an insertion opened early stays open: bond x2 for as long as it waits);
      3. an insertion's closing CZ (same side, same wires as an open one) is executed as soon as it is executable.
    ANY order is exact; these rules only decide how long an un-cancelled CZ stays open."""
    import sys
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    E = set(Eidx)
    is_left = lambda i: i <= split[gates[i][1][0]]                                  # noqa: E731
    qL = [collections.deque() for _ in range(n)]
    qR = [collections.deque() for _ in range(n)]
    for i in sorted(E, reverse=True):
        if is_left(i):
            for w in gates[i][1]:
                qL[w].append(i)
    for i in sorted(E):
        if not is_left(i):
            for w in gates[i][1]:
                qR[tau[w]].append(i)
    Q = {"L": qL, "R": qR}
    wires_of = lambda i, side: tuple(gates[i][1]) if side == "L" else tuple(tau[w] for w in gates[i][1])   # noqa: E731
    other = lambda i, side, w: [v for v in wires_of(i, side) if v != w][0]            # noqa: E731
    # ---- 1. mirror pairs by doubly-consistent LCS ----
    # COMMUTATION-AWARE alignment (SYNCD_LCS_COMMUTE=1; default OFF: mixed census, see status doc 2026-09-24): on a wire, CZs separated only by DIAGONAL u3s
    # (theta = 0 mod 2pi within SYNCD_DIAG_TOL) or by nothing commute, and the two halves of a block may list such
    # a run in different orders (MEASURED 2026-09-24 on d3s1_w20r, a mirrored Pauli-frame wrap cz(0,46) X_0 G X_0
    # cz(0,46) next to a CZ(46,30) double: wire 46 reads "30 30 0 0" on the L side and "0 0 30 30" on the R side, so
    # the order-sensitive LCS on wire 46 votes for the 30s while wire 0 votes for the 0s and the wrap's two CZs stay
    # unpaired -> two open insertions each; 35-40 % of exactly mirrored wraps were lost that way and w60r's block 3385
    # hit TooBig at bond 470).  Inside each commuting run the labels are put in a canonical order (stable sort by
    # partner label: equal labels keep their surface order, and equal-label CZs on equal wires are the SAME gate, so
    # any consistent choice is exact) before the LCS; the schedule below still executes gates in their true order.
    votes = collections.Counter()
    commute = os.environ.get("SYNCD_LCS_COMMUTE", "0").strip() not in ("0", "false", "no", "off", "")
    dtol = float(os.environ.get("SYNCD_DIAG_TOL", "1e-6"))

    def _canon(q, side, w):
        """CZ gate indices of queue q (surface order) with commuting runs stable-sorted by partner label."""
        if not commute:
            return [i for i in q if gates[i][0] == "cz"]
        out, run = [], []
        for i in q:
            if gates[i][0] == "cz":
                run.append(i)
                continue
            th = gates[i][2][0] if gates[i][2] is not None else 0.0
            th = abs(th) % (2 * math.pi)
            if min(th, 2 * math.pi - th) < dtol:                    # diagonal u3: the run continues
                continue
            out.extend(sorted(run, key=lambda j: other(j, side, w))); run = []
        out.extend(sorted(run, key=lambda j: other(j, side, w)))
        return out

    # EVIDENCE-WEIGHTED alignment (SYNCD_LCS_EVIDENCE = lambda, default 1; 0 = plain LCS): a match is worth
    # 1 + lambda * (number of exact-twin u3 neighbours of the pair, 0..4, as excise.evidence counts them).  MEASURED
    # 2026-09-24 on d3_s1 + 40 ONE-SIDED identity insertions cz(a,z) G cz(a,z) per block (the platform's own
    # obfuscation shape): every inserted CZ carries a label already present on its wires, the plain LCS sometimes
    # aligns the genuine mirror CZ(a,z) with an inserted one on one wire and with the genuine one on the other, the
    # vote splits and a real mirror pair is lost (block 3260: 169 -> 134 pairs, TooBig at bond 512 after 72 CZ).
    # Inserted CZs have no twin neighbours, mirror pairs have 2-4, so the weight steers the alignment.
    lam = float(os.environ.get("SYNCD_LCS_EVIDENCE", "1"))
    ev_tol = float(os.environ.get("SYNCD_EV_TOL", "1e-3"))
    listL = [list(q) for q in qL]
    listR = [list(q) for q in qR]
    posL = [{i: k for k, i in enumerate(lst)} for lst in listL]
    posR = [{i: k for k, i in enumerate(lst)} for lst in listR]

    def _fold(th):
        a = abs(th) % (2 * math.pi)
        return 2 * math.pi - a if a > math.pi else a

    def _twin_u(i, j):
        if gates[i][0] != "u" or gates[j][0] != "u":
            return False
        a, b = _fold(gates[i][2][0]), _fold(gates[j][2][0])
        return abs(a - b) < ev_tol or abs(a + b - math.pi) < ev_tol

    def _evidence(l, r):
        """Exact-twin u3 neighbours of the candidate pair (l on L, r on R), inner and outer on each of l's wires."""
        k = 0
        for x in gates[l][1]:
            y = tau[x]                      # r sits on physical wire tau x, i.e. relabelled wire x: queue index x
            kl, kr = posL[x].get(l), posR[x].get(r)
            if kl is None or kr is None:
                continue
            for d in (-1, +1):
                a_, b_ = kl + d, kr + d
                if 0 <= a_ < len(listL[x]) and 0 <= b_ < len(listR[x]) and _twin_u(listL[x][a_], listR[x][b_]):
                    k += 1
        return k

    for w in range(n):
        Ls = _canon(qL[w], "L", w)
        Rs = _canon(qR[w], "R", w)
        la = [other(i, "L", w) for i in Ls]
        lb = [other(i, "R", w) for i in Rs]
        if lam > 0:
            matches = _lcs_w(la, lb, lambda i_, j_: 1.0 + lam * _evidence(Ls[i_], Rs[j_]))
        else:
            matches = _lcs(la, lb)
        for a_, b_ in matches:
            votes[(Ls[a_], Rs[b_])] += 1
    pair_of = {}
    for (l, r), v in votes.items():
        if v == 2 and ("L", l) not in pair_of and ("R", r) not in pair_of:
            pair_of[("L", l)] = ("R", r)
            pair_of[("R", r)] = ("L", l)
    # ASAP level on each side (distance from the surface in CZ layers)
    level = {}
    for side in ("L", "R"):
        lv = [0] * n
        order = sorted((i for i in E if is_left(i) == (side == "L") and gates[i][0] == "cz"), reverse=(side == "L"))
        for i in order:
            a, b = wires_of(i, side)
            v = max(lv[a], lv[b]) + 1
            lv[a] = lv[b] = v
            level[(side, i)] = v
    ops = []
    done = set()
    stats = {"pairs": 0, "mirror_pairs_found": len(pair_of) // 2, "insertions": 0, "closings": 0, "late_twins": 0,
             "cycle_breaks": 0, "max_open": 0}
    open_same = collections.Counter()
    open_single = set()

    def emit(side, i):
        t, q, p = gates[i]
        ws = wires_of(i, side)
        ops.append((side, "cz" if t == "cz" else "u", ws, None if t == "cz" else u3m(*p), i))
        for w in ws:
            assert Q[side][w][0] == i, "schedule order violation"
            Q[side][w].popleft()
        done.add((side, i))
        if t == "cz" and (side, i) not in pair_of:
            key = (side, frozenset(ws))
            open_same[key] += 1
            if open_same[key] % 2 == 1:
                stats["insertions"] += 1
            else:
                stats["closings"] += 1
        stats["max_open"] = max(stats["max_open"], sum(1 for v in open_same.values() if v % 2) + len(open_single))

    def executable(side, i):
        return all(Q[side][w] and Q[side][w][0] == i for w in wires_of(i, side))

    in_progress = set()
    debug_cycles = bool(os.environ.get("SYNCD_DEBUG_CYCLES"))

    def ensure_ready(side, g):
        """Execute whatever precedes g on its wires (on its side)."""
        for w in wires_of(g, side):
            guard = 0
            while Q[side][w] and Q[side][w][0] != g:
                h = Q[side][w][0]
                guard += 1
                if guard > 100000:
                    raise RuntimeError("syncd schedule: no progress")
                if gates[h][0] != "cz":
                    emit(side, h)
                    continue
                if (side, h) in pair_of and pair_of[(side, h)] not in done:
                    do_pair((side, h))
                    if (side, h) not in done:                       # cycle: the pair could not complete -> h goes alone
                        if debug_cycles:
                            p_ = pair_of[(side, h)]
                            stats.setdefault("_cycles", []).append({
                                "gate": (side, h), "partner": p_, "needed_by": (side, g), "in_progress": sorted(in_progress),
                                "partner_heads": {w_: (Q[p_[0]][w_][0] if Q[p_[0]][w_] else None) for w_ in wires_of(p_[1], p_[0])}})
                        ensure_ready(side, h)
                        if (side, h) not in done:
                            emit(side, h)
                            open_single.add((side, h))
                            stats["cycle_breaks"] += 1
                else:
                    ensure_ready(side, h)
                    if (side, h) not in done:
                        emit(side, h)
                        if (side, h) in pair_of:                    # its twin went alone earlier
                            open_single.discard(pair_of[(side, h)])
                            stats["late_twins"] += 1

    def do_pair(node):
        a = node if node[0] == "L" else pair_of[node]
        b = pair_of[a]
        if a in done or b in done or a in in_progress:
            return
        in_progress.add(a)
        ensure_ready(*a)
        ensure_ready(*b)
        if a not in done and b not in done and executable(*a) and executable(*b):
            emit(*a); emit(*b)
            stats["pairs"] += 1
        in_progress.discard(a)
        close_wraps()

    def close_wraps():
        again = True
        while again:
            again = False
            for side in ("L", "R"):
                for w in range(n):
                    if not Q[side][w]:
                        continue
                    h = Q[side][w][0]
                    if gates[h][0] != "cz":
                        if all(Q[side][v][0] == h for v in wires_of(h, side)):
                            emit(side, h); again = True
                        continue
                    if (side, h) in pair_of or not executable(side, h):
                        continue
                    if open_same[(side, frozenset(wires_of(h, side)))] % 2 == 1:
                        emit(side, h); again = True

    close_wraps()
    # order of the lockstep pairs (SYNCD_PAIR_ORDER): "L" = the L member's ASAP level (historical), "max" / "sum" = both
    # members' levels, so a pair that is shallow on one side but deep on the other is not forced early on the deep side
    # (under one-sided insertions that is what opens the deep side's insertions early and creates lockstep cycles).
    _order = os.environ.get("SYNCD_PAIR_ORDER", "L").strip().lower()
    if _order == "max":
        _key = lambda k: (max(level[k], level[pair_of[k]]), level[k], k[1])            # noqa: E731
    elif _order == "sum":
        _key = lambda k: (level[k] + level[pair_of[k]], level[k], k[1])               # noqa: E731
    else:
        _key = lambda k: (level[k], k[1])                                             # noqa: E731
    for node in sorted((k for k in pair_of if k[0] == "L"), key=_key):
        do_pair(node)
    # tails: whatever is left (insertions beyond the last mirror pair, twins of cycle-broken singles)
    for side in ("L", "R"):
        rest = sorted((i for i in E if is_left(i) == (side == "L") and (side, i) not in done), reverse=(side == "L"))
        for i in rest:
            if (side, i) in done:
                continue
            ensure_ready(side, i)
            if (side, i) not in done:
                emit(side, i)
                if (side, i) in pair_of:
                    stats["late_twins"] += 1
    left = sum(len(q) for q in qL) + sum(len(q) for q in qR)
    if left:
        raise RuntimeError(f"syncd schedule incomplete: {left} wire slots left")
    stats["open_at_end"] = sum(1 for v in open_same.values() if v % 2)
    # STRUCTURAL sanity of tau (diagnostic, never a gate): in every real block each tau-pair carries at least one
    # "core" CZ(x, tau x) inside the cut -- MEASURED 24/24 pairs on both d3 samples, 20/20 on d2_s1. A pair with no
    # core CZ would mean the discovered tau pairs two wires the excised region does not actually swap.
    with_core = {frozenset(gates[i][1]) for i in E if gates[i][0] == "cz" and tau[gates[i][1][0]] == gates[i][1][1]}
    stats["tau_pairs"] = sum(1 for x in range(n) if x < tau[x])
    stats["pairs_with_core"] = len(with_core)
    if os.environ.get("SYNCD_DEBUG_PAIRS"):
        stats["_pair_of"] = dict(pair_of)
    return ops, stats


def route(ops, n, pos0, seed=123, trials=None):
    """One SabreSwap routing of the joint schedule on a line. Returns [('swap', p, None) | ('op', p_tuple, k)]."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.passes import SabreSwap
    trials = int(os.environ.get("SYNCD_SABRE_TRIALS", "200")) if trials is None else trials
    qc = QuantumCircuit(n)
    for k, (side, kind, ws, mat, i) in enumerate(ops):
        qc.append(Gate(name=f"g{len(ws)}", num_qubits=len(ws), params=[], label=str(k)), [pos0[w] for w in ws])
    out = SabreSwap(coupling_map=CouplingMap.from_line(n), heuristic="decay", trials=trials, seed=seed)(qc)
    seq = []
    for inst in out.data:
        ps = tuple(out.find_bit(q).index for q in inst.qubits)
        if inst.operation.name == "swap":
            seq.append(("swap", ps, None))
        else:
            seq.append(("op", ps, int(inst.operation.label)))
    return seq


def route_super(ops, n, tau, seed=123, trials=None):
    """Route on SUPER-SITES: a tau-pair always occupies two neighbouring sites, so the pair-internal CZ(x, tau x) link
    (open for most of the absorption) never crosses another site. Returns (order0, seq) with seq = [('swap', site) |
    ('op', k)] at the SITE level and order0 = initial wire order."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.passes import SabreSwap
    trials = int(os.environ.get("SYNCD_SABRE_TRIALS", "200")) if trials is None else trials
    groups = []
    gid = {}
    for x in range(n):
        if x in gid:
            continue
        g = [x] if tau[x] == x else [x, tau[x]]
        for w in g:
            gid[w] = len(groups)
        groups.append(g)
    G = len(groups)
    qc = QuantumCircuit(G)
    for k, (side, kind, ws, mat, i) in enumerate(ops):
        gs = sorted({gid[w] for w in ws})
        qc.append(Gate(name=f"g{len(gs)}", num_qubits=len(gs), params=[], label=str(k)), gs)
    strict = os.environ.get("SYNCD_ROUTER", "strict").strip().lower() == "strict"
    if strict and G > 1:
        # STRICT order (default): the schedule opens an insertion just in time; SabreSwap only keeps the per-site order
        # and would execute an opener whenever its sites happen to be neighbours, i.e. far too early.
        K = int(os.environ.get("SYNCD_ROUTE_LOOKAHEAD", "12"))
        inter = [(k, tuple(sorted({gid[w] for w in o[2]}))) for k, o in enumerate(ops) if o[1] == "cz" and gid[o[2][0]] != gid[o[2][1]]]
        nxt_idx = {k: j for j, (k, _) in enumerate(inter)}
        posg = list(range(G))                             # group -> position
        atg = list(range(G))                              # position -> group
        data = []                                         # ('swap', P) | ('op', k, P or None)
        for k, o in enumerate(ops):
            if k not in nxt_idx:
                data.append(("op", k, None))
                continue
            j = nxt_idx[k]
            A, B = inter[j][1]
            while abs(posg[A] - posg[B]) > 1:
                best = None
                for mover, target in ((A, B), (B, A)):
                    step = 1 if posg[target] > posg[mover] else -1
                    P = min(posg[mover], posg[mover] + step)
                    ga, gb = atg[P], atg[P + 1]
                    posg[ga], posg[gb] = posg[gb], posg[ga]           # tentative
                    cost = 0.0
                    for jj in range(j, min(j + K, len(inter))):
                        x_, y_ = inter[jj][1]
                        cost += abs(posg[x_] - posg[y_]) * (0.8 ** (jj - j))
                    posg[ga], posg[gb] = posg[gb], posg[ga]
                    if best is None or cost < best[0]:
                        best = (cost, P)
                P = best[1]
                ga, gb = atg[P], atg[P + 1]
                atg[P], atg[P + 1] = gb, ga
                posg[ga], posg[gb] = P + 1, P
                data.append(("swap", P))
            data.append(("op", k, min(posg[A], posg[B])))
    else:
        out = SabreSwap(coupling_map=CouplingMap.from_line(G), heuristic="decay", trials=trials, seed=seed)(qc) if G > 1 else qc
        data = []
        for inst in out.data:
            ps_ = sorted(out.find_bit(q).index for q in inst.qubits)
            if inst.operation.name == "swap":
                data.append(("swap", ps_[0]))
            else:
                data.append(("op", int(inst.operation.label), ps_[0]))
    # expand to site level
    line = [list(g) for g in groups]                     # super-position -> wires in site order
    order0 = [w for g in line for w in g]
    seq = []

    def site_of(P):
        return sum(len(line[q]) for q in range(P))

    for item in data:
        if item[0] == "swap":
            P = item[1]
            A, B = line[P], line[P + 1]
            base = site_of(P)
            cur = A + B
            for bi in range(len(B)):                       # bubble every wire of B to the left across A
                for step in range(len(A)):
                    sidx = base + len(A) + bi - 1 - step
                    seq.append(("swap", sidx))
                    j = sidx - base
                    cur[j], cur[j + 1] = cur[j + 1], cur[j]
            line[P], line[P + 1] = cur[:len(B)], cur[len(B):]
            continue
        k = item[1]
        side, kind, ws, mat, i = ops[k]
        if kind == "cz" and gid[ws[0]] != gid[ws[1]]:
            P = item[2]
            A, B = line[P], line[P + 1]
            a = ws[0] if gid[ws[0]] == [gid[w] for w in A][0] else ws[1]
            b = ws[1] if a == ws[0] else ws[0]
            if A[-1] != a:                                 # flip A so that a faces B
                seq.append(("swap", site_of(P)))
                A.reverse()
            if B[0] != b:
                seq.append(("swap", site_of(P + 1)))
                B.reverse()
        seq.append(("op", k))
    return order0, seq


def absorb(gates, n, tau, Eidx, split, twins, cutoff=6e-4, max_bond=None, seed=123, log=print, deadline=None,
           give_up_bond=None):
    """D = Pi_tau . E for the cut Eidx. Returns (mpo, wire_at, info); wire_at[s] = wire carried by site s (both legs)."""
    t0 = time.time()
    # caps raised 512/384 -> 1024/768 on 2026-09-24: MEASURED on d3_s1 + 80 one-sided identity insertions per block, the
    # smaller block needs bond 1024 and comes out with its norm intact (10^-0.100) and D err 0.010 in 234-379 s per view;
    # with the old caps it was TooBig after 49 CZ and fell to the reference path. Hopeless blocks now run to the sync
    # budget (V13_SYNC_BUDGET_S) instead of giving up in seconds: that only delays their reference worker. Memory:
    # a bond-1024 c128 MPO is ~3 GB per attempt (the validator's container has 85 GB).
    max_bond = int(os.environ.get("SYNCD_MAX_BOND", "1024")) if max_bond is None else max_bond
    # give up EARLY when the operator is clearly not cancelling: every real block peaks at bond 24-64 (one synthetic
    # at 128); MEASURED on the synthetic "very low twin" block of h7 the bond sits on the 512 cap from 170 CZ on, the
    # norm is being cut away (10^-0.3, -0.6, ...) and the run only ends at its time budget (246 s per view).
    # give-up = the cap: a block that NEEDS the cap saturates at it while keeping its norm (d3_s1+160 block 3340: peak 1024,
    # norm 10^-0.100, D err 0.010), so a give-up below the cap would discard exactly the blocks the cap rescues; hopeless
    # blocks are bounded by the time budget instead (V13_SYNC_BUDGET_S) and rejected by the D gates.
    give_up_bond = int(os.environ.get("SYNCD_GIVE_UP_BOND", "1024")) if give_up_bond is None else give_up_bond
    ops, st = schedule(gates, n, tau, Eidx, split, twins, log)
    order, seq = route_super(ops, n, tau, seed=seed)
    nsw = sum(1 for s_ in seq if s_[0] == "swap")
    log(f"  syncd: {len(ops)} ops ({sum(1 for o in ops if o[1] == 'cz')} CZ), schedule: {st['pairs']}/{st['mirror_pairs_found']} mirror "
        f"pairs in lockstep, {st['insertions']} insertions opened / {st['closings']} closed, {st['late_twins']} late twins, "
        f"{st['cycle_breaks']} cycle breaks (max open {st['max_open']}, open at end {st['open_at_end']}), "
        f"{st['pairs_with_core']}/{st['tau_pairs']} tau-pairs carry a core CZ; routing: {nsw} site swaps "
        f"[{time.time()-t0:.1f}s]")
    pos0 = {w: k for k, w in enumerate(order)}
    M = SyncMPO(n, cutoff=cutoff, max_bond=max_bond)
    # explicit SWAP_tau on neighbouring sites
    SW = np.zeros((2, 2, 2, 2), complex)                  # o1 i1 o2 i2
    for a in (0, 1):
        for b in (0, 1):
            SW[b, a, a, b] = 0.5                          # out = (b, a) for in = (a, b); /2 = the two sites' 1/sqrt2
    U, S, Vh = np.linalg.svd(SW.reshape(4, 4))
    k = int(np.count_nonzero(S > 1e-12))
    for x in range(n):
        if x < tau[x]:
            a_, b_ = sorted((pos0[x], pos0[tau[x]]))
            assert b_ == a_ + 1
            M.T[a_] = (U[:, :k] * np.sqrt(S[:k])).reshape(1, 2, 2, k)
            M.T[b_] = (np.sqrt(S[:k])[:, None] * Vh[:k]).reshape(k, 2, 2, 1)
    M.canonicalise()
    wire_at = list(order)
    every = float(os.environ.get("SYNCD_LOG_EVERY", "30"))
    last = time.time()
    ncz = 0
    for kk, item in enumerate(seq):
        if item[0] == "swap":
            a_ = item[1]
            M.two(a_, "swap")
            wire_at[a_], wire_at[a_ + 1] = wire_at[a_ + 1], wire_at[a_]
        else:
            side, kind, ws, mat, i = ops[item[1]]
            ps = [wire_at.index(w) for w in ws]
            if kind == "u":
                M.one(ps[0], mat, side)
            else:
                a_, b_ = sorted(ps)
                assert b_ == a_ + 1, "routing/layout mismatch"
                M.two(a_, "cz", side)
                ncz += 1
        if give_up_bond and M.peak > give_up_bond:
            raise TooBig(f"bond {M.peak} > {give_up_bond} after {ncz} CZ")
        if deadline is not None and (kk & 63) == 0 and time.time() > deadline:
            raise TooBig(f"deadline after {ncz} CZ")
        if time.time() - last > every:
            last = time.time()
            log(f"  syncd: {kk}/{len(seq)} routed ops, {ncz} CZ, max bond now {max(M.bonds())} (peak {M.peak}), "
                f"log10 norm2 {M.log10_norm2():+.3f} [{time.time()-t0:.0f}s]")
    info = {"engine": "syncd", "ops": len(ops), "swaps": nsw, "peak_bond": M.peak, "final_bond": max(M.bonds()),
            "log10_norm2": M.log10_norm2(), "svds": M.nsvd, "secs": time.time() - t0, **st}
    log(f"  syncd: absorbed {ncz} CZ in {time.time()-t0:.0f}s: final bond {info['final_bond']}, peak {M.peak}, "
        f"log10 norm2 {info['log10_norm2']:+.3f}, {M.nsvd} SVDs")
    return M, wire_at, info


def to_TS(M, n):
    """Site tensors in the (tensor, index names) form measure_D's decode uses; de-normalised (x sqrt2 per site)."""
    import torch
    TS = []
    for s in range(n):
        A = M.T[s] * math.sqrt(2.0)
        inds = [f"_b{s-1}", f"k{s}", f"b{s}", f"_b{s}"]
        if s == 0:
            A = A[0]; inds = inds[1:]
        if s == n - 1:
            A = A[..., 0]; inds = inds[:-1]
        TS.append((torch.tensor(np.ascontiguousarray(A), dtype=torch.complex128), inds))
    return TS
