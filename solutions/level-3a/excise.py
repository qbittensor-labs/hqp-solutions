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

"""tau-block excision for HQP peaked circuits.

Construction model (verified 2026-09-11, verify_tau.py): each inserted mirror block is
    U . SWAP_tau . U^dag(relabelled by tau)
which as an operator is the wire permutation Pi_tau (NOT the identity -- excising it as the
identity is why E52 found nothing). Lossy "masking" patches make the block only approximately
Pi_tau; replacing the block's inner core by Pi_tau removes that loss as well as the depth.

A cut is a per-wire contiguous run of gates E (from the first to the last excised CZ on each
wire, 1q gates in between included). Replacing E by Pi_tau is exact iff E is an inner mirror
core U_c . SWAP . U_c^dag. Necessary conditions checked here (all loud, never repaired):
  * convex: every CZ outside E is on the same side (before/after) of E on both its wires;
  * pair-closed: no doubly-consistent CZ twin pair has exactly one member in E ("orphan");
  * every wire moved by tau is touched by E.
Gates after E on wire v act on the logical qubit tau(v), so they are relabelled; boundary u3s
are KEPT (they carry U's outer 1q layer, which cancels through Pi). Output bit on original wire
w = reduced qubit Pinv[w] with Pinv composed over blocks (tau1[tau2[w]] for two blocks).

CLI:  python3 excise.py QASM OUT.qasm CENTRE:TAU[,CENTRE:TAU...]   (TAU = "a-b,c-d,...")
Env:  EXC_MAX_DIST (900)  max distance (gate index) of an added twin pair from the centre
      EXC_SEED_BAND (0)   0 = seed from innermost twins; >0 = also force tau-pair CZs within band
      EXC_UNMATCHED (0)   1 = allow absorbing unmatched CZs that are next-outside on both wires
"""
import os, re, sys, math, collections
import numpy as np

EXC_MAX_DIST = int(os.environ.get('EXC_MAX_DIST', '900'))
EXC_SEED_BAND = int(os.environ.get('EXC_SEED_BAND', '0'))
EXC_UNMATCHED = int(os.environ.get('EXC_UNMATCHED', '0'))
EXC_WINDOW = int(os.environ.get('EXC_WINDOW', '0'))       # 0 = whole circuit; else twin search in c+-WINDOW
EXC_CONFIRM = int(os.environ.get('EXC_CONFIRM', '1'))     # require u3-twin evidence for every added CZ pair
EXC_TWIN_TOL = float(os.environ.get('EXC_TWIN_TOL', '1e-6'))
EXC_LOOKAHEAD = int(os.environ.get('EXC_LOOKAHEAD', '3'))   # tentative-item search depth (0 = off)


# ----------------------------------------------------------------------------- io
def _ev(x):
    return float(eval(x, {'pi': math.pi, '__builtins__': {}}))

def parse_qasm(text):
    n = None; g = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(('OPENQASM', 'include', 'creg', 'barrier', '//')):
            continue
        m = re.match(r'qreg\s+\w+\[(\d+)\];', s)
        if m: n = int(m.group(1)); continue
        m = re.match(r'u3?\(([^,]+),([^,]+),([^)]+)\)\s*\w+\[(\d+)\];', s)
        if m: g.append(('u', (int(m.group(4)),), tuple(_ev(x) for x in m.group(1, 2, 3)))); continue
        m = re.match(r'cz\s+\w+\[(\d+)\],\s*\w+\[(\d+)\];', s)
        if m: g.append(('cz', (int(m.group(1)), int(m.group(2))), None)); continue
        if s.startswith('measure'):
            continue
        raise ValueError(f'unsupported gate line: {s!r}')
    return n, g

def to_qasm(n, g):
    out = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{n}];']
    for t, q, p in g:
        if t == 'cz': out.append(f'cz q[{q[0]}],q[{q[1]}];')
        else: out.append(f'u({p[0]!r},{p[1]!r},{p[2]!r}) q[{q[0]}];')
    return '\n'.join(out) + '\n'

def parse_tau(n, s):
    t = list(range(n))
    for p in s.split(','):
        a, b = (int(x) for x in p.strip().split('-'))
        t[a] = b; t[b] = a
    return t


# ----------------------------------------------------------------------------- structure
def wire_ops(g, n):
    W = [[] for _ in range(n)]
    for i, (t, q, p) in enumerate(g):
        for x in q: W[x].append(i)
    pos = {}
    for x in range(n):
        for k, i in enumerate(W[x]): pos[(i, x)] = k
    return W, pos

def _lcs(a, b):
    n1, n2 = len(a), len(b)
    D = np.zeros((n1 + 1, n2 + 1), np.int32)
    for i in range(n1):
        for j in range(n2):
            D[i + 1, j + 1] = D[i, j] + 1 if a[i] == b[j] else max(D[i, j + 1], D[i + 1, j])
    i, j, m = n1, n2, []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and D[i, j] == D[i - 1, j - 1] + 1: m.append((i - 1, j - 1)); i -= 1; j -= 1
        elif D[i - 1, j] >= D[i, j - 1]: i -= 1
        else: j -= 1
    return m[::-1]

def cz_twins(g, n, c, tau, lo=0, hi=None):
    """Doubly-consistent mirror pairs: left CZ l=(x,y), lo<=l<c  <->  right CZ r=(tau x,tau y),
    c<=r<hi. Per wire w: CZs on w after c (outward) vs CZs on tau(w) before c (outward), aligned
    by LCS of partner labels (right partners mapped through tau). A pair is kept only if BOTH of
    its wires vote for it. The [lo,hi) window keeps a neighbouring block out. Returns right->left."""
    hi = len(g) if hi is None else hi
    votes = collections.Counter()
    for w in range(n):
        R = [(i, [x for x in g[i][1] if x != w][0]) for i in range(c, hi) if g[i][0] == 'cz' and w in g[i][1]]
        L = [(i, [x for x in g[i][1] if x != tau[w]][0]) for i in range(c - 1, lo - 1, -1) if g[i][0] == 'cz' and tau[w] in g[i][1]]
        for a, b in _lcs([tau[x] for _, x in R], [x for _, x in L]):
            votes[(R[a][0], L[b][0])] += 1
    return {r: l for (r, l), v in votes.items() if v == 2}

def _fold(th):
    a = abs(th) % (2 * math.pi)
    return 2 * math.pi - a if a > math.pi else a

def theta_twin(p1, p2, tol):
    """u3 twins under U -> U^dag: |theta| equal (Rz sweeping changes only phi/lambda) or
    pi-|theta| (a Pauli-X frame pushed through CZs)."""
    a, b = _fold(p1[0]), _fold(p2[0])
    return abs(a - b) < tol or abs(a + b - math.pi) < tol


# ----------------------------------------------------------------------------- the cut
class Cut:
    """Per-wire inclusive position range [s[w], e[w]] in wire_ops order. An EMPTY range is kept as
    a cursor (s = split, e = split-1) at the block centre, so 'left of E' / 'right of E' are always
    defined: left = U side (kept, original wires), right = U^dag side (kept, relabelled)."""
    def __init__(self, g, n, c, tau):
        self.g, self.n, self.c, self.tau = g, n, c, tau
        self.W, self.pos = wire_ops(g, n)
        self.s = []; self.e = []
        for x in range(n):
            k0 = next((k for k, i in enumerate(self.W[x]) if i >= c), len(self.W[x]))
            self.s.append(k0); self.e.append(k0 - 1)

    def snapshot(self): return (list(self.s), list(self.e))
    def restore(self, snap): self.s, self.e = list(snap[0]), list(snap[1])
    def empty(self, x): return self.s[x] > self.e[x]

    def inside(self, i, x):
        k = self.pos[(i, x)]
        return self.s[x] <= k <= self.e[x]

    def side(self, i, x):
        k = self.pos[(i, x)]
        return 'B' if k < self.s[x] else ('A' if k > self.e[x] else 'E')

    def cz_in(self):
        return [i for i, (t, q, p) in enumerate(self.g) if t == 'cz' and all(self.inside(i, x) for x in q)]

    def inconsistent(self):
        return [i for i, (t, q, p) in enumerate(self.g) if t == 'cz' and self.side(i, q[0]) != self.side(i, q[1])]

    def extend_to(self, i, x):
        k = self.pos[(i, x)]
        self.s[x] = min(self.s[x], k); self.e[x] = max(self.e[x], k)

    def next_out(self, x, direction):
        """Index of the first CZ on wire x just outside E (direction -1 = left, +1 = right)."""
        W = self.W[x]
        k = (self.s[x] - 1) if direction < 0 else (self.e[x] + 1)
        while 0 <= k < len(W):
            i = W[k]
            if self.g[i][0] == 'cz': return i
            k += direction
        return None


def confirmed(cut, l, r, tol):
    """A CZ twin pair is confirmed by u3 evidence: some u3 adjacent to l on wire x (inner or outer
    side) is a theta-twin of the mirror-position u3 adjacent to r on wire tau(x). R<->P look-alike
    pairs and chance LCS matches essentially never have exact-angle neighbours."""
    g, tau, W, pos = cut.g, cut.tau, cut.W, cut.pos
    for x in g[l][1]:
        y = tau[x]
        if y not in g[r][1]: continue
        kl, kr = pos[(l, x)], pos[(r, y)]
        for dl, dr in ((-1, +1), (+1, -1)):          # outer<->outer, inner<->inner
            a, b = kl + dl, kr + dr
            if 0 <= a < len(W[x]) and 0 <= b < len(W[y]):
                ga, gb = g[W[x][a]], g[W[y][b]]
                if ga[0] == 'u' and gb[0] == 'u' and theta_twin(ga[2], gb[2], tol):
                    return True
    return False


def evidence(cut, l, r, tol):
    """Number of exact theta-twin u3 neighbours (outer and inner, on each wire of the pair): 0..4."""
    g, tau, W, pos = cut.g, cut.tau, cut.W, cut.pos
    k = 0
    for x in g[l][1]:
        y = tau[x]
        if y not in g[r][1]: continue
        kl, kr = pos[(l, x)], pos[(r, y)]
        for dl, dr in ((-1, +1), (+1, -1)):
            a, b = kl + dl, kr + dr
            if 0 <= a < len(W[x]) and 0 <= b < len(W[y]):
                ga, gb = g[W[x][a]], g[W[y][b]]
                if ga[0] == 'u' and gb[0] == 'u' and theta_twin(ga[2], gb[2], tol):
                    k += 1
    return k


def _pair_addable(cut, l, r):
    g = cut.g
    ql, qr = g[l][1], g[r][1]
    return all(cut.next_out(x, -1) == l for x in ql) and all(cut.next_out(x, +1) == r for x in qr)

def _add_pair(cut, l, r):
    for x in cut.g[l][1]: cut.extend_to(l, x)
    for x in cut.g[r][1]: cut.extend_to(r, x)

def _grow_confirmed(cut, twins, conf, max_dist):
    c = cut.c; total = 0
    while True:
        added = 0
        for r, l in twins.items():
            if not conf[r] or abs(l - c) > max_dist or abs(r - c) > max_dist: continue
            if cut.inside(l, cut.g[l][1][0]): continue
            if _pair_addable(cut, l, r):
                _add_pair(cut, l, r); added += 1
        total += added
        if not added: return total

def _frontier(cut, twins, inv, conf, max_dist):
    """Tentative items: unconfirmed-but-addable twin pairs, and unmatched CZs that are the next
    CZ outside E on BOTH of their wires on the same side (absorbing one keeps E convex)."""
    g, c = cut.g, cut.c; items = []
    for r, l in twins.items():
        if conf[r] or abs(l - c) > max_dist or abs(r - c) > max_dist: continue
        if not cut.inside(l, g[l][1][0]) and _pair_addable(cut, l, r):
            items.append(('pair', l, r))
    seen = set()
    for x in range(cut.n):
        for d in (-1, 1):
            i = cut.next_out(x, d)
            if i is None or i in seen or i in twins or i in inv or abs(i - c) > max_dist: continue
            if all(cut.next_out(y, d) == i for y in g[i][1]):
                seen.add(i); items.append(('cz', i, d))
    items.sort(key=lambda it: abs((it[1] + it[2]) / 2 - c) if it[0] == 'pair' else abs(it[1] - c))
    return items

def _apply_item(cut, it):
    if it[0] == 'pair': _add_pair(cut, it[1], it[2])
    else:
        for y in cut.g[it[1]][1]: cut.extend_to(it[1], y)


def block_cut(g, n, c, tau, twins=None, max_dist=None, log=print, window=None,
              require_confirm=None, tol=None, lookahead=None, **_ignored):
    """Grow E outward from the centre. Confirmed twin pairs are always added. When stuck, try
    tentative items (core / masking CZs, unconfirmed pairs) up to `lookahead` deep; keep them
    only if they unlock at least one more CONFIRMED pair, otherwise roll back. Beyond the block
    edge nothing confirmed is left to unlock, so growth cannot wander into R<->P look-alikes."""
    max_dist = EXC_MAX_DIST if max_dist is None else max_dist
    window = EXC_WINDOW if window is None else window
    require_confirm = EXC_CONFIRM if require_confirm is None else require_confirm
    tol = EXC_TWIN_TOL if tol is None else tol
    lookahead = EXC_LOOKAHEAD if lookahead is None else lookahead
    lo, hi = (max(0, c - window), min(len(g), c + window)) if window else (0, len(g))
    twins = cz_twins(g, n, c, tau, lo, hi) if twins is None else twins
    inv = {l: r for r, l in twins.items()}
    cut = Cut(g, n, c, tau)
    conf = {r: (confirmed(cut, l, r, tol) if require_confirm else True) for r, l in twins.items()}
    log(f"  twins: {len(twins)} doubly-consistent CZ pairs in window [{lo},{hi}), {sum(conf.values())} u3-confirmed")
    ntent = nconf = 0

    def try_unlock(depth):
        """Depth-limited search for a sequence of tentative items that unlocks a confirmed pair."""
        for it in _frontier(cut, twins, inv, conf, max_dist):
            snap = cut.snapshot()
            _apply_item(cut, it)
            k = _grow_confirmed(cut, twins, conf, max_dist)
            if k > 0:
                return 1 + 0 * k
            if depth > 1 and try_unlock(depth - 1):
                return 1
            cut.restore(snap)
        return 0

    while True:
        nconf += _grow_confirmed(cut, twins, conf, max_dist)
        if not lookahead or not try_unlock(lookahead):
            break
        ntent += 1
    log(f"  growth: {nconf} confirmed pairs added directly, {ntent} tentative unlocks kept")
    return cut, twins



def run_pairs(g, n, c, tau, twins, lo=0, hi=None, p_in=0.85, p_out=0.12):
    """Twin pairs that lie inside the dense CZ-skeleton RUN on all four of their wire-sides.

    MEASURED on d3_s1 (clean/skel3.py): going outward from the centre every wire shows ~12 twinned CZs per side and
    then a SHARP end (the base's own gates; chance R<->P look-alikes beyond it are isolated). A maximum-likelihood
    change point per wire-side (twin density p_in inside, p_out outside) marks the run; a pair inside the run on
    both wires of both members is a real mirror pair with overwhelming odds EVEN WITH ZERO angle evidence
    (structural masking removes the angle twins, not the skeleton)."""
    hi = len(g) if hi is None else hi
    Ls = set(twins.values()); Rs = set(twins.keys())
    a, b = math.log(p_in / p_out), math.log((1 - p_in) / (1 - p_out))
    inrun = {}
    for x in range(n):
        for side, rng, S in ((-1, range(c - 1, lo - 1, -1), Ls), (1, range(c, hi), Rs)):
            seq = [i for i in rng if g[i][0] == 'cz' and x in g[i][1]]
            first = next((k for k, i in enumerate(seq) if i in S), None)
            if first is None:
                inrun[(x, side)] = set(); continue
            best, acc, bk = 0.0, 0.0, first
            for k in range(first, len(seq)):
                acc += a if seq[k] in S else b
                if acc > best: best, bk = acc, k + 1
            inrun[(x, side)] = set(seq[:bk])
    return {(r, l) for r, l in twins.items()
            if all(l in inrun[(x, -1)] for x in g[l][1]) and all(r in inrun[(x, 1)] for x in g[r][1])}



def extend_cut(cut, twins, tol=None, min_ev=None, log=print):
    """Grow a finished (convex, pair-closed) cut outward over the block's EDGE layers.

    The envelope starts from pairs with >= EXC_MIN_EVIDENCE (3 of 4) angle-twin neighbours; the outermost one or two
    CZ layers of a block can never reach that (their outer u3 neighbours are merged with the base's gates), so ~60
    CZ per block edge stayed behind as a thin mirror shell (MEASURED d3_s1 at tol 1e-3: 1553 kept gates vs 1065 of
    true base). A pair is added here only if it is (a) inside the skeleton run on all four wire-sides, (b) carries
    >= EXC_EXTEND_MIN_EV (default 1) angle evidence, and (c) ADDABLE: the very next CZ outside the cut on both wires
    of both members -- so the cut stays convex and pair-closed by construction. Returns the number of pairs added."""
    tol = EXC_TWIN_TOL if tol is None else tol
    min_ev = int(os.environ.get('EXC_EXTEND_MIN_EV', '1')) if min_ev is None else min_ev
    if os.environ.get('EXC_EXTEND', '1') != '1':
        return 0
    g, n, c, tau = cut.g, cut.n, cut.c, cut.tau
    runp = run_pairs(g, n, c, tau, twins)
    probe = Cut(g, n, c, tau)
    cand = [(r, l) for (r, l) in runp if evidence(probe, l, r, tol) >= min_ev]
    added = 0
    while True:
        k = 0
        for r, l in cand:
            if cut.inside(l, g[l][1][0]) or cut.inside(r, g[r][1][0]):
                continue
            if _pair_addable(cut, l, r):
                snap = cut.snapshot()
                _add_pair(cut, l, r)
                if cut.inconsistent():
                    cut.restore(snap)
                    continue
                k += 1
        added += k
        if not k:
            break
    if added:
        log(f"  extend: +{added} edge pairs (skeleton run, evidence >= {min_ev}, addable) -> {len(cut.cz_in())} CZ")
    return added


def envelope_cut(g, n, c, tau, twins=None, log=print, window=None, tol=None, max_dist=None,
                 min_conf_depth=None, peel_mode=None, **_ignored):
    """Cut from the OUTSIDE: E = envelope of all u3-confirmed CZ twin pairs (so the SWAP core,
    wherever it sits, is inside). Then peel until E is convex and pair-closed.

    peel_mode='pairs' (default) removes whole TWIN PAIRS: both members of a pair always leave E
    together, so E stays a mirror core U_c . SWAP . U_c^dag by construction. The old 'wires' mode
    pulled a single wire's boundary past the offending CZ, which strands a twin's partner inside E
    and silently destroys the mirror -- measured by e70/e71: d2 with 0 peels gives F0=0.84, with 36
    asymmetric peels F0=1e-32, and d3 B1 with 24 asymmetric peels F0=0.06 plus an X-leak of 1.6."""
    window = EXC_WINDOW if window is None else window
    tol = EXC_TWIN_TOL if tol is None else tol
    max_dist = EXC_MAX_DIST if max_dist is None else max_dist
    peel_mode = os.environ.get('EXC_PEEL_MODE', 'close') if peel_mode is None else peel_mode
    lo, hi = (max(0, c - window), min(len(g), c + window)) if window else (0, len(g))
    twins = cz_twins(g, n, c, tau, lo, hi) if twins is None else twins
    probe = Cut(g, n, c, tau)
    min_ev = int(os.environ.get('EXC_MIN_EVIDENCE', '1')) if min_conf_depth is None else min_conf_depth
    good = [(r, l) for r, l in twins.items()
            if evidence(probe, l, r, tol) >= min_ev and abs(l - c) <= max_dist and abs(r - c) <= max_dist]
    log(f"  twins: {len(twins)} doubly-consistent CZ pairs in [{lo},{hi}), {len(good)} with >= {min_ev} twin neighbours within {max_dist}")

    def build(inc):
        cut = Cut(g, n, c, tau)
        for r, l in inc:
            for x in g[l][1]: cut.extend_to(l, x)
            for x in g[r][1]: cut.extend_to(r, x)
        return cut

    if peel_mode == 'depth':
        # Symmetric by construction:each logical wire keeps the innermost d_x confirmed pairs, and a
        # pair enters E only if BOTH of its logical wires keep it -- so both members always enter or
        # leave together and E stays a mirror core. Convexity is reached by REDUCING depths only.
        import collections as _c
        bywire = _c.defaultdict(list)
        for (r, l) in sorted(good, key=lambda rl: max(abs(rl[0] - c), abs(rl[1] - c))):
            for x in g[l][1]: bywire[x].append((r, l))
        depth = {x: len(v) for x, v in bywire.items()}
        idx = {(x, p): i for x, v in bywire.items() for i, p in enumerate(v)}
        for _ in range(20000):
            inc = [p for p in good if all(idx.get((x, p), 10 ** 9) < depth.get(x, 0) for x in g[p[1]][1])]
            cut = build(inc)
            E = set(cut.cz_in())
            bad = cut.inconsistent()
            orph = [(r, l) for r, l in twins.items() if (r in E) != (l in E)]
            if not bad and not orph:
                log(f"  envelope[depth]: {len(E)} CZ from {len(inc)}/{len(good)} pairs "
                    f"(depths {min(depth.values())}-{max(depth.values())})")
                return cut, twins
            touched = set()
            for i in set(bad) | {r if r in E else l for r, l in orph}:
                for x in g[i][1]:
                    touched.add(x if x in depth else tau[x])
            if not touched or all(depth.get(x, 0) == 0 for x in touched):
                break
            for x in touched:
                if depth.get(x, 0) > 0: depth[x] -= 1
        log(f"  envelope[depth]: FAILED to reach convexity (depths exhausted)")
        return build([]), twins

    if peel_mode == 'hybrid':
        # Extend to swallow a straddling masking CZ only while that keeps the boundary INSIDE the
        # confirmed-twin envelope on that wire; otherwise shrink past it (and drop its twin too, so
        # E stays pair-symmetric). Pure 'close' extends unconditionally, which on d3_s2's B1 pulled
        # a non-block gate into E and collapsed it (measured F0 = 0.000, X-leak 7e4).
        cut = build(good)
        lim_s = list(cut.s); lim_e = list(cut.e)          # confirmed envelope = the extension limit
        inv = {l: r for r, l in twins.items()}
        grown = shrunk = 0
        for _ in range(10000):
            E = set(cut.cz_in())
            bad = cut.inconsistent()
            orph = [(r, l) for r, l in twins.items() if (r in E) != (l in E)]
            if not bad and not orph: break
            for i in set(bad) | {r if r in E else l for r, l in orph}:
                sides = [cut.side(i, y) for y in g[i][1]]
                if 'E' not in sides:
                    # before E on one wire, after E on the other: only extending can fix this
                    for y in g[i][1]: cut.extend_to(i, y)
                    grown += 1
                    continue
                for x in g[i][1]:
                    k = cut.pos[(i, x)]
                    if cut.inside(i, x): continue
                    if (k >= lim_s[x] and k <= lim_e[x]):  # inside the confirmed envelope -> extend
                        cut.extend_to(i, x); grown += 1
                    else:                                   # would leave it -> shrink the other wire(s)
                        for y in g[i][1]:
                            if not cut.inside(i, y): continue
                            ky = cut.pos[(i, y)]
                            if i < c: cut.s[y] = ky + 1
                            else: cut.e[y] = ky - 1
                            shrunk += 1
                        tw_partner = inv.get(i) or twins.get(i)
                        if tw_partner is not None:          # keep pair symmetry
                            for y in g[tw_partner][1]:
                                if not cut.inside(tw_partner, y): continue
                                ky = cut.pos[(tw_partner, y)]
                                if tw_partner < c: cut.s[y] = ky + 1
                                else: cut.e[y] = ky - 1
        else:
            raise RuntimeError('hybrid peel did not converge')
        log(f"  envelope[hybrid]: {len(cut.cz_in())} CZ ({grown} extensions, {shrunk} shrink steps)")
        return cut, twins

    if peel_mode == 'close':
        # EXTEND to consistency: an inconsistent CZ is an unmatched masking CZ straddling the
        # boundary, so pull it fully INTO E (patches must be excised whole) instead of shrinking
        # past it, and pull in any twin pair that ends up half-included. Only ever grows, so the
        # mirror core stays symmetric; growth is bounded by max_dist.
        cut = build(good); grown = 0
        # EXC_ORPHAN_MIN_EV (default 1; 0 = the pre-2026-09-18 behaviour, MEASURED identical on d3_s1/s2 + 6 v2 circuits): pair-closure only for twin pairs with at
        # least this much u3 evidence. MEASURED 2026-09-18 (scratchpad discov/s5_overext): with 0, an
        # evidence-0 LCS coincidence (2856,4167) with one member inside E cascaded 268 base CZs into the
        # cut while the audit still said PASS; real d3_s1 block 3120 carries such near-miss pairs too
        # (e.g. (2802,3883)) that merely did not fire. 1 keeps closure for every pair that could be real.
        orphan_min_ev = int(os.environ.get('EXC_ORPHAN_MIN_EV', '1'))
        ev_of = {r: evidence(probe, l, r, tol) for r, l in twins.items()} if orphan_min_ev > 0 else None
        # EXC_ORPHAN_RUN=1 (default): an evidence-0 orphan that sits INSIDE the skeleton run is a real mirror pair
        # whose angle twins were masked away -- leaving it half-included makes E asymmetric. MEASURED on h7
        # (ground-truth tags): 3 of its 4 audit orphans were exactly that (both members in the block), the 4th
        # a chance base<->block pair, which the run rule rejects.
        runp = run_pairs(g, n, c, tau, twins, lo, hi) if os.environ.get('EXC_ORPHAN_RUN', '1') == '1' else set()
        for _ in range(10000):
            E = set(cut.cz_in())
            bad = cut.inconsistent()
            orph = [(r, l) for r, l in twins.items() if (r in E) != (l in E)
                    and (ev_of is None or ev_of[r] >= orphan_min_ev or (r, l) in runp)]
            if not bad and not orph:
                break
            for i in bad:
                for x in g[i][1]: cut.extend_to(i, x)
                grown += 1
            for r, l in orph:
                m = l if r in E else r
                for x in g[m][1]: cut.extend_to(m, x)
                grown += 1
        else:
            raise RuntimeError('close-peel did not converge')
        span = cut.cz_in()
        log(f"  envelope[close]: {len(span)} CZ ({grown} extensions to reach convexity + pair closure)")
        extend_cut(cut, twins, tol=tol, log=log)
        return cut, twins

    if peel_mode != 'pairs':                       # legacy asymmetric peeling (kept for A/B only)
        cut = build(good); peeled = 0
        for _ in range(100000):
            E = set(cut.cz_in()); bad = cut.inconsistent()
            orph = [(r, l) for r, l in twins.items() if (r in E) != (l in E)]
            if not bad and not orph: break
            for i in set(bad) | {r if r in E else l for r, l in orph}:
                for x in g[i][1]:
                    if not cut.inside(i, x): continue
                    k = cut.pos[(i, x)]
                    if i < c: cut.s[x] = k + 1
                    else: cut.e[x] = k - 1
                peeled += 1
        log(f"  envelope[wires]: {len(cut.cz_in())} CZ after {peeled} asymmetric peel steps")
        return cut, twins

    inc = list(good)
    removed = 0
    for _ in range(100000):
        cut = build(inc)
        E = set(cut.cz_in())
        bad = cut.inconsistent()
        orph = [(r, l) for r, l in twins.items() if (r in E) != (l in E)]
        if not bad and not orph:
            break
        # offending gates: each one is removed from E by dropping the outermost included pair
        # whose member reaches it on that wire and side.
        drop = set()
        for i in set(bad) | {r if r in E else l for r, l in orph}:
            for x in g[i][1]:
                if not cut.inside(i, x): continue
                k = cut.pos[(i, x)]
                # drop only the single OUTERMOST pair reaching this wire on the offending side
                cands = []
                for (r, l) in inc:
                    for m in (l, r):
                        if x in g[m][1]:
                            km = cut.pos[(m, x)]
                            if (i < c and km <= k) or (i >= c and km >= k):
                                cands.append((abs(km - cut.pos[(cut.W[x][min(max(cut.s[x], 0), len(cut.W[x]) - 1)], x)]), km, (r, l)))
                if cands:
                    drop.add(min(cands, key=lambda t: t[1])[2] if i < c else max(cands, key=lambda t: t[1])[2])
        if not drop:                                # nothing attributable: fall back to the widest pair
            if not inc: break
            drop = {max(inc, key=lambda rl: max(abs(rl[0] - c), abs(rl[1] - c)))}
        inc = [p for p in inc if p not in drop]
        removed += len(drop)
        if not inc:
            break
    cut = build(inc)
    log(f"  envelope[pairs]: {len(cut.cz_in())} CZ from {len(inc)}/{len(good)} confirmed pairs "
        f"({removed} pairs dropped to reach convexity)")
    return cut, twins


# ----------------------------------------------------------------------------- discovery (truth-free)
# theta values shared by MANY gates (Rz-only u3s, the SWAP cores' Hadamards, X frames): a twin at one of them is not
# evidence of a mirror pairing, it is a coincidence available to every wire.  MEASURED 2026-09-24 on 8 known-answer
# blocks whose tau was wrong at every discovery window (genhard h6/h7/h10, dense h32/h35): on every mis-paired wire the
# chosen partner had 0-1 GENERIC twins against 1-5 for the true partner, and all of its lead came from these values
# (the cores put two exact pi/2 u3s per wire on ONE side of the centre for BOTH wires of a pair, which favours every
# partner but the true one).  Generic-only counts give the correct tau on all 8 blocks at every window.
_SPECIAL_THETA = (0.0, math.pi / 2, math.pi)

def _is_special_theta(th, tol):
    f = _fold(th)
    return any(abs(f - s) < tol for s in _SPECIAL_THETA)

def _window(g, c, half):
    """Index range of the scan window around c.  EXC_WIN_UNIT=u3 (default): the `half * EXC_U3_PER_GATE` nearest u3
    gates on each side (EXC_U3_PER_GATE 0.67 = the public samples' u3 share, so the window equals the historical
    +-half GATES on them, and it no longer shrinks when CZ-only insertions dilute the circuit: MEASURED 2026-09-24
    on d3_s1 + 80 one-sided identity insertions per block, the second block's refined twin fraction fell 0.29 -> 0.249,
    under the 0.25 gate, and the loose pass rejected it too -> a certain wrong answer).  EXC_WIN_UNIT=gate: +-half gates."""
    if os.environ.get('EXC_WIN_UNIT', 'u3').strip().lower() != 'u3':
        return max(0, c - half), min(len(g), c + half)
    k = max(1, int(round(half * float(os.environ.get('EXC_U3_PER_GATE', '0.67')))))
    lo, m = c, 0
    while lo > 0 and m < k:
        lo -= 1
        if g[lo][0] == 'u': m += 1
    hi, m = c, 0
    while hi < len(g) and m < k:
        if g[hi][0] == 'u': m += 1
        hi += 1
    return lo, hi

def _twin_counts(g, n, c, half, tol, generic_only=False):
    """M[a,b] = number of u3 on wire a in the left half-window with an exact theta-twin on wire b in the right one
    (window: _window).  generic_only: ignore u3s whose theta is one of the special values (see _SPECIAL_THETA)."""
    left = collections.defaultdict(list); right = collections.defaultdict(list)
    lo, hi = _window(g, c, half)
    for i in range(lo, hi):
        t, q, p = g[i]
        if t != 'u': continue
        if generic_only and _is_special_theta(p[0], tol): continue
        (left if i < c else right)[q[0]].append(_fold(p[0]))
    M = np.zeros((n, n))
    R = {b: np.array(v) for b, v in right.items()}
    for a, va in left.items():
        for b, vb in R.items():
            for th in va:
                if np.any(np.abs(vb - th) < tol) or np.any(np.abs(vb + th - math.pi) < tol):
                    M[a, b] += 1
    return M

def _assign_involution(S):
    """Maximum-weight perfect matching of the symmetric score S (Hungarian; greedy symmetric fallback)."""
    from scipy.optimize import linear_sum_assignment
    n = S.shape[0]
    S = S.copy(); np.fill_diagonal(S, -1e9)
    r, col = linear_sum_assignment(-S)
    tau = list(range(n))
    for a, b in zip(r, col): tau[a] = b
    if not all(tau[tau[a]] == a for a in range(n)):      # greedy symmetric matching fallback
        order = sorted(((S[a, b], a, b) for a in range(n) for b in range(a + 1, n)), reverse=True)
        tau = list(range(n)); used = set()
        for s_, a, b in order:
            if a in used or b in used: continue
            tau[a] = b; tau[b] = a; used |= {a, b}
    return tau

def _core_counts(g, n, c, R):
    """K[a,b] = number of CZ(a,b) within +-R gates of the centre.  The SWAP core of a mirror block is a cluster of 2-3
    CZs between the two wires of every tau-pair near the centre, so K is a THIRD source of tau evidence that does not
    need a single twin u3 -- the only evidence left for a pair whose wires carry 1-2 layers of gates.  MEASURED
    2026-09-25 (t_coresig, R=150): genadv h68/h69 23/24 and 24/24 pairs have >= 1 core CZ (mostly 2-3), background
    38-45 non-pair CZs spread over many wire pairs; real d3_s1 blocks 20/24 and 18/24 pairs (mostly 2 CZs, offsets up to
    +-150: the real cores are spread), background 63."""
    K = np.zeros((n, n))
    for i in range(max(0, c - R), min(len(g), c + R)):
        if g[i][0] == 'cz':
            a, b = g[i][1]
            K[a, b] += 1; K[b, a] += 1
    return K

def _cz_consistency(g, n, c, half, tau):
    """C[a,b] = number of left CZ(a,y) in [c-half,c) for which a right CZ(b, tau[y]) exists in [c,c+half): the
    mirror image of U's CZ(a,y) is U^dag's CZ(tau a, tau y), so under the right tau every mirrored CZ votes for its
    wire's partner.  Independent of the u3 angles (a second source of evidence for tau)."""
    Rset = collections.defaultdict(set)
    for i in range(c, min(len(g), c + half)):
        if g[i][0] == 'cz':
            x, y = g[i][1]; Rset[x].add(y); Rset[y].add(x)
    C = np.zeros((n, n))
    for i in range(max(0, c - half), c):
        if g[i][0] != 'cz': continue
        x, y = g[i][1]
        for a, y_ in ((x, y), (y, x)):
            ty = tau[y_]
            for b in range(n):
                if b != a and ty in Rset[b]: C[a, b] += 1
    return C

def refine_tau_cz(g, n, c, half, tau0, S0, lam=1.0, iters=6):
    """Iterate the assignment on S0 + lam * (C + C^T) with C the CZ consistency under the current tau (label
    propagation from the correctly paired majority of the wires).  Returns the fixed point (or the last iterate)."""
    tau = list(tau0)
    for _ in range(int(iters)):
        C = _cz_consistency(g, n, c, half, tau)
        tau2 = _assign_involution(S0 + float(lam) * (C + C.T))
        if tau2 == tau: break
        tau = tau2
    return tau

def discover_tau(g, n, c, half=400, tol=1e-6, mode='all'):
    """Involution tau maximising exact-theta twins across c (Hungarian on the symmetrised counts).
    Truth-free: uses only gate angles. Returns (tau, score, frac).
    mode: 'all' (every twin counts: the scan's and the historical default), 'generic' (special-theta u3s ignored,
    see _twin_counts) or 'gencz' (generic counts refined by the CZ consistency, refine_tau_cz).  score and frac are
    always reported on the ALL-twin counts so the centre scan's numbers keep their meaning."""
    M = _twin_counts(g, n, c, half, tol)
    if mode == 'all':
        tau = _assign_involution(M + M.T)
    else:
        Mg = _twin_counts(g, n, c, half, tol, generic_only=True)
        S0 = Mg + Mg.T
        if mode == 'gencore':
            # 'gencore' = gencz + the SWAP-core CZ counts (EXC_TAU_CORE_MU x _core_counts within EXC_TAU_CORE_R gates):
            # pairs whose wires carry almost no u3 twins (MEASURED genadv h68 2605 / h69 3250: row max 0-1, 6-18 gates
            # in the block) are still pinned by their core.  A weight of 2 per core CZ cannot outvote a well-evidenced
            # wire (CZ consistency 10-17) but decides an otherwise unevidenced one.
            S0 = S0 + float(os.environ.get('EXC_TAU_CORE_MU', '2.0')) * _core_counts(g, n, c, int(os.environ.get('EXC_TAU_CORE_R', '200')))
        tau = _assign_involution(S0)
        if mode in ('gencz', 'gencore'):
            tau = refine_tau_cz(g, n, c, half, tau, S0, lam=float(os.environ.get('EXC_TAU_CZ_LAMBDA', '1.0')),
                                iters=int(os.environ.get('EXC_TAU_CZ_ITERS', '6')))
    score = sum(M[a, tau[a]] for a in range(n))
    lo, _hi = _window(g, c, half)
    nleft = sum(1 for i in range(lo, c) if g[i][0] == 'u')
    return tau, float(score), float(score) / max(nleft, 1)

def discover_centres(g, n, step=None, half=None, tol=1e-6, min_frac=None, log=print):
    """Scan candidate centres; a mirror block shows as a sharp maximum of the best-tau twin
    fraction. Returns [(centre, tau, frac)] for separated peaks above min_frac."""
    step = int(os.environ.get('EXC_SCAN_STEP', '50')) if step is None else step
    half = int(os.environ.get('EXC_SCAN_HALF', '300')) if half is None else half
    min_frac = float(os.environ.get('EXC_MIN_FRAC', '0.25')) if min_frac is None else min_frac
    # Pre-gate for picking candidate peaks off the COARSE grid. It is deliberately far below
    # min_frac: the coarse sample is not the block's real twin fraction, only whatever the grid
    # happened to land on. MEASURED: sweeping the grid phase 0..40 gates moves d3_s1's second
    # block between 0.256 and 0.290, and the shipped phase is the WORST of the six tested -- so
    # against a 0.25 gate that real block cleared by 2.4% by luck alone. Gate on the REFINED
    # fraction instead (below), which is phase-independent.
    pre = min(min_frac, float(os.environ.get('EXC_PRE_FRAC', '0.15')))
    prof = []
    for c in range(half, len(g) - half, step):
        tau, sc, fr = discover_tau(g, n, c, half, tol)
        prof.append((c, fr))
    log("  centre scan: " + ' '.join(f"{c}:{fr:.2f}" for c, fr in prof if fr > 0.1))

    def _peaks(gate):
        pk = []
        for k, (c, fr) in enumerate(prof):
            if fr < gate: continue
            if all(fr >= prof[j][1] for j in range(max(0, k - 4), min(len(prof), k + 5))):
                if not pk or c - pk[-1][0] > 4 * step:
                    pk.append((c, fr))
        return pk

    def _refine(c0):
        best = None
        for c in range(c0 - step, c0 + step + 1, max(1, step // 10)):
            tau, sc, fr = discover_tau(g, n, c, half, tol)
            if best is None or fr > best[2]: best = (c, tau, fr)
        return best

    cands = [b for b in (_refine(c0) for c0, _ in _peaks(pre)) if b]
    out = [b for b in cands if b[2] >= min_frac]

    # No relaxed second pass, deliberately. Refining BEFORE the gate already roughly doubles the
    # separation -- measured across all six public samples, refined real blocks land at
    # 0.2879-0.3400 and block-free d1 peaks at 0.1210/0.2250, a 28% gap where the coarse-grid
    # numbers left only 12% (0.2563 vs 0.2250). Any floor loose enough to rescue a hypothetical
    # sub-0.25 block also admits d1-grade noise, and a spurious block is not free: it would draw D
    # measurement rungs and MAX_FRAC of the wall away from the stage that actually solves d1/d2.
    # cap 4 -> 6 (2026-09-24): a five-block known-answer circuit (genadv h51) is a certain failure at 4, while the refined-
    # fraction gate (>= 0.25; spurious peaks refine to <= 0.135 on 26 known-answer circuits, 0.225 on d1_s1's real
    # swept block) is what keeps spurious blocks out, not the cap.
    return out[:int(os.environ.get('EXC_MAX_BLOCKS', '6'))]


def plan_excision(text, which='all', log=print):
    """Truth-free end to end: discover block centres + tau, cut each block from the outside,
    audit, reduce. which: 'all' (excise every block) or 'last' (excise only the last block and
    split at the first -- d2's shape). Returns dict(reduced_qasm, Pinv, center, blocks) or None."""
    n, g = parse_qasm(text)
    found = discover_centres(g, n, log=log)
    if not found:
        log('  [plan] no mirror blocks found'); return None
    half_tau = int(os.environ.get('EXC_TAU_HALF', '600'))
    blocks = []
    for c, _, fr in found:
        tau, sc, fr2 = discover_tau(g, n, c, half_tau)
        blocks.append((c, tau, fr2))
    cs = [b[0] for b in blocks]
    use = blocks if which == 'all' else blocks[-1:]
    cuts = []
    for c, tau, fr in use:
        k = cs.index(c)
        gaps = [cs[k] - cs[k - 1]] if k > 0 else []
        gaps += [cs[k + 1] - cs[k]] if k + 1 < len(cs) else []
        win = min(gaps) // 2 if gaps else 0
        cut, tw = envelope_cut(g, n, c, tau, window=win, log=log)
        a = audit(cut, tw, log=log)
        if a['orphans'] or a['inconsistent'] or a['untouched']:
            log(f'  [plan] block at {c} fails audit -> not excised'); continue
        cuts.append(cut)
    if not cuts:
        return None
    red, Pinv, dropped = reduce_multi(g, n, cuts)
    W, pos = wire_ops(g, n)
    exc = set()
    for ct in cuts:
        for x in range(n):
            for kk in range(ct.s[x], ct.e[x] + 1): exc.add(W[x][kk])
    c_split = (cs[0] if which != 'all' else (cuts[0].c + cuts[-1].c) // 2) if len(cs) > 1 else cs[0]
    center = sum(1 for i in range(c_split) if i not in exc) / len(red)
    log(f"  [plan] blocks {[(b[0], round(b[2], 2)) for b in blocks]}, excised {len(cuts)}, reduced "
        f"{len(red)} gates / {sum(1 for t, _, _ in red if t == 'cz')} CZ, center {center:.4f}")
    return dict(reduced_qasm=to_qasm(n, red), Pinv=Pinv, center=center, blocks=blocks, n=n)


def audit(cut, twins, log=print, tol=None):
    g, n, tau = cut.g, cut.n, cut.tau
    E = set(cut.cz_in())
    orph = [(r, l) for r, l in twins.items() if (r in E) != (l in E)]
    # EXC_SOFT_AUDIT=1 (default): only orphans that could be REAL count -- angle evidence >= 1 or inside the
    # skeleton run. What is left over are evidence-0 LCS coincidences outside the run (chance R<->P look-alikes),
    # which used to veto an otherwise exact cut (h7: not excised at all).
    if os.environ.get('EXC_SOFT_AUDIT', '1') == '1' and orph:
        tol_ = EXC_TWIN_TOL if tol is None else tol
        probe = Cut(g, n, cut.c, tau)
        runp = run_pairs(g, n, cut.c, tau, twins)
        hard = [(r, l) for r, l in orph if evidence(probe, l, r, tol_) >= 1 or (r, l) in runp]
        if len(hard) != len(orph):
            log(f"  audit: {len(orph) - len(hard)} of {len(orph)} orphans are evidence-0 pairs outside the skeleton run -> ignored")
        orph = hard
    bad = cut.inconsistent()
    empty = [x for x in range(n) if cut.empty(x) and tau[x] != x]
    depth = [sum(1 for k in range(cut.s[x], cut.e[x] + 1) if g[cut.W[x][k]][0] == 'cz') for x in range(n)]
    pairs_in = sum(1 for r, l in twins.items() if r in E and l in E)
    span = (min(E), max(E)) if E else (None, None)
    log(f"  cut: {len(E)} CZ excised, span {span}, twin pairs inside {pairs_in}/{len(twins)}, "
        f"orphans {len(orph)}, inconsistent CZ {len(bad)}, untouched moved wires {len(empty)}, "
        f"per-wire CZ depth min/median/max {min(depth)}/{int(np.median(depth))}/{max(depth)}")
    return dict(cz=len(E), orphans=len(orph), inconsistent=len(bad), untouched=len(empty), pairs_in=pairs_in)


# ----------------------------------------------------------------------------- reduction
def reduce_multi(g, n, cuts, corrections=None):
    """cuts: list of Cut objects in time order (non-overlapping per wire). Returns (gates, Pinv)
    with Pinv[w] = reduced qubit that holds original output wire w.
    corrections: optional list (one per cut) of {wire: (theta, phi, lam)} 1q gates that stand in
    for the residual D of E = Pi_tau . D (measured by e71_cutfix.py); each is emitted on the
    wire's input side of E, right after its last kept gate before E."""
    if corrections:
        return _reduce_with_corrections(g, n, cuts, corrections)
    W, pos = wire_ops(g, n)
    region = {}
    for x in range(n):
        for k, i in enumerate(W[x]):
            r = 0; excised = False
            for b, ct in enumerate(cuts):
                if k > ct.e[x]: r = b + 1
                elif ct.s[x] <= k <= ct.e[x]: excised = True; r = b; break
            region[(i, x)] = (r, excised)
    # relabel map after b blocks: rel[b][v] = tau1[tau2[...tau_b[v]]]
    rel = [list(range(n))]
    for ct in cuts:
        prev = rel[-1]
        rel.append([prev[ct.tau[v]] for v in range(n)])
    out = []; dropped = 0
    for i, (t, q, p) in enumerate(g):
        info = [region[(i, x)] for x in q]
        if len(set(info)) != 1:
            raise RuntimeError(f'gate {i} {t}{q} straddles regions {info}: cut is not convex')
        r, excised = info[0]
        if excised: dropped += 1; continue
        out.append((t, tuple(rel[r][x] for x in q), p))
    return out, rel[-1], dropped

def u3_from_unitary(M, eps=1e-9):
    """(theta, phi, lam) with u3 == M up to a global phase, for ANY 2x2 unitary M.
    The previous inline conversion took phi and lam from the phases of the OFF-diagonal elements and fell
    back to 0 when they vanished -- so an exactly diagonal M = diag(1, e^{i a}) (or anything whose
    off-diagonals are < 1e-9) was emitted as the IDENTITY and its Z-rotation silently dropped
    (caught 2026-09-19 by a toy D = A.Phi unit test: overlap 0.94 instead of 1). Now the diagonal
    phase is always taken from M11/M00 when |M00| is usable."""
    c, s_ = abs(M[0, 0]), abs(M[1, 0])
    th = 2 * math.atan2(s_, c)
    if c > eps:
        la = float(np.angle(-M[0, 1]) - np.angle(M[0, 0])) if abs(M[0, 1]) > eps else 0.0
        ph = float(np.angle(M[1, 1]) - np.angle(M[0, 0])) - la
    else:                                                   # theta = pi: only phi - lam is defined
        ph = float(np.angle(M[1, 0]) - np.angle(-M[0, 1])); la = 0.0
    return float(th), ph, la


def _reduce_with_corrections(g, n, cuts, corrections):
    # ("tau",) record of a measured D: the wire map the excised region REALLY applies (see d_decode._residual_perm)
    if any((corr or {}).get(("tau",)) for corr in corrections):
        import copy as _copy
        cuts2 = []
        for ct, corr in zip(cuts, corrections):
            c2 = _copy.copy(ct)
            if (corr or {}).get(("tau",)):
                c2.tau = list(corr[("tau",)])
            cuts2.append(c2)
        cuts = cuts2
    red, Pinv, dropped = reduce_multi(g, n, cuts)          # validates convexity / regions
    W, pos = wire_ops(g, n)
    rel = [list(range(n))]
    for ct in cuts:
        prev = rel[-1]; rel.append([prev[ct.tau[v]] for v in range(n)])
    anchor = collections.defaultdict(list)                 # original gate index -> corrections after it
    lb = lambda ct, x: W[x][ct.s[x] - 1] if ct.s[x] > 0 else -1
    for b, (ct, corr) in enumerate(zip(cuts, corrections)):
        items = list((corr or {}).items())
        side = dict(items).get(("phase_side",), os.environ.get("V12_D_PHASE_SIDE", "out"))
        # phase_side == 'in': D = A . Phi, the two-body phases act FIRST. This anchored emitter appends per wire, so
        # emit the phase terms first and hang every 1q block of a phase-pair wire AFTER the last phase term that
        # touches it (sectioned.py is the clean implementation; this keeps the legacy path consistent with it).
        if side == "in":
            items = [kv for kv in items if isinstance(kv[0], tuple) and kv[0][0] == 'cz'] + \
                    [kv for kv in items if not (isinstance(kv[0], tuple) and kv[0][0] == 'cz')]
        late = {}                                            # wire -> anchor of the last phase term on it
        for key, prm in items:
            if key == ("resid",):
                # This legacy emitter cannot place the residual correction (it anchors per wire); dropping it
                # silently would emit an UNCORRECTED D and look fine. sectioned.py is the production emitter.
                raise RuntimeError("reduce_multi cannot emit a residual correction -- use sectioned.sectioned()")
            if key in (("phase_side",), ("tau",), ("cz_sides",)):   # emission-order / corrected-wire-map records
                continue
            if isinstance(key, tuple) and key[0] == 'A':     # general 1q block of D as a u3
                _, qx = key
                M = np.array([complex(e[0], e[1]) for e in prm]).reshape(2, 2)
                at_ = max(lb(ct, qx), late.get(qx, -1)) if side == "in" else lb(ct, qx)
                anchor[at_].append(('u', (rel[b][qx],), u3_from_unitary(M)))
            elif isinstance(key, tuple):                    # 2-body diagonal term of D
                _, qa, qb = key
                A, B = rel[b][qa], rel[b][qb]
                at = max(lb(ct, qa), lb(ct, qb))
                late[qa] = max(late.get(qa, -1), at); late[qb] = max(late.get(qb, -1), at)
                if prm is None or abs(abs(float(prm)) - math.pi) < 1e-9:
                    anchor[at].append(('cz', (A, B), None))
                else:                                       # controlled-phase cp(beta) in u/cz
                    beta = float(prm); H = (math.pi / 2, 0.0, math.pi)
                    anchor[at].extend([('u', (A,), (0.0, 0.0, beta / 2)), ('u', (B,), (0.0, 0.0, beta / 2)),
                                       ('u', (B,), H), ('cz', (A, B), None), ('u', (B,), H),
                                       ('u', (B,), (0.0, 0.0, -beta / 2)),
                                       ('u', (B,), H), ('cz', (A, B), None), ('u', (B,), H)])
            else:
                anchor[lb(ct, key)].append(('u', (rel[b][key],), tuple(prm)))
    excised = set()
    for ct in cuts:
        for x in range(n):
            for k in range(ct.s[x], ct.e[x] + 1): excised.add(W[x][k])
    out = list(anchor.get(-1, []))
    ri = 0
    for i in range(len(g)):
        if i not in excised:
            out.append(red[ri]); ri += 1
        out.extend(anchor.get(i, []))
    assert ri == len(red), (ri, len(red))
    return out, Pinv, dropped


def map_truth_to_reduced(truth, Pinv):
    """truth[w] = bit on original wire w  ->  string indexed by reduced qubit."""
    red = ['0'] * len(truth)
    for w, b in enumerate(truth): red[Pinv[w]] = b
    return ''.join(red)

def map_bits_to_original(bits_red, Pinv):
    return ''.join(bits_red[Pinv[w]] for w in range(len(bits_red)))


# ----------------------------------------------------------------------------- clifford check
def _clifford_table():
    H = np.array([[1, 1], [1, -1]]) / math.sqrt(2); S = np.diag([1, 1j])
    def key(M):
        k = np.argmax(np.abs(M.ravel()) > 1e-9); ph = M.ravel()[k] / abs(M.ravel()[k])
        return tuple(np.round((M / ph).ravel(), 6))
    T = {key(np.eye(2)): (np.eye(2), '')}; fr = [(np.eye(2), '')]
    while fr:
        nf = []
        for M, w in fr:
            for G, gn in ((H, 'h'), (S, 's')):
                N = G @ M; k = key(N)
                if k not in T: T[k] = (N, w + gn); nf.append((N, w + gn))
        fr = nf
    return list(T.values())
_CL = None
def _u3(t, p, l):
    return np.array([[math.cos(t / 2), -np.exp(1j * l) * math.sin(t / 2)],
                     [np.exp(1j * p) * math.sin(t / 2), np.exp(1j * (p + l)) * math.cos(t / 2)]])

def clifford_check(cut, log=print):
    """Round every u3 inside E to its nearest Clifford and test whether the resulting Clifford
    maps Z_i -> +Z_tau(i) and X_i -> +X_tau(i). Exact twins round to inverse Cliffords, so a true
    U_c SWAP U_c^dag core passes exactly; failures localise masking / mis-cut wires."""
    global _CL
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford
    if _CL is None:
        _CL = _clifford_table()
    CM = np.array([m for m, _ in _CL])
    g, n, tau = cut.g, cut.n, cut.tau
    qc = QuantumCircuit(n); dist = []
    for i, (t, q, p) in enumerate(g):
        if not all(cut.inside(i, x) for x in q): continue
        if t == 'cz': qc.cz(*q)
        else:
            M = _u3(*p); f = np.abs(np.einsum('kij,ij->k', CM.conj(), M)) / 2
            k = int(np.argmax(f)); dist.append(math.sqrt(max(0, 1 - f[k] ** 2)))
            for ch in _CL[k][1]: getattr(qc, ch)(q[0])
    T = Clifford(qc).tableau.astype(int)
    zok = xok = zneg = xneg = 0; badw = []
    for i in range(n):
        x, z, sgn = T[n + i, :n], T[n + i, n:2 * n], T[n + i, 2 * n]
        zg = x.sum() == 0 and z.sum() == 1 and z[tau[i]] == 1
        x2, z2, sgn2 = T[i, :n], T[i, n:2 * n], T[i, 2 * n]
        xg = x2.sum() == 1 and z2.sum() == 0 and x2[tau[i]] == 1
        zok += zg; xok += xg; zneg += zg and sgn; xneg += xg and sgn2
        if not (zg and xg): badw.append(i)
    log(f"  clifford-rounded E: Z_i->Z_tau(i) {zok}/{n} (neg {zneg}), X_i->X_tau(i) {xok}/{n} (neg {xneg}); "
        f"rounding dist mean {np.mean(dist) if dist else 0:.3f}; failing wires {badw[:16]}{'...' if len(badw) > 16 else ''}")
    return zok, xok, zneg, xneg, badw


# ----------------------------------------------------------------------------- CLI
if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    n, g = parse_qasm(open(src).read())
    specs = [(int(s.split(':', 1)[0]), s.split(':', 1)[1]) for s in sys.argv[3].split(';')]
    cuts = []
    for c, ts in specs:
        tau = parse_tau(n, ts)
        print(f"block centre {c}:")
        cut, tw = block_cut(g, n, c, tau)
        a = audit(cut, tw)
        clifford_check(cut)
        if a['orphans'] or a['inconsistent'] or a['untouched']:
            print("  !! cut fails audit -- refusing to write reduced circuit"); sys.exit(2)
        cuts.append(cut)
    red, Pinv, dropped = reduce_multi(g, n, cuts)
    open(dst, 'w').write(to_qasm(n, red))
    open(dst + '.map', 'w').write(' '.join(map(str, Pinv)) + '\n')
    ncz = sum(1 for t, _, _ in red if t == 'cz')
    print(f"reduced: {len(red)} gates ({ncz} CZ), dropped {dropped}; Pinv -> {dst}.map")
