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

"""Reduced circuit in SECTIONED order, with exact junction positions.

reduce_multi keeps the file order, in which gates from before a cut, between cuts and after the last
cut are interleaved. The operator-level readout needs to start absorbing at a junction (MEASURED on
reduced d3_s1: starting at the list midpoint instead of the block-1 junction cost 10^-2.4 of operator
norm before the collapse began), so emit the same gates as
    [region 0] [D of cut 1] [region 1] [D of cut 2] ... [region B]
Every cut is convex and pair-closed, so on each wire all region-b gates precede cut b: this is a valid
topological order of the same circuit, and junction b is simply the length of everything before it.
"""
import math

import numpy as np

import excise as X


def _correction_gates(ct, corr, rel_b):
    """Gate list (reduced labels) for one cut's measured D -- same emission rules as
    excise._reduce_with_corrections, minus the anchoring."""
    out = []
    H = (math.pi / 2, 0.0, math.pi)
    # 1q blocks ('A', wire) first, then the two-body terms, each term's own gate sequence kept intact
    # (a controlled-phase is an ORDERED u/cz sequence; sorting gates by type would break it --
    # caught by the per-wire equivalence check against reduce_multi: 47/48 wires before this fix).
    items = list((corr or {}).items())
    # TIME ORDER of D = A . Phi vs Phi . A: measure_D records which side the two-body phases sit on
    # (("phase_side",) -> "in" = phases act on the INPUT bits = FIRST in time). Older D's carry no record
    # and keep the historical order (1q blocks first) unless V12_D_PHASE_SIDE overrides.
    import os as _os
    side = dict(items).get(("phase_side",), _os.environ.get("V12_D_PHASE_SIDE", "out"))
    sides = dict(items).get(("cz_sides",)) or {}
    resid = dict(items).get(("resid",)) or []
    items = [kv for kv in items if kv[0] not in (("phase_side",), ("tau",), ("cz_sides",), ("resid",))]
    a_items = [kv for kv in items if isinstance(kv[0], tuple) and kv[0][0] == 'A']
    other = [kv for kv in items if not (isinstance(kv[0], tuple) and kv[0][0] == 'A')]
    ordered = None
    if sides:
        # PER-TERM sides (d_decode): a two-body term is keyed, per endpoint, on the wire's input bit ("pre": it must
        # come BEFORE that wire's 1q block), its output bit ("post": after it) or either ("free"). Emit any order that
        # satisfies all of them (Kahn); a contradictory set falls back to the single global side.
        nodes = [("A", kv) for kv in a_items] + [("T", kv) for kv in other]
        a_node = {kv[0][1]: k for k, (kind, kv) in enumerate(nodes) if kind == "A"}
        succ = {k: set() for k in range(len(nodes))}
        indeg = {k: 0 for k in range(len(nodes))}
        for k, (kind, kv) in enumerate(nodes):
            if kind != "T" or not (isinstance(kv[0], tuple) and len(kv[0]) == 3):
                continue
            _, qa, qb = kv[0]
            sd = sides.get(f"{qa},{qb}") or sides.get(f"{qb},{qa}")
            if not sd:
                sd = ["pre", "pre"] if side == "in" else ["post", "post"]
            elif sides.get(f"{qa},{qb}") is None:
                sd = [sd[1], sd[0]]
            for w, s_w in ((qa, sd[0]), (qb, sd[1])):
                if w not in a_node or s_w == "free":
                    continue
                u, v = (k, a_node[w]) if s_w == "pre" else (a_node[w], k)
                if v not in succ[u]:
                    succ[u].add(v); indeg[v] += 1
        ready = [k for k in range(len(nodes)) if indeg[k] == 0]
        order = []
        while ready:
            k = ready.pop(0); order.append(k)
            for v in sorted(succ[k]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
        if len(order) == len(nodes):
            ordered = [nodes[k][1] for k in order]
    items = ordered if ordered is not None else ((other + a_items) if side == "in" else (a_items + other))
    for key, prm in items:
        if isinstance(key, tuple) and key[0] == 'A':
            _, qx = key
            M = np.array([complex(e[0], e[1]) for e in prm]).reshape(2, 2)
            out.append(('u', (rel_b[qx],), X.u3_from_unitary(M)))
        elif isinstance(key, tuple):
            _, qa, qb = key
            A, B = rel_b[qa], rel_b[qb]
            if prm is None or abs(abs(float(prm)) - math.pi) < 1e-9:
                out.append(('cz', (A, B), None))
            else:
                beta = float(prm)
                out.extend([('u', (A,), (0.0, 0.0, beta / 2)), ('u', (B,), (0.0, 0.0, beta / 2)),
                            ('u', (B,), H), ('cz', (A, B), None), ('u', (B,), H),
                            ('u', (B,), (0.0, 0.0, -beta / 2)),
                            ('u', (B,), H), ('cz', (A, B), None), ('u', (B,), H)])
        else:
            out.append(('u', (rel_b[key],), tuple(prm)))
    # D = P_sigma . R . D'  -- the residual correction acts AFTER the ansatz, and the permutation is
    # carried by the relabelling, so R's gates simply follow, in the same (reduced) labels.
    for t, q, p in resid:
        out.append((t, tuple(rel_b[w] for w in q), p if p is None else tuple(p)))
    return out


def sectioned(g, n, cuts, corrections=None):
    """Returns (sections, Pinv). sections[b] = gate list of region b followed by cut b's D."""
    W, _pos = X.wire_ops(g, n)
    kpos = {(i, x): k for x in range(n) for k, i in enumerate(W[x])}
    # a block's D may carry a CORRECTED wire map (("tau",) record: the discovered tau was wrong on a few wires, or a
    # pair's SWAP gadget lies outside the cut) -- the relabelling must follow what the excised region really does
    taus = []
    for b, ct in enumerate(cuts):
        fix = (corrections[b] or {}).get(("tau",)) if corrections else None
        taus.append(list(fix) if fix else list(ct.tau))
    rel = [list(range(n))]
    for t_ in taus:
        prev = rel[-1]
        rel.append([prev[t_[v]] for v in range(n)])
    sections = [[] for _ in range(len(cuts) + 1)]
    for i, (t, q, p) in enumerate(g):
        if any(ct.s[x] <= kpos[(i, x)] <= ct.e[x] for ct in cuts for x in q):
            continue                                            # excised
        rs = set()
        for x in q:
            r = 0
            for b, ct in enumerate(cuts):
                if kpos[(i, x)] > ct.e[x]:
                    r = b + 1
            rs.add(r)
        if len(rs) != 1:
            raise RuntimeError(f"gate {i} {t}{q} straddles regions {rs}: cut is not convex")
        r = rs.pop()
        sections[r].append((t, tuple(rel[r][x] for x in q), p))
    if corrections:
        for b, (ct, corr) in enumerate(zip(cuts, corrections)):
            sections[b].extend(_correction_gates(ct, corr, rel[b]))
    return sections, rel[-1]
