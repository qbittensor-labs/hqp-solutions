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

"""Decode a measured mirror-block operator D into gates:  D = (x)_w A_w . Phi,  Phi = exp(i sum phi_ab xi_a xi_b).

REFERENCE-FREE version (2026-09-19). The original decode read every 2x2 block against the all-zeros element
<0..0|D|0..0>. When some wire carries an X-type 1q block (a Pauli-X frame pushed through the CZs -- common in the
real blocks) that element is EXACTLY zero, every 2x2 block reads as the zero matrix, and a perfectly product-like
operator is rejected as "unitarity 0.00 / F0 0.000": MEASURED on d3_s1 block 1435 (628-CZ cut), absorbed cleanly to
bond 2-16 with 10^-0.10 of the norm lost, then thrown away by the decode. Here the reference is the dominant
output string of D|0..0> (a product state for a product-like D, read off its MPS marginals), and the two-body
phases come from a four-element ratio in which every 1q factor cancels.

decode(amp, n, tau, ref_out, log) -> (corr, info)      amp(xo_bits, xi_bits) -> complex, bits indexed by WIRE.
"""
import itertools
import math
import os

import numpy as np


def dominant_output(mps_tensors):
    """mps_tensors[j]: (Dl, 2, Dr) of D|0..0> in SITE order -> greedy-by-conditional-marginal bit per site."""
    n = len(mps_tensors)
    R = [None] * (n + 1)
    R[n] = np.ones((1, 1), complex)
    for j in range(n - 1, -1, -1):
        T = mps_tensors[j]
        R[j] = np.einsum("akb,bc,dkc->ad", T, R[j + 1], T.conj())
        s = np.abs(R[j]).max()
        if s > 0:
            R[j] = R[j] / s
    v = np.ones((1,), complex)
    bits = []
    for j in range(n):
        best = None
        for k in (0, 1):
            v2 = v @ mps_tensors[j][:, k, :]
            wgt = float(np.real(v2 @ R[j + 1] @ v2.conj()))
            if best is None or wgt > best[0]:
                best = (wgt, k, v2)
        bits.append(best[1])
        nv = np.linalg.norm(best[2])
        v = best[2] / nv if nv > 0 else best[2]
    return bits


def _residual_perm(amp, n, tau, ref):
    """sigma with  D = P_sigma . D'  (D' a product operator): where does an input excitation on wire a re-appear?

    D = Pi_tau^-1 . E assumes the excised region E moves wire a to tau(a). When the DISCOVERED tau is wrong on a few
    wires (MEASURED on the synthetic block 4560 of h10: tau paired 5-32 and 36-41, the block really swaps 5-36 and
    32-41 -- thin wires make the twin-count matching ambiguous) or a pair's SWAP gadget is not inside the cut, D keeps
    a bare permutation: <1_a|D|1_a> = 0 and the excitation shows up, at full strength, on another wire's output.
    Returns {a: b} for the wires that move (empty when D is permutation-free)."""
    xi0 = [0] * n
    base = abs(amp(ref, xi0))
    moved = {}
    for a in range(n):
        xo = list(ref); xo[a] = 1 - ref[a]
        xi = list(xi0); xi[a] = 1
        if abs(amp(xo, xi)) >= 0.2 * base:
            continue                                   # the excitation stays on wire a (possibly with an X-type block:
        xo = list(ref)                                 # then it is the UN-flipped output that carries it)
        if abs(amp(xo, xi)) >= 0.2 * base:
            continue
        best = None
        for b in range(n):
            if b == a:
                continue
            for flip in (1, 0):
                xo = list(ref)
                if flip:
                    xo[b] = 1 - ref[b]
                v = abs(amp(xo, xi))
                if best is None or v > best[0]:
                    best = (v, b)
        if best and best[0] > 0.5 * base:
            moved[a] = best[1]
    # COMPLETE the map (2026-09-20). A wire whose 2x2 is singular in the assumed frame gives an unreliable
    # "stays put" test -- its amplitudes are degenerate -- so only ONE direction of a mis-paired cycle is detected:
    # MEASURED on h6's block 3210, the loop finds 41 -> 30 and 21 -> 14 and then rejects the whole thing because the
    # sources {21,41} are not the destinations {14,30}. The missing edges are forced: if a moves to b and b is not
    # itself a source, then b must move to a (E is an involution composed with the assumed one). With that closure the
    # map becomes the permutation {41:30, 30:41, 21:14, 14:21}, i.e. the excised region really pairs (41,14) and
    # (21,30) rather than the discovered (30,14) and (41,21).
    if moved and sorted(moved) != sorted(moved.values()):
        dests = [b for b in moved.values() if b not in moved]
        if dests and len(set(moved.values())) == len(moved):
            back = {}
            for a_, b_ in moved.items():
                if b_ in dests:
                    back[b_] = a_
            cand = dict(moved); cand.update(back)
            if sorted(cand) == sorted(cand.values()) and len(set(cand.values())) == len(cand):
                moved = cand
    if moved and (sorted(moved) != sorted(moved.values())):
        return {}                                      # not a permutation of the affected wires: leave it to the gate
    return moved


def decode(amp, n, tau, ref_out, log=print, centre=None, max_pairs_phase=0.05):
    xi0 = [0] * n
    ref = list(ref_out)
    sigma = {}
    tau_fixed = None
    if os.environ.get("D_DECODE_UNSWAP", "1") == "1":
        try:
            sigma = _residual_perm(amp, n, tau, ref)
        except Exception as e:                                    # noqa: BLE001
            log(f"  decode: residual-permutation test failed ({type(e).__name__})")
        if sigma:
            amp0 = amp

            def amp(xo, xi, _a=amp0, _s=dict(sigma)):              # noqa: F811   D' = P_sigma^-1 . D
                xo2 = list(xo)
                for a_, b_ in _s.items():
                    xo2[b_] = xo[a_]                               # D' output wire a  <-  D output wire sigma(a)
                return _a(xo2, xi)
            ref2 = list(ref)
            for a_, b_ in sigma.items():
                ref2[a_] = ref[b_]
            ref = ref2
            tau_fixed = [tau[sigma.get(a_, a_)] for a_ in range(n)]      # E really moves a to tau(sigma(a))
            ok_inv = all(tau_fixed[tau_fixed[a_]] == a_ for a_ in range(n))
            log(f"  decode: D carries a residual permutation {sigma} -> the excised region's true wire map differs from the "
                f"discovered tau on these wires; corrected tau is {'an involution' if ok_inv else 'NOT an involution'}")
    base = amp(ref, xi0)
    # ---- 1q blocks against the reference (input reference is all zeros, so no phase correction is needed) ----
    Braw = {}
    for w in range(n):
        Mw = np.zeros((2, 2), complex)
        for o in (0, 1):
            xo = list(ref); xo[w] = o
            for i in (0, 1):
                xi = list(xi0); xi[w] = i
                Mw[o, i] = amp(xo, xi)
        Braw[w] = Mw
    Aw, sing = {}, {}
    for w in range(n):
        U_, s_, Vh_ = np.linalg.svd(Braw[w])
        sing[w] = (float(s_.max()), float(s_.min()))
        Aw[w] = U_ @ Vh_
    dom = {w: [int(np.argmax(np.abs(Aw[w][:, i]))) for i in (0, 1)] for w in range(n)}     # dominant output per input bit
    model0 = np.prod([Aw[w][ref[w], 0] for w in range(n)])
    scale = base / model0 if abs(model0) > 1e-12 else 1.0

    def model(xo, xi, phases):
        m = scale * np.prod([Aw[w][xo[w], xi[w]] for w in range(n)])
        return m * np.exp(1j * sum(p for (a_, b_, p) in phases if xi[a_] and xi[b_]))

    def el(setbits):
        """element with the given INPUT bits set to 1 (others 0) and every wire on its dominant output."""
        xi = list(xi0)
        for w in setbits:
            xi[w] = 1
        xo = [dom[w][xi[w]] for w in range(n)]
        return amp(xo, xi), xo, xi

    e0, _, _ = el(())
    e1 = {w: el((w,))[0] for w in range(n)}
    two = []
    for a_, b_ in itertools.combinations(range(n), 2):
        e11 = el((a_, b_))[0]
        num, den = e11 * e0, e1[a_] * e1[b_]
        if abs(num) > 1e-14 and abs(den) > 1e-14:
            ph = float(np.angle(num / den))
            if abs(ph) > max_pairs_phase:
                two.append((a_, b_, ph))
    rng = np.random.default_rng(0)
    worst3 = 0.0
    for _ in range(120):
        tri = sorted(int(v) for v in rng.choice(n, 3, replace=False))
        m, xo, xi = el(tri)
        pm = model(xo, xi, two)
        if abs(m) > 1e-12 and abs(pm) > 1e-12:
            worst3 = max(worst3, abs(float(np.angle(m / pm))))
    errs = []
    typ = abs(e0)
    for _ in range(60):
        xi = [int(rng.random() < 0.3) for _ in range(n)]
        xo = [dom[w][xi[w]] for w in range(n)]
        if rng.random() < 0.5:
            f = int(rng.integers(n)); xo[f] = 1 - xo[f]
        m = amp(xo, xi); pm = model(xo, xi, two)
        if max(abs(m), abs(pm)) > 1e-3 * typ:
            errs.append(abs(m - pm) / max(abs(m), abs(pm)))
    med_err = float(np.median(errs)) if errs else 1.0
    # which side do the two-body phases sit on (input bits = first in time, or output bits)?
    pair_wires = sorted({w for (x_, y_, _p) in two for w in (x_, y_)})
    hot = [w for w in pair_wires if max(abs(Aw[w][0, 1]), abs(Aw[w][1, 0])) > 0.1]
    e_in, e_out = [], []
    for w in hot:
        partners = [y_ if x_ == w else x_ for (x_, y_, _p) in two if w in (x_, y_)]
        for _rep in range(6):
            xi = [int(rng.random() < 0.3) for _ in range(n)]
            for pw in partners:
                xi[pw] = 1
            xo = [dom[v][xi[v]] for v in range(n)]
            xo[w] = 1 - xi[w] if abs(Aw[w][1 - xi[w], xi[w]]) > 0.1 else xo[w]      # an element that FLIPS the hot wire
            m = amp(xo, xi)
            bm = scale * np.prod([Aw[v][xo[v], xi[v]] for v in range(n)])
            pm_in = bm * np.exp(1j * sum(p_ for (x_, y_, p_) in two if xi[x_] and xi[y_]))
            pm_out = bm * np.exp(1j * sum(p_ for (x_, y_, p_) in two if xo[x_] and xo[y_]))
            den = max(abs(m), abs(bm), 1e-15)
            e_in.append(abs(m - pm_in) / den); e_out.append(abs(m - pm_out) / den)
    phase_side = "out" if (e_in and float(np.median(e_out)) < float(np.median(e_in))) else "in"
    # ---- PER-TERM side (D_DECODE_TERM_SIDES=0 disables) ----
    # MEASURED 2026-09-19 on d3_s2 block 3090: wire 44 carries a genuinely non-diagonal 1q block (|off-diagonal| 0.33)
    # AND two pi-terms -- the one with wire 35 acts on its INPUT bit (one-sided 'in' model: rel. error 6e-4, 'out':
    # 0.69), the one with wire 3 on its OUTPUT bit ('in': 0.72, 'out': 0.03). One global side gets one of them wrong
    # whatever it picks (both global fits: error 1.9). So every two-body term records, per endpoint, whether it is keyed
    # on the wire's input bit ("pre": before the 1q block in time), its output bit ("post") or either ("free": the
    # block is diagonal). X-type blocks are "pre": the phases were read against INPUT bits.
    term_sides = {}
    med_err2 = None
    if os.environ.get("D_DECODE_TERM_SIDES", "1") == "1" and two:
        def _block(w, setbits):
            xb = list(xi0)
            for v in setbits:
                xb[v] = 1
            Mh = np.zeros((2, 2), complex)
            for o in (0, 1):
                for i in (0, 1):
                    xi = list(xb); xi[w] = i
                    xo = [dom[v][xi[v]] for v in range(n)]; xo[w] = o
                    other = np.prod([Aw[v][xo[v], xi[v]] for v in range(n) if v != w])
                    Mh[o, i] = amp(xo, xi) / other if abs(other) > 1e-15 else 0.0
            return Mh
        blocks0 = {}
        for (a_, b_, ph) in two:
            sd = []
            for w, pw in ((a_, b_), (b_, a_)):
                offd = max(abs(Aw[w][0, 1]), abs(Aw[w][1, 0]))
                dg = max(abs(Aw[w][0, 0]), abs(Aw[w][1, 1]))
                if offd <= 0.1:
                    sd.append("free")
                elif dg <= 0.1:
                    sd.append("pre")
                else:
                    if w not in blocks0:
                        blocks0[w] = _block(w, ())
                    M0, M1 = blocks0[w], _block(w, (pw,))
                    Zp = np.diag([1.0, np.exp(1j * ph)])
                    e_pre = float(np.linalg.norm(M1 - M0 @ Zp)); e_post = float(np.linalg.norm(M1 - Zp @ M0))
                    sd.append("pre" if e_pre <= e_post else "post")
            term_sides[(a_, b_)] = tuple(sd)

        def model2(xo, xi):
            m = scale * np.prod([Aw[w][xo[w], xi[w]] for w in range(n)])
            ph_ = 0.0
            for (a_, b_, p_) in two:
                sa, sb = term_sides[(a_, b_)]
                if (xo[a_] if sa == "post" else xi[a_]) and (xo[b_] if sb == "post" else xi[b_]):
                    ph_ += p_
            return m * np.exp(1j * ph_)
        rng2 = np.random.default_rng(0)
        errs2 = []
        errs1b = []                                            # the one-sided model on the SAME elements
        hot_all = [w for w in range(n) if min(max(abs(Aw[w][0, 1]), abs(Aw[w][1, 0])), max(abs(Aw[w][0, 0]), abs(Aw[w][1, 1]))) > 0.1]
        for k_ in range(60 + 20 * len(hot_all)):
            xi = [int(rng2.random() < 0.3) for _ in range(n)]
            xo = [dom[w][xi[w]] for w in range(n)]
            if k_ >= 60:                                       # elements that FLIP a hot wire against its dominant output
                f = hot_all[(k_ - 60) % len(hot_all)]; xo[f] = 1 - xo[f]
            elif rng2.random() < 0.5:
                f = int(rng2.integers(n)); xo[f] = 1 - xo[f]
            m = amp(xo, xi); pm = model2(xo, xi)
            bm = scale * np.prod([Aw[w][xo[w], xi[w]] for w in range(n)])
            key_bits = xo if phase_side == "out" else xi
            pm1 = bm * np.exp(1j * sum(p_ for (x_, y_, p_) in two if key_bits[x_] and key_bits[y_]))
            if max(abs(m), abs(pm)) > 1e-3 * typ:
                errs2.append(abs(m - pm) / max(abs(m), abs(pm)))
                errs1b.append(abs(m - pm1) / max(abs(m), abs(pm1), 1e-300))
        med_err2 = float(np.median(errs2)) if errs2 else 1.0
        mean_err2 = float(np.mean(errs2)) if errs2 else 1.0
        mean_err1 = float(np.mean(errs1b)) if errs1b else 1.0
    worst_uni = min(s[1] / max(s[0], 1e-30) for s in sing.values())
    on_pair = sum(1 for a_, b_, ph in two if tau[a_] == b_ and abs(abs(ph) - math.pi) < 0.05)
    nondiag = [w for w in range(n) if max(abs(Aw[w][0, 1]), abs(Aw[w][1, 0])) > 0.1]
    info = {"model_err": med_err, "three_body": float(worst3), "unitary": float(worst_uni), "two_on_pair": on_pair,
            "two_off_pair": len(two) - on_pair, "nondiag_wires": nondiag, "F0": float(abs(amp(xi0, xi0)) ** 2),
            "Fref": float(abs(base) ** 2), "ref_ones": int(sum(ref)), "phase_side": phase_side, "n_err_samples": len(errs),
            "phase_side_errs": (float(np.median(e_in)) if e_in else None, float(np.median(e_out)) if e_out else None, hot)}
    info["sigma"] = {int(k_): int(v_) for k_, v_ in sigma.items()}
    corr = {("phase_side",): phase_side}
    if term_sides:
        mixed = sorted(k_ for k_, v_ in term_sides.items() if "post" in v_)
        info["term_sides_post"] = [list(k_) for k_ in mixed]
        info["model_err_one_sided"] = med_err
        info["model_err_term_sides"] = med_err2
        # the MEDIAN is blind to the minority of elements that flip a hot wire (unit test: one-sided median 0.000 while
        # the emitted operator was off by 0.67) -> choose by the MEAN over the same, hot-inclusive element set
        info["mean_err_one_sided"] = mean_err1
        info["mean_err_term_sides"] = mean_err2
        if mean_err2 <= mean_err1 + 1e-9:
            info["model_err"] = med_err2
            corr[("cz_sides",)] = {f"{a_},{b_}": list(v_) for (a_, b_), v_ in term_sides.items()}
    if tau_fixed is not None:
        corr[("tau",)] = [int(v_) for v_ in tau_fixed]
    for w in range(n):
        corr[("A", w)] = [[Aw[w][i][j].real, Aw[w][i][j].imag] for i in range(2) for j in range(2)]
    for a_, b_, ph in two:
        corr[("cz", a_, b_)] = None if abs(abs(ph) - math.pi) < 1e-3 else float(ph)
    return corr, info
