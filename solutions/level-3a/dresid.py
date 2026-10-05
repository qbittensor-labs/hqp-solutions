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

"""Residual correction for a measured block operator D.

The decode fits D with a PRODUCT ansatz  D' = (x)_w A_w . Phi  (1q blocks + DIAGONAL two-body phases)
and lets the wire relabelling carry the permutation, so what the readout actually applies differs from
what was measured by

        R = P_sigma^-1 . M . D'^-1          (M = the absorbed MPO, sigma = the decode's residual perm)

MEASURED 2026-09-21 on every public sample and on the k5 stress circuit: R is the identity to within
0.4-1.6 % on blocks the pipeline gets right (|<I,R>|/||R|| = 0.984-1.000) and keeps only **29 %** of
the operator on the one block that makes k5 answer wrong -- stable to 0.3 % across truncation cutoffs
1.2e-3 / 6e-4 / 3e-4 and across router seeds, so it is real structure and not truncation noise. In the
absorber's own (super-site) order R has bond 3-4 even in that bad case: the missing part is LOCAL
structure on tau-pairs that a diagonal phase cannot express.

So: extract R greedily as a shallow brick-wall of two-qubit gates in the absorber's order and append
them to the block's correction gates. Nothing downstream changes -- every readout path (ladder,
ensemble, operator readout, Pauli-path gate) just sees a few more gates.

Env (all optional, defaults shown):
    HQP_D_RESID=1              0 disables the whole correction
    HQP_RESID_MODE=both,pursuit,brickwall   fits to try; the best (then cheapest) is kept
    HQP_RESID_PASSES=4         brick-wall passes (one pass = even bonds then odd bonds)
    HQP_RESID_TARGET=0.9999    stop once |<I,R>|/||R|| reaches this
    HQP_RESID_GATE_TOL=1e-3    skip a gate that is a phase times the identity to within this
    HQP_RESID_STEP_GAIN=1e-4   pursuit: stop when the best pair's relative gain falls below this
    HQP_RESID_STALL=3          pursuit: non-improving steps tolerated before giving up
    HQP_RESID_MAX_GATES=120    hard cap on two-qubit gates per block
    HQP_RESID_MIN_GAIN=1e-4    emit nothing unless the fit improves the fidelity by at least this
    HQP_RESID_FID_SLACK=0.005  prefer a cheaper fit within this much of the best
    HQP_RESID_CUTOFF=1e-6      truncation cutoff for R -- much tighter than D's own (see build())
    HQP_RESID_MAX_BOND=512     bond cap while building and fitting R
    HQP_RESID_MAX_LOST=1e-3    building R may not discard more than this, else no correction at all
    HQP_RESID_KAK_TOL=1e-8     log a warning above this per-gate decomposition round-trip error
    HQP_RESID_POLISH=30        variational sweeps over the fitted gates (see _polish)
    HQP_RESID_POLISH_GAIN=1e-5 keep the polished gates only if they beat the greedy fit by this
Set in excision_solver.measure_D: HQP_RESID_ROUNDS=4, HQP_RESID_ROUND_GAIN=2e-3 (re-derive R and fit
again), HQP_RESID_RESCUE=0.99 (excise a block the ansatz gate rejects when the EMITTED D is this good).
Reference-absorber path (2026-09-24, ref_align.py): HQP_D_RESID_REF=1 (0 disables the correction there),
HQP_D_REALIGN_SIDE=L (L|R|auto: which leg is sorted into the other's wire frame first),
HQP_D_REALIGN_CUTOFF=<HQP_RESID_CUTOFF> (truncation while realigning; the bond cap is HQP_RESID_MAX_BOND).
"""
import math
import os
import time

import numpy as np

ENABLED = os.environ.get("HQP_D_RESID", "1") != "0"


def _u3(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], complex)


H2 = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2.0)


class Layout:
    """An MPO plus the site -> wire map it is currently in, under site exchanges."""

    def __init__(self, M, wire_at):
        self.M = M
        self.cur = list(wire_at)
        self.pos = {w: s for s, w in enumerate(self.cur)}
        self.peak = max(M.bonds()) if M.n > 1 else 1
        self.nsw = 0

    def _exch(self, s):
        self.peak = max(self.peak, self.M.two(s, "swap"))
        self.nsw += 1
        a, b = self.cur[s], self.cur[s + 1]
        self.cur[s], self.cur[s + 1] = b, a
        self.pos[a], self.pos[b] = s + 1, s

    def bring_adjacent(self, a, b):
        pa, pb = sorted((self.pos[a], self.pos[b]))
        while pb > pa + 1:
            self._exch(pb - 1)
            pb -= 1
        return min(self.pos[a], self.pos[b])


def two_general(M, s, G4, side):
    """Apply a general two-site gate G4[(p1,p2),(o1,o2)] to sites s, s+1 of a SyncMPO.
    side 'R' = G . M (output legs), side 'L' = M . G (input legs)."""
    if M.centre <= s:
        M.move(s)
    else:
        M.move(s + 1)
    A, B = M.T[s], M.T[s + 1]
    Dl, Dr = A.shape[0], B.shape[3]
    th = np.tensordot(A, B, axes=(3, 0))                      # l o1 i1 o2 i2 r
    G = np.asarray(G4, complex).reshape(2, 2, 2, 2)           # p1 p2 o1 o2
    if side == "R":
        th = np.einsum("abcd,lcxdyr->laxbyr", G, th)          # G . M : acts on the OUTPUT legs
    else:
        th = np.einsum("lxcydr,cdab->lxaybr", th, G)          # M . G : acts on the INPUT legs
    mat = np.ascontiguousarray(th).reshape(Dl * 4, 4 * Dr)
    try:
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        import scipy.linalg as sla
        U, S, Vh = sla.svd(mat, full_matrices=False, lapack_driver="gesvd")
    M.nsvd += 1
    s2 = S ** 2
    tail = np.cumsum(s2[::-1])[::-1]
    keep = np.nonzero(tail > M.cutoff * s2.sum())[0]
    k = max(1, min(int(keep[-1]) + 1 if keep.size else 1, M.max_bond))
    tot = float(s2.sum())
    if tot > 0:
        M.lost += float((S[k:] ** 2).sum()) / tot
    M.peak = max(M.peak, k)
    M.T[s] = U[:, :k].reshape(Dl, 2, 2, k)
    M.T[s + 1] = (S[:k, None] * Vh[:k]).reshape(k, 2, 2, Dr)
    M.centre = s + 1
    return k


def identity_fidelity(M):
    """|<I, R>| / ||R|| with I the identity MPO in the same normalisation."""
    E = np.ones((1,), complex)
    for s in range(M.n):
        E = E @ (np.einsum("akkr->ar", M.T[s]) / math.sqrt(2.0))
    nrm = np.ones((1, 1), complex)
    for s in range(M.n):
        T = M.T[s]
        nrm = np.einsum("ab,aoic,boid->cd", nrm, T.conj(), T)
    n2 = float(np.real(nrm[0, 0]))
    return abs(complex(E[0])) / math.sqrt(max(n2, 1e-300)), n2


def _traces(M):
    """Per-site identity contraction tr[s] (Dl x Dr), and the prefix/suffix products."""
    tr = [np.einsum("akkr->ar", M.T[s]) / math.sqrt(2.0) for s in range(M.n)]
    pre = [np.ones((1,), complex)]
    for s in range(M.n):
        pre.append(pre[-1] @ tr[s])
    suf = [None] * (M.n + 1)
    suf[M.n] = np.ones((1,), complex)
    for s in range(M.n - 1, -1, -1):
        suf[s] = tr[s] @ suf[s + 1]
    return tr, pre, suf


def _env(M, s):
    """4x4 environment of sites s, s+1: R contracted with the identity everywhere else."""
    _tr, pre, suf = _traces(M)
    th = np.tensordot(M.T[s], M.T[s + 1], axes=(3, 0))        # l o1 i1 o2 i2 r
    E = np.einsum("l,loxpyr,r->oxpy", pre[s], th, suf[s + 2])  # o1 i1 o2 i2
    return E.transpose(0, 2, 1, 3).reshape(4, 4)              # (o1 o2) x (i1 i2)


def _best_pair(M, tr=None, pre=None, suf=None):
    """Over ALL site pairs (a < b), the one whose best two-qubit gate gains the most overlap with I.

    With everything else traced out, <I, W> is proportional to |Tr E| and the best unitary on the pair
    turns that into the nuclear norm ||E||_*, so the exact gain of acting on a pair is ||E||_* - |Tr E|
    and the greedy choice is simply the largest one. Gates are emitted as circuit gates, so a pair does
    NOT have to be adjacent -- restricting the fit to adjacent pairs is what makes it plateau.
    """
    n = M.n
    if tr is None:
        tr, pre, suf = _traces(M)
    best = None
    for a in range(n - 1):
        X = np.einsum("l,loir->oir", pre[a], M.T[a])
        for b in range(a + 1, n):
            if b > a + 1:
                X = np.einsum("oir,rk->oik", X, tr[b - 1])
            Y = np.einsum("oir,rpqs,s->oipq", X, M.T[b], suf[b + 1])
            E = Y.transpose(0, 2, 1, 3).reshape(4, 4)
            sv = np.linalg.svd(E, compute_uv=False)
            gain = float(sv.sum() - abs(np.trace(E)))
            if best is None or gain > best[0]:
                best = (gain, a, b, E)
    return best


def _polar(E):
    """Unitary G maximising Re Tr(G^dag E)."""
    U, _S, Vh = np.linalg.svd(E)
    return U @ Vh


_P4 = np.array([0, 2, 1, 3])


def _swap4(G):
    """The same two-qubit gate with its two factors exchanged."""
    return np.asarray(G, complex)[np.ix_(_P4, _P4)]


def _is_trivial(G, tol):
    """Is G a phase times the identity?"""
    ph = np.trace(G) / 4.0
    if abs(ph) < 1e-12:
        return False
    ph /= abs(ph)
    return float(np.linalg.norm(G - ph * np.eye(4))) <= tol


def strip_ansatz(L, gl):
    """R <- R . D'^-1 on the INPUT leg; gl = the ansatz gate list in WIRE labels, circuit order."""
    for t, q, p in gl:
        if t == "u":
            L.M.one(L.pos[q[0]], _u3(*p).conj().T, "L")
        else:
            L.M.two(L.bring_adjacent(q[0], q[1]), "cz", side="L")


def strip_perm(L, pairs):
    """R <- P^-1 . R on the OUTPUT leg, P an involution given as pairs, as SWAPs built from CZ+H."""
    for a, b in pairs:
        s = L.bring_adjacent(a, b)
        sa, sb = (s, s + 1) if L.cur[s] == a else (s + 1, s)
        for hi in (sb, sa, sb):
            L.M.one(hi, H2, "R")
            L.M.two(s, "cz", side="R")
            L.M.one(hi, H2, "R")


def _decompose(G, w_msb, w_lsb):
    """A general two-qubit unitary as ('u'|'cz') gates (qiskit's CZ-basis KAK), up to a global phase.

    G is indexed (o_first, o_second) x (i_first, i_second) with the FIRST site most significant, which
    is qiskit's qubit 1; the second site is qubit 0.
    """
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import CZGate
    from qiskit.synthesis import TwoQubitBasisDecomposer
    dec = TwoQubitBasisDecomposer(CZGate(), euler_basis="U")
    # approximate=False: the decomposer's DEFAULT is to trade basis gates for fidelity, which silently
    # emits a 2-CZ approximation of a 3-CZ gate. MEASURED: that costs ~1e-6 of operator fidelity per
    # affected gate and showed up as the emitted circuit disagreeing with the extraction it came from.
    qc = dec(Operator(np.asarray(G, complex)), approximate=False)
    wires = {0: int(w_lsb), 1: int(w_msb)}
    out = _to_gates(qc, wires)
    # The decomposer is not exact for every gate: MEASURED, a gate near a degenerate Weyl point comes
    # back with 3.7e-10 infidelity (2.7e-5 in amplitude) and re-decomposing the leftover does not help,
    # because forcing three CZs on a near-identity residual is that same ill-conditioned case. The
    # effect is bounded and four orders below the residual we are correcting (0.0065 on a good block),
    # so it is accepted -- but never silently: the worst round trip is recorded and logged.
    err = 1.0 - abs(np.vdot(_dense2(out, int(w_msb), int(w_lsb)).reshape(-1),
                            np.asarray(G, complex).reshape(-1)) / 4.0)
    KAK_WORST[0] = max(KAK_WORST[0], float(err))
    return out


KAK_WORST = [0.0]


def _dense2(gl, w_msb, w_lsb):
    """Dense 4x4 of a two-wire ('u'|'cz') gate list, w_msb first."""
    idx = {int(w_msb): 0, int(w_lsb): 1}
    U = np.eye(4, dtype=complex)
    for t, q, p in gl:
        if t == "u":
            ops = [np.eye(2, dtype=complex), np.eye(2, dtype=complex)]
            ops[idx[q[0]]] = _u3(*p)
            U = np.kron(ops[0], ops[1]) @ U
        else:
            U = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex) @ U
    return U


def _to_gates(qc, wires):
    out = []
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        nm = inst.operation.name
        if nm == "cz":
            out.append(("cz", (wires[qs[0]], wires[qs[1]]), None))
        elif nm in ("u", "u3"):
            out.append(("u", (wires[qs[0]],), tuple(float(x) for x in inst.operation.params)))
        elif nm in ("barrier", "id"):
            continue
        else:
            raise ValueError(f"unexpected gate {nm} from the KAK decomposition")
    return out


_T_END = [None]          # in-round time budget for one build() (HQP_RESID_FIT_MAX_S), see build()


def _past():
    return _T_END[0] is not None and time.time() > _T_END[0]


def _fit(L, mode, target, passes, gtol, maxg):
    """Greedily peel R into two-qubit gates. Returns (extracted, fid, trace, stop).

    'brickwall' sweeps adjacent bonds (cheap, catches the tau-pair structure); 'pursuit' repeatedly
    takes the globally best pair; 'both' runs the brick-wall first and then pursues what is left.
    Each extracted gate is applied to R, so the emitted circuit is R ~ G_1 G_2 ... G_k and the gates
    must be emitted in REVERSE order of extraction.
    """
    n = L.M.n
    extracted = []
    fid, _ = identity_fidelity(L.M)
    trace = [fid]
    stop = "target"
    if mode in ("brickwall", "both"):
        for _p in range(passes):
            if fid >= target or len(extracted) >= maxg:
                break
            if _past():
                stop = "time"
                break
            for off in (0, 1):
                for s in range(off, n - 1, 2):
                    if len(extracted) >= maxg:
                        break
                    G = _polar(_env(L.M, s))
                    if _is_trivial(G, gtol):
                        continue
                    two_general(L.M, s, G.conj().T, "R")
                    extracted.append((L.cur[s], L.cur[s + 1], G))
                fid, _ = identity_fidelity(L.M)
                trace.append(fid)
                if fid >= target:
                    break
        if mode == "brickwall":
            stop = "brick-wall passes exhausted" if fid < target else "target"
    if mode in ("pursuit", "both"):
        min_gain = float(os.environ.get("HQP_RESID_STEP_GAIN", "1e-4"))
        stall = 0
        while fid < target and len(extracted) < maxg:
            if _past():
                stop = "time"
                break
            best = _best_pair(L.M)
            if best is None:
                stop = "no pair"
                break
            gain, a, b, E = best
            base = abs(complex(np.trace(E)))
            if gain <= min_gain * max(base, 1e-12):
                stop = "gain below threshold"
                break
            G = _polar(E)
            if _is_trivial(G, gtol):
                stop = "best gate is trivial"
                break
            wa, wb = L.cur[a], L.cur[b]
            s = L.bring_adjacent(wa, wb)
            Gc = G if L.cur[s] == wa else _swap4(G)
            two_general(L.M, s, Gc.conj().T, "R")
            extracted.append((wa, wb, G))
            fid, _ = identity_fidelity(L.M)
            trace.append(fid)
            # a unitary cannot lower the overlap, so a step that does not improve means the TRUNCATION
            # ate the gain; allow a couple of those before giving up
            if trace[-1] <= trace[-2] + 1e-9:
                stall += 1
                if stall >= int(os.environ.get("HQP_RESID_STALL", "3")):
                    stop = "truncation-limited (stalled)"
                    break
            else:
                stall = 0
    if len(extracted) >= maxg:
        stop = "gate cap"
    return extracted, fid, trace, stop


def _polish(R0, wire_at, extracted, sweeps, log=None):
    """Variationally re-optimise the extracted gates against their TRUE environments.

    The greedy peel picks each gate against the identity, which is a matching pursuit and stalls at a
    local optimum: MEASURED, k5 block 3180 stops at 0.955 although its R's unitarity deficit is 7e-4,
    i.e. the same as a block the fit takes to 0.992 -- so the plateau is the fit, not the operator.

    With C = G_1 G_2 ... G_k (as an operator product) the objective is Tr(C^dag R), and cyclicity gives
        Tr(C^dag R) = Tr(G_j^dag . Y_j),    Y_j = G_{j-1}^dag..G_1^dag . R . G_k^dag..G_{j+1}^dag,
    so the optimal G_j is the unitary polar factor of Y_j's environment, and sweeping j costs two local
    applications per step:  Y_{j+1} = G_j^dag . Y_j . G_{j+1}.
    """
    import copy
    k = len(extracted)
    if k == 0:
        return extracted, None
    G = [np.array(g) for _a, _b, g in extracted]
    wires = [(a, b) for a, b, _g in extracted]
    best = None
    for _sw in range(sweeps):
        Y = Layout(copy.deepcopy(R0), wire_at)
        for j in range(k - 1, 0, -1):                             # Y_1 = R . G_k^dag .. G_2^dag
            _apply(Y, wires[j], G[j].conj().T, "L")
        for j in range(k):
            s = Y.bring_adjacent(*wires[j])
            E = _env(Y.M, s)
            if Y.cur[s] != wires[j][0]:
                E = _swap4(E)
            G[j] = _polar(E)
            if j + 1 < k:
                _apply(Y, wires[j], G[j].conj().T, "R")
                _apply(Y, wires[j + 1], G[j + 1], "L")
        # Score this sweep by rebuilding the peeled residual from scratch (no accumulated drift).
        # W must be C^dag R = G_k^dag .. G_1^dag R, and applying on side "R" prepends, so G_1^dag goes
        # FIRST. Applying them the other way round silently scores a different operator -- it turned an
        # exact solution into 0.105 in the known-answer test (clean/t_polish.py).
        W = Layout(copy.deepcopy(R0), wire_at)
        for j in range(k):
            _apply(W, wires[j], G[j].conj().T, "R")
        fid, _ = identity_fidelity(W.M)
        if best is None or fid > best[0] + 1e-12:
            best = (fid, [g.copy() for g in G])
        else:
            break
    if best is None:
        return extracted, None
    return [(wires[j][0], wires[j][1], best[1][j]) for j in range(k)], best[0]


def _apply(L, pair, G4, side):
    """Apply a two-qubit gate on `pair` (first wire = most significant) to L's MPO, routing as needed."""
    s = L.bring_adjacent(*pair)
    two_general(L.M, s, G4 if L.cur[s] == pair[0] else _swap4(G4), side)


def build(M, wire_at, ansatz_gates, sigma, log=None, cutoff=None):
    """Returns (gates, info). `gates` are extra correction gates in WIRE labels, to be emitted AFTER
    the ansatz gates. Empty when the correction is not worth (or not safe) applying."""
    log = log or (lambda m: None)
    info = {"resid": True}
    if not ENABLED:
        return [], {"resid": False, "reason": "disabled"}
    import copy
    n = M.n
    Mc = copy.deepcopy(M)
    # R is built at a MUCH tighter cutoff than D itself. At D's own 6e-4, stripping the ansatz discarded
    # 1.1e-3 of R's weight on a real block (d3_s1 3120) -- enough to trip the reliability guard below.
    # R stays near-product, so tightening costs nothing: `cutoff` (D's) is only a floor-of-last-resort.
    Mc.cutoff = float(os.environ.get("HQP_RESID_CUTOFF", "1e-6"))
    Mc.max_bond = int(os.environ.get("HQP_RESID_MAX_BOND", "512"))
    L = Layout(Mc, wire_at)
    lost0 = Mc.lost
    try:
        strip_ansatz(L, ansatz_gates)
        pairs = sorted({(min(a, b), max(a, b)) for a, b in (sigma or {}).items() if a != b})
        if pairs:
            strip_perm(L, pairs)
    except Exception as e:                                        # noqa: BLE001
        log(f"  resid: could not build R ({type(e).__name__}: {str(e)[:90]}) -> no correction")
        return [], {"resid": False, "reason": f"build failed: {type(e).__name__}"}
    # Stripping the ansatz off R routes long-range CZs through site exchanges, so the bond can grow and
    # be truncated. If that truncation is not negligible, R is not the residual any more and everything
    # built on it -- the correction AND the |<I,R>| trust number -- would be believable and wrong.
    strip_lost = L.M.lost - lost0
    info["strip_lost"] = strip_lost
    max_lost = float(os.environ.get("HQP_RESID_MAX_LOST", "1e-3"))
    if strip_lost > max_lost:
        log(f"  resid: building R discarded {strip_lost:.2e} of its weight (> {max_lost:g}, bond cap "
            f"{Mc.max_bond}) -- R is not reliable, no correction and no overlap reported")
        return [], {"resid": False, "reason": f"strip truncation {strip_lost:.2e}",
                    "strip_lost": strip_lost}
    fid0, _ = identity_fidelity(L.M)
    info["fid0"] = fid0
    target = float(os.environ.get("HQP_RESID_TARGET", "0.9999"))
    passes = int(os.environ.get("HQP_RESID_PASSES", "4"))
    gtol = float(os.environ.get("HQP_RESID_GATE_TOL", "1e-3"))
    maxg = int(os.environ.get("HQP_RESID_MAX_GATES", "120"))
    # The two strategies stall on different things (MEASURED on k5 block 3180: brick-wall alone
    # 0.294 -> 0.916 in 14 gates, pursuit alone -> 0.907 in 8, brick-wall THEN pursuit -> 0.934 in 19),
    # and which one wins is block-dependent. Each costs ~0.1 s, so run them all and keep the best fit
    # rather than betting on a mode.
    modes = [m.strip().lower() for m in os.environ.get("HQP_RESID_MODE", "both,pursuit,brickwall").split(",") if m.strip()]
    base = L
    results = []
    # IN-ROUND TIME BUDGET (HQP_RESID_FIT_MAX_S, default 300 s per build()): the fit is pure Python (best-pair searches
    # over ~1100 wire pairs per extracted gate, up to 120 gates, three modes, then polishing); MEASURED 2026-09-24 on
    # genadv h53 block 3020 a single round ran > 33 min single-threaded while the D stage waited (h35's reference
    # worker spent 1142 s of its 1144 s here). Past the budget: the current pass/pursuit stops ("time"), the remaining
    # modes are skipped, polishing is skipped; whatever was extracted is kept and scored as usual.
    _T_END[0] = time.time() + float(os.environ.get("HQP_RESID_FIT_MAX_S", "300"))
    for m in modes:
        if _past() and results:
            log(f"  resid: fit time budget spent -> mode '{m}' and the rest skipped")
            break
        Lm = Layout(copy.deepcopy(base.M), base.cur)
        ext, f, tr_, stop = _fit(Lm, m, target, passes, gtol, maxg)
        results.append((f, m, ext, tr_, stop, Lm))
        log(f"  resid: fit '{m}': {fid0:.6f} -> {f:.6f} with {len(ext)} two-qubit gates ({stop})")
    # The correction is not free: its gates are entangling, and they make the reduced circuit harder for
    # the readout (MEASURED on k5, whose reduced circuit went 1447 -> 1583 gates and whose chi-512 peak
    # weight fell 17x). So among fits that are within HQP_RESID_FID_SLACK of the best, take the cheapest.
    results.sort(key=lambda r: -r[0])
    slack = float(os.environ.get("HQP_RESID_FID_SLACK", "0.005"))
    near = [r for r in results if r[0] >= results[0][0] - slack]
    near.sort(key=lambda r: (len(r[2]), -r[0]))
    fid, mode, extracted, trace, stop, L = near[0]
    info["stop"] = stop
    info["fits"] = {m: round(f, 6) for f, m, _e, _t, _s, _L in results}
    # The greedy fits are matching pursuits and stall at a local optimum; a variational sweep over the
    # SAME gates, each re-optimised against its true environment, costs two local applications per gate.
    npol = int(os.environ.get("HQP_RESID_POLISH", "30"))
    if npol > 0 and extracted and _past():
        log("  resid: fit time budget spent -> polishing skipped")
    if npol > 0 and extracted and not _past():
        try:
            _e2, _f2 = _polish(base.M, base.cur, extracted, npol, log=log)
            if _f2 is not None and _f2 > fid + float(os.environ.get("HQP_RESID_POLISH_GAIN", "1e-5")):
                log(f"  resid: variational polish {fid:.6f} -> {_f2:.6f} ({len(extracted)} gates)")
                extracted, fid = _e2, _f2
                info["polished"] = True
        except Exception as e:                                    # noqa: BLE001
            log(f"  resid: polish failed ({type(e).__name__}: {str(e)[:80]}) -> greedy fit kept")

    info["trace"] = [round(f, 6) for f in trace]
    info.update({"fid1": fid, "ngates": len(extracted), "bond": max(L.M.bonds()), "lost": L.M.lost,
                 "mode": mode, "steps": len(trace) - 1})
    gain = fid - fid0
    if not extracted or gain < float(os.environ.get("HQP_RESID_MIN_GAIN", "1e-4")):
        info["applied"] = False
        log(f"  resid: R already |<I,R>|={fid0:.6f} (fit {fid:.6f}, {len(extracted)} gates) -> no correction")
        return [], info
    if os.environ.get("HQP_RESID_DEBUG") == "1":
        info["_extracted"] = [(int(a), int(b), np.array(g)) for a, b, g in extracted]
    out = []
    for wa, wb, G in reversed(extracted):                         # R = G_1 G_2 ... G_k -> emit G_k first
        out.extend(_decompose(G, int(wa), int(wb)))
    info["applied"] = True
    info["ncz"] = sum(1 for g in out if g[0] == "cz")
    info["kak_worst"] = KAK_WORST[0]
    if KAK_WORST[0] > float(os.environ.get("HQP_RESID_KAK_TOL", "1e-8")):
        log(f"  resid: WARNING the two-qubit decomposition is inexact for some gate "
            f"(worst round-trip infidelity {KAK_WORST[0]:.2e})")
    log(f"  resid: |<I,R>| {fid0:.6f} -> {fid:.6f} with {len(extracted)} two-qubit gates "
        f"({info['ncz']} CZ, {len(out)} gates total, R bond {info['bond']})")
    return out, info
