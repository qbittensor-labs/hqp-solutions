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

"""tau-block excision stage for HQP difficulty-3 circuits.

Why this exists: a d3 circuit is a shallow trained peaked circuit R > P with mirror blocks
`U . SWAP_tau . U^dag` inserted. Each block is, as an operator, the wire permutation Pi_tau times
a small residual D. Excising the blocks (replacing them by the relabeling and re-inserting the
MEASURED D) turns d3 back into the d1-shaped circuit it was built from -- and that shape is
directly simulable by the canonical forward-MPS ladder, which cracks it in minutes. Validated
2026-09-11/12: d3_s1 -> Hamming 0 at chi 256/512/1024 (blind, truth-free discovery).

Every stage fails CLOSED: a block whose D does not pass its quality gate is simply not excised,
and if nothing can be excised the caller falls back to the existing unswap path.

Env (all optional, defaults shown):
  HQP_EXCISE=1            master switch
  HQP_EXCISE_MIN_Q=44     only attempt at or above this qubit count
  HQP_EXCISE_CHI=256,512,1024   canonical ladder on the reduced circuit
  HQP_EXCISE_MARGIN=1.15  margin floor for a stable top to be trusted
  HQP_EXCISE_D_BUDGET=2400   per-block GPU seconds for the D measurement
  D_MAX_MODEL_ERR=0.35    reject a block whose D model misfits random matrix elements
  D_MAX_3BODY=0.05        reject a block whose D is not 1+2-body
  EXC_MIN_EVIDENCE=3, EXC_PEEL_MODE=close, EXC_MAX_DIST=900, EXC_TWIN_TOL=1e-6   cut knobs
"""
import os, time, math, itertools
from collections import Counter
import numpy as np

import wall_watchdog

# The cut knobs validated end-to-end (d3_s1 and d3_s2 both -> Hamming 0). excise.py's own defaults
# are the permissive ones; pin the validated values here unless the caller overrides them.
os.environ.setdefault("EXC_MIN_EVIDENCE", "3")
os.environ.setdefault("EXC_PEEL_MODE", "close")
os.environ.setdefault("EXC_MAX_DIST", "900")

EXCISE_ON = os.environ.get("HQP_EXCISE", "1").strip() not in ("0", "false", "False")
MIN_Q = int(os.environ.get("HQP_EXCISE_MIN_Q", "44"))
CHIS = [int(x) for x in os.environ.get("HQP_EXCISE_CHI", "256,512,1024").split(",")]
MARGIN = float(os.environ.get("HQP_EXCISE_MARGIN", "1.15"))
# Per-block GPU budget for the D measurement. A hard 2400 s was too small for d3_s2's second
# block (it needs ~2000-3300 s), and an incomplete absorption is rejected outright, so scale the
# budget to the time actually left instead of a fixed constant.
D_BUDGET = float(os.environ.get("HQP_EXCISE_D_BUDGET", "3600"))
D_BUDGET_MIN = float(os.environ.get("HQP_EXCISE_D_BUDGET_MIN", "900"))
# Time held back from D measurement for the chi ladder that actually produces the answer.
# 1200 s could not pay for it: trust needs a top that is STABLE ACROSS RUNGS under TWO gate
# orders, i.e. >= 2 rungs x 2 orders, and on 48q chi 256/512 measured ~5/~10 min each
# (~1800 s). A measured D with no ladder to read it is worth nothing, so reserve the ladder.
CANON_RESERVE = float(os.environ.get("HQP_EXCISE_CANON_RESERVE", "2400"))
# Fraction of the remaining wall the excision stage may consume (retries included); the rest
# is left to the fallbacks. d3 wants nearly all of it; d2-shaped circuits still need unswap.
MAX_FRAC = float(os.environ.get("HQP_EXCISE_MAX_FRAC", "0.9"))
ORDERS = [o.strip() for o in os.environ.get("HQP_EXCISE_ORDERS", "dag,raw").split(",") if o.strip()]
# A wrong answer costs a whole validation run (3 are required), and two gate orders disagreeing is
# exactly the signature of the failed production run, so agreement is the trust signal.
REQUIRE_AGREEMENT = os.environ.get("HQP_EXCISE_REQUIRE_AGREEMENT", "1").strip() not in ("0", "false", "False")
# Trust bar when only ONE gate order ever reached a verdict (out of time / ORDERS has one entry),
# i.e. when cross-order agreement cannot be the discriminator. MARGIN (1.15) is inside the measured
# noise band (chi-stable wrong argmaxes at 1.17/1.15/1.14); solving runs measured 1.18-1.21 (d3_s1,
# which trusts via AGREEMENT anyway) and 1.47-1.71 (d3_s2). 1.35 clears the noise band entirely.
SOLO_MARGIN = float(os.environ.get("HQP_EXCISE_SOLO_MARGIN", "1.35"))
# ---- independent-draw consensus --------------------------------------------------------
# A single D can fit its own MPO perfectly and still be WRONG: that is exactly how the
# 2026-09-15 validator run ended in IncorrectFailure -- one D's chi-ladder argmax, submitted
# because nothing better existed. Different (cutoff, seed) draws produce genuinely independent
# reduced circuits, so the bitstring MOST of them agree on is a far better pick than the
# highest margin of any single one. CONSENSUS agreeing votes are needed to trust, spanning at
# least MIN_DRAWS distinct D measurements; with at most 2 gate orders per ladder run, 4 votes
# necessarily span >= 2 independent draws. At the deadline the PLURALITY winner is submitted.
CONSENSUS = int(os.environ.get("HQP_EXCISE_CONSENSUS", "4"))
MIN_DRAWS = int(os.environ.get("HQP_EXCISE_MIN_DRAWS", "2"))
# Cost growth from one chi rung to the next. MPS contraction/SVD cost rises superlinearly in chi;
# measured on 48q the 256->512->1024 steps each cost roughly 3-4x the previous rung.
RUNG_GROWTH = float(os.environ.get("HQP_EXCISE_RUNG_GROWTH", "4.0"))
# Calibrated on measured blocks: d3_s1 B1/B2 = 0.15, d3_s2 B2 = 0.06, d3_s2 B1 = 0.42 (all of
# which SOLVE); budget-truncated garbage = 0.71 and 1.00 (must be rejected).
MAX_MODEL_ERR = float(os.environ.get("D_MAX_MODEL_ERR", "0.5"))
# 3-body residual: solving blocks measured 1e-5 (d3_s1), 3e-6 (d3_s2 B2) and 2.1e-2 (the d3_s2 B1
# that solved); genuine garbage sits at 3.1e-1+. 0.05 rejected a 0.057 that was otherwise fine.
MAX_3BODY = float(os.environ.get("D_MAX_3BODY", "0.1"))
D_CUTOFF = float(os.environ.get("HQP_EXCISE_D_CUTOFF", "6e-4"))
D_MAXBOND = int(os.environ.get("HQP_EXCISE_D_MAXBOND", "3072"))
RETRY_CUTOFF = float(os.environ.get("HQP_EXCISE_RETRY_CUTOFF", "2e-4"))
# (cutoff, seed) ladder per block: keep the first attempt that passes the D gate.
# Measured: d3_s2's B1 failed repeatedly at 6e-4 (model err 0.26-0.99, chaotic) and passed
# decisively at 2e-4 (0.049). Cheap cutoff first, then tighter -- accuracy is what matters.
# ---- the (cutoff, seed) draw schedule --------------------------------------------------------
# Rung 1 is unchanged, so every block that passes first time behaves exactly as validated.
# What changed is the RETRY order. The old ladder escalated the cutoff while holding seed 123
# fixed, on the theory that a failed draw needs more accuracy. MEASURED on d3_s2's block 2 -- the
# only first-rung failure in either public sample, and it failed there twice (fits 0.517 / 1.000):
#     seed 123 @6e-4 -> FAIL, FAIL          seed 456 @6e-4 -> PASS 1057 s (fit 0.145)
#     seed 123 @2e-4 -> PASS 1331-1782 s    seed 789 @6e-4 -> PASS 1049 s (fit 0.041)
#                                           seed 321 @6e-4 -> PASS 1622 s (fit 0.185)
# So that failure was an unlucky SEED, not insufficient accuracy, and a free reseed at the cheap
# cutoff fixes it FASTER than the tighter cutoff does. The schedule now alternates: reseed first
# (cheap, and the measured fix), then escalate the cutoff (the previously validated fix), so it
# hedges both causes instead of betting everything on accuracy.
SEEDS = [int(x) for x in os.environ.get("HQP_EXCISE_SEEDS", "123,456,789,321,654,987").split(",") if x.strip()]
_CUTOFFS = [D_CUTOFF, RETRY_CUTOFF, RETRY_CUTOFF / 2]


def _build_schedule():
    """Interleave reseeds and cutoff escalations, cheapest-and-most-likely first, no duplicates."""
    sched, seen = [], set()
    def add(c, sd):
        if (c, sd) not in seen:
            seen.add((c, sd)); sched.append((c, sd))
    add(_CUTOFFS[0], SEEDS[0])                       # rung 1: unchanged
    for k in range(1, max(len(SEEDS), len(_CUTOFFS))):
        if k < len(SEEDS):
            add(_CUTOFFS[0], SEEDS[k])               # a fresh seed at the cheap cutoff
        if k < len(_CUTOFFS):
            add(_CUTOFFS[k], SEEDS[0])               # then the next cutoff down
    for c in _CUTOFFS[1:]:                           # budget-filling tail: tighter cutoffs x seeds
        for sd in SEEDS[1:]:
            add(c, sd)
    return sched


ATTEMPTS = _build_schedule()
if os.environ.get("HQP_EXCISE_ATTEMPTS"):
    ATTEMPTS = [(float(a.split(":")[0]), int(a.split(":")[1])) for a in os.environ["HQP_EXCISE_ATTEMPTS"].split(",")]
# Only walk past the first few rungs for genuinely d3-structured circuits (>= this many mirror
# blocks). A d1/d2-shaped circuit still needs its canonical/unswap fallback, so it must not have
# its wall eaten here.
EXTEND_MIN_BLOCKS = int(os.environ.get("HQP_EXCISE_EXTEND_MIN_BLOCKS", "2"))
# Consecutive sweeps where no block produced a better D before giving up (chaotic measurement:
# one dry sweep is not evidence that the next seed also misses).
MAX_DRY_SWEEPS = int(os.environ.get("HQP_EXCISE_MAX_DRY_SWEEPS", "2"))
# Rungs available to a circuit that is NOT d3-structured (keeps its fallback budget intact).
BASE_RUNGS = int(os.environ.get("HQP_EXCISE_BASE_RUNGS", "3"))


def attempt_at(i):
    """(cutoff, seed) for rung i."""
    return ATTEMPTS[min(i, len(ATTEMPTS) - 1)]


def n_rungs(blocks_found):
    if blocks_found < EXTEND_MIN_BLOCKS:
        return min(BASE_RUNGS, len(ATTEMPTS))
    return len(ATTEMPTS)


def _circ_to_gates(circ):
    """QuantumCircuit (u/cz only) -> excise.py gate list."""
    idx = {q: i for i, q in enumerate(circ.qubits)}
    g = []
    for inst in circ.data:
        nm = inst.operation.name
        if nm in ("barrier", "measure"):
            continue
        qs = tuple(idx[q] for q in inst.qubits)
        if nm == "cz":
            g.append(("cz", qs, None))
        elif nm in ("u", "u3"):
            g.append(("u", qs, tuple(float(p) for p in inst.operation.params)))
        else:
            raise ValueError(f"excision needs a raw u/cz circuit, found {nm}")
    return g


def _absorber():
    """The reference absorber, imported lazily: the synchronous engine (syncd.py) does not need the quimb stack."""
    # HQP_EXCISE_ABSORBER=ref (default) | fork. The D-measurement absorbs a mirror block centred on its
    # seam; 'ref' is the authors' mpo_compress_unswap (unswap_ref.py, + deadline/stall guards + norm
    # meter, no unseeded jitter). MEASURED 2026-09-18: d3_s1 ham 0 / margin 1.66 / 2802 s, block
    # absorption loses 10^-0.08 of operator norm, deterministic per seed. 'fork' = our modified copy,
    # 1800x worse at full absorption on d2 and the source of the run-to-run D lottery.
    if os.environ.get("HQP_EXCISE_ABSORBER", "ref").strip().lower() == "fork":
        from unswap import mpo_compress_unswap
    else:
        from unswap_ref import mpo_compress_unswap
    return mpo_compress_unswap


def _consolidate_sections(qc, junctions):
    """Consolidate each section of qc separately and concatenate, so that no 2q block straddles a junction and a
    junction is an exact instruction index of the result (same as op_readout.consolidate_sections, kept here so a
    D worker does not import the readout stack). Returns (circuit, [instruction index of every junction])."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
    from qiskit.transpiler import PassManager
    data = [inst for inst in qc.data if inst.operation.name not in ("barrier", "measure")]
    bounds = [0] + [int(j) for j in junctions] + [len(data)]
    out = QuantumCircuit(qc.num_qubits)
    idx = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sec = QuantumCircuit(qc.num_qubits)
        for inst in data[a:b]:
            sec.append(inst.operation, [sec.qubits[qc.find_bit(q).index] for q in inst.qubits])
        if b > a:
            sec = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(sec)
        out.compose(sec, inplace=True)
        idx.append(len(out.data))
    return out, idx[:-1]


def wire_split(gates, n, cut, twins, tau, centre, Eidx, mode="clip", log=None):
    """Per-wire split of a cut at the MIRROR SURFACE:  split[w] = index such that the cut's gates on wire w with
    index <= split[w] belong to the L (U) half and the others to the R (U^dag) half.

    Why (MEASURED 2026-09-19, d3_s2): the qasm lists a block in a topological order that is far from layer order,
    so the index split `i < centre` puts ~95 L-type gates (40 CZ, two layers' worth) AFTER the explicit swaps and
    ~40 R-type gates before them. The operator is the same, but the absorber works from the split outward and those
    gates cannot cancel until the window has grown past the misplacement: the 'rough phase' at t_u 100-200 that
    costs half of a plain cut's measurement and more than half of an edge-extended cut's.

    ANY per-wire split gives exactly D = Pi.E provided every 2q gate is wholly before or wholly after it (then
    [L gates][SWAP_tau][R gates relabelled] preserves every wire's order), so the twin structure is only used to
    choose a good one: on wire w the split must lie in the gap between the last L twin and the first R twin.
      mode "clip" : the index centre clipped into each wire's gap (no core detection needed)
      mode "core" : the pair's own swap moment (between its two core CZ(x, tau x)) where one is found, else "clip"
      mode "index": the historical index split (split[w] = centre - 1 everywhere)
    Returns (split, info)."""
    W = cut.W
    E = set(Eidx)
    info = {"mode": mode, "clipped": 0, "bad_gap": 0, "moved": 0, "forced": 0}
    split = {w: centre - 1 for w in range(n)}
    if mode == "index":
        return split, info
    Ls, Rs = set(twins.values()), set(twins.keys())
    lo, hi = {}, {}
    for w in range(n):
        seq = [i for i in W[w] if i in E]
        l = [i for i in seq if i in Ls]
        r = [i for i in seq if i in Rs]
        lo[w] = max(l) if l else -1
        hi[w] = (min(r) - 1) if r else 10 ** 9
        if lo[w] > hi[w]:                     # an R twin before an L twin on one wire: twin misdetection -> no constraint
            info["bad_gap"] += 1
            lo[w], hi[w] = -1, 10 ** 9
        v = min(max(centre - 1, lo[w]), hi[w])
        if mode == "late" and hi[w] < 10 ** 9:
            v = hi[w]                         # the whole twin-free surface zone of the wire goes to the L half
        elif mode == "early" and lo[w] >= 0:
            v = lo[w]                         # ... or to the R half
        info["clipped"] += int(v != centre - 1)
        split[w] = v
    if mode == "core":
        # the pair's OWN swap moment: between the two closest CZ(x, tau x) of the twin-free zone of the pair; the
        # file order lists a pair's core up to ~300 gates away from the scan centre (MEASURED -212..+334 on d3_s2)
        info["cores"] = 0
        for x in range(n):
            y = tau[x]
            if not x < y:
                continue
            zl = max(lo[x], lo[y])
            zr = min(hi[x], hi[y]) + 1
            cstar = [i for i in W[x] if i in E and gates[i][0] == "cz" and set(gates[i][1]) == {x, y} and zl < i < zr]
            if not cstar:
                continue
            if len(cstar) >= 2:
                k = min(range(len(cstar) - 1), key=lambda j: cstar[j + 1] - cstar[j])
                c1, c2 = cstar[k], cstar[k + 1]
            else:
                c1 = c2 = cstar[0]
            info["cores"] += 1
            for w in (x, y):
                seq = [i for i in W[w] if i in E]
                s_w = c1
                for i in seq[seq.index(c1) + 1:]:
                    if i >= c2 and c2 != c1:
                        break
                    t, q, p_ = gates[i]
                    if t == "cz":
                        break
                    th = abs(p_[0]) % (2 * math.pi)
                    th = 2 * math.pi - th if th > math.pi else th
                    if not (th < 0.06 or abs(th - math.pi) < 0.06):
                        break                                 # the non-diagonal 1q gate between the two core CZs
                    s_w = i
                split[w] = min(max(s_w, lo[w]), hi[w])
    if mode == "core2":
        # The surface of a pair (x, tau x) is  ... H C*1 H C*2 H ...  (C* = CZ(x, tau x); = SWAP.CZ), possibly with a
        # Pauli-frame WRAP  cz(x,z) [diag|X] C* [diag|X] cz(x,z)  around C*1 or C*2 (MEASURED on d3_s2: 17 wraps, 10
        # mirror pairs the index-based twin finder cannot see because BOTH members lie on one side of the scan
        # centre, 3 loners; nothing else is untwinned). Split every wire at the first NON-diagonal 1q gate after C*1:
        # both CZs of a wrap then stay on one side, and everything before a wire's core is L whatever its file index.
        info["cores"] = 0
        def _cls(p_):
            th = abs(p_[0]) % (2 * math.pi)
            th = 2 * math.pi - th if th > math.pi else th
            return "d" if (th < 0.06 or abs(th - math.pi) < 0.06) else "h"
        for x in range(n):
            y = tau[x]
            if not x < y:
                continue
            seqx = [i for i in W[x] if i in E]
            seqy = [i for i in W[y] if i in E]
            cst = [i for i in seqx if gates[i][0] == "cz" and set(gates[i][1]) == {x, y}]
            best = None
            for c1, c2 in zip(cst[:-1], cst[1:]):
                ok = True
                nh = 0
                for seq in (seqx, seqy):
                    between = seq[seq.index(c1) + 1: seq.index(c2)]
                    if any(gates[i][0] == "cz" and (i in Ls or i in Rs) for i in between):
                        ok = False                      # a twinned CZ between them: not a core
                    nh += sum(1 for i in between if gates[i][0] != "cz" and _cls(gates[i][2]) == "h")
                if ok and nh >= 1:
                    d = abs((c1 + c2) / 2.0 - centre)
                    if best is None or d < best[0]:
                        best = (d, c1, c2)
            if best is None:
                continue
            info["cores"] += 1
            _, c1, c2 = best
            for w, seq in ((x, seqx), (y, seqy)):
                s_w = c1
                for i in seq[seq.index(c1) + 1: seq.index(c2)]:
                    if gates[i][0] != "cz" and _cls(gates[i][2]) == "h":
                        break
                    s_w = i
                split[w] = s_w
    two = sorted(i for i in E if len(gates[i][1]) == 2)
    # TWIN-SAFE resolution of wire-mixed 2q gates (EXC_SPLIT_TWINSAFE=1, default). The historical core2 rule always
    # RAISES the later wire's split (the gate becomes wholly L).  MEASURED 2026-09-24 on d3_s1 + 40 one-sided identity
    # insertions cz(a,z) G cz(a,z) per block (the platform's own obfuscation shape): 70 such moves instead of 20, the
    # splits drift towards the R half, 10 evidence-backed twin pairs end up with BOTH members on the L side, the
    # per-wire LCS loses 35 mirror pairs and the sync engine hits TooBig (bond 512) after 72 CZ.  A twin (member of
    # the cut's evidence-backed twin map) is KNOWN to belong to its half, so a split must never move across one: when
    # raising would cross an R twin and lowering the other wire's split (the gate becomes wholly R) crosses no L twin,
    # lower instead.  Each wire keeps one direction of movement (up or down) so the loop terminates; any order of the
    # halves stays exact (see the docstring), so a forced move is never a correctness risk, only a bond-cost one.
    twinsafe = mode == "core2" and os.environ.get("EXC_SPLIT_TWINSAFE", "1").strip() not in ("0", "false", "no", "off", "")
    direction = {}
    info["lowered"] = 0
    for it in range(20000):
        moved = False
        for i in two:
            a, b = gates[i][1]
            ba, bb = i <= split[a], i <= split[b]
            if ba == bb:
                continue
            w_after, w_before = (b, a) if ba else (a, b)
            if mode == "core2":
                # the index-based twin gaps do not apply (true L gates can lie after the scan centre): resolve a
                # mixed gate towards the side of its wire whose split is at a detected core... simply raise
                if twinsafe:
                    cross_r = sum(1 for j in W[w_after] if j in E and split[w_after] < j <= i and j in Rs)
                    cross_l = sum(1 for j in W[w_before] if j in E and i - 1 < j <= split[w_before] and j in Ls)
                    can_lower = direction.get(w_before, "down") == "down" and cross_l == 0
                    can_raise = direction.get(w_after, "up") == "up"
                    if cross_r > 0 and can_lower:
                        split[w_before] = i - 1
                        direction[w_before] = "down"
                        info["lowered"] += 1
                    elif can_raise or not can_lower:
                        if cross_r:
                            info["forced"] += 1
                        split[w_after] = i
                        direction[w_after] = "up"
                    else:
                        split[w_before] = i - 1
                        direction[w_before] = "down"
                        info["lowered"] += 1
                else:
                    split[w_after] = i
            elif i <= hi[w_after] or it >= 200:          # raise the later split: the gate becomes wholly L
                info["forced"] += int(i > hi[w_after])
                split[w_after] = i
            elif i - 1 >= lo[w_before]:                   # or lower the earlier one: the gate becomes wholly R
                split[w_before] = i - 1
            else:                                         # twin constraints on both wires disagree: exactness first
                info["forced"] += 1
                split[w_after] = i
            info["moved"] += 1
            moved = True
        if not moved:
            break
    return split, info


def _resid_rounds(M_, wire_at, cut, corr, dinfo, n, cutoff, log, centre):
    """Residual correction rounds, shared by the synchronous and the reference paths of measure_D.

    M_ = the measured operator as a syncd.SyncMPO in ONE frame (site s carries wire_at[s] on both legs); corr = the
    decode's ansatz (updated in place with the ("resid",) gates); dinfo gets resid_fid0/fid1/gates/rounds/secs."""
    import dresid
    import sectioned as _SEC
    _t = time.time()
    # ROUNDS: re-deriving R from the measured MPO beats continuing to peel the working copy,
    # whose bond cap and cutoff have already eaten some of the structure. MEASURED on k5 block
    # 3180: one round 0.294 -> 0.934, a second round (R rebuilt against the corrected ansatz)
    # 0.934 -> 0.955, while a converged block gains nothing and stops after one.
    _rounds = int(os.environ.get("HQP_RESID_ROUNDS", "4"))
    _min = float(os.environ.get("HQP_RESID_ROUND_GAIN", "2e-3"))
    # TIME BUDGET (HQP_RESID_MAX_S, default 600 s): the rounds had no bound; on a bond-1024 operator one build can take
    # many minutes and the sync attempt stayed silent long past its absorption budget (genadv h53 block 3020,
    # 2026-09-24: no result in 2471 s). Further rounds are skipped once the budget is spent; what was fitted so far
    # is kept (the trust gates read the last fidelity measured).
    _max_s = float(os.environ.get("HQP_RESID_MAX_S", "600"))
    _acc, _f0, _f1, _r = [], None, None, -1
    for _r in range(_rounds):
        if _r > 0 and time.time() - _t > _max_s:
            log(f"  resid: block {centre}: {time.time()-_t:.0f}s spent -> no further rounds (HQP_RESID_MAX_S {_max_s:g})")
            _r -= 1
            break
        _ans = _SEC._correction_gates(cut, corr, list(range(n)))
        _rg, _rinfo = dresid.build(M_, wire_at, _ans, dinfo.get("sigma") or {}, log=log,
                                   cutoff=cutoff)
        if _f0 is None:
            _f0 = _rinfo.get("fid0")
        # a round that declines to report (e.g. R could not be built reliably) must not erase
        # the fidelity the earlier rounds measured -- that would hide the block from the trust
        # condition entirely, which is the opposite of what the guard is for
        _prev = _f1
        if _rinfo.get("fid1") is not None:
            _f1 = _rinfo["fid1"]
        if not _rg:
            break
        _acc = _acc + list(_rg)
        corr[("resid",)] = _acc
        if _prev is not None and _f1 is not None and _f1 - _prev < _min:
            break
    dinfo["resid_fid0"] = _f0
    dinfo["resid_fid1"] = _f1
    dinfo["resid_gates"] = len(_acc)
    dinfo["resid_rounds"] = max(_r + 1, 0)
    dinfo["resid_secs"] = time.time() - _t
    if _acc:
        log(f"  resid: block {centre}: |<I,R>| {_f0:.6f} -> {_f1:.6f} over {_r + 1} round(s), "
            f"{len(_acc)} gates ({sum(1 for g in _acc if g[0] == 'cz')} CZ)")
    return _acc, _f0, _f1


def measure_D(gates, n, centre, tau, window, to_backend, log, budget=None, cutoff=None, seed=123,
              cut_override=None):
    """Measure D = Pi_tau^-1 . E for the cut at `centre`, and gate on its quality.

    Returns (cut, correction, info) or (None, None, info) if the block must not be excised.
    """
    import torch
    import fp32_patch  # noqa: F401  robust SVD/QR cascade -- must precede any quimb/unswap use
    import excise as X
    from qiskit import QuantumCircuit
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
    from qiskit.transpiler import PassManager
    budget = D_BUDGET if budget is None else budget
    cutoff = D_CUTOFF if cutoff is None else cutoff
    info = {"centre": centre}
    # cut_override=(cut, twins): measure D for a cut built elsewhere (e.g. the whole-block SKELETON cut)
    cut, tw = cut_override if cut_override is not None else X.envelope_cut(gates, n, centre, tau, window=window, log=log)
    audit = X.audit(cut, tw, log=log)
    info["audit"] = audit
    if audit["orphans"] or audit["inconsistent"] or audit["untouched"]:
        log(f"  excise: block {centre} fails the cut audit -> not excised")
        return None, None, info
    W, pos = X.wire_ops(gates, n)
    Eidx = sorted({W[x][k] for x in range(n) for k in range(cut.s[x], cut.e[x] + 1)})
    qc = QuantumCircuit(n)
    n_before = sum(1 for i in Eidx if i < centre)
    npairs = sum(1 for x in range(n) if x < tau[x])
    involution = all(tau[tau[x]] == x for x in range(n))
    # default "centre" since 2026-09-19: MEASURED on d3_s1 block 3120 (380-CZ cut, cutoff 6e-4, seed 123, same GPU
    # sharing): swap at the END -> model err 0.372, unitarity 0.00, gate FAILED after 4365 s; swap at the
    # CENTRE -> model err 0.034, unitarity 0.98, 3-body 0.04, 2391 s. "end" keeps the pre-v13 behaviour.
    swap_at = os.environ.get("HQP_D_SWAP_AT", "centre").strip().lower()
    # HQP_D_SPLIT (reference absorber): where the two halves meet. "index" (default, the validated v13 behaviour) = index
    # split + float ratio over a globally consolidated circuit; "clip"/"core"/"core2" = per-wire split with an EXACT
    # integer start index (per-section consolidation) -- NOT yet A/B-measured on the reference absorber.
    split_mode = os.environ.get("HQP_D_SPLIT", "index").strip().lower()      # reference absorber: the validated historical split
    _engine0 = os.environ.get("HQP_D_ENGINE", "ref").strip().lower()
    if _engine0 in ("sync", "sync_only"):
        # the synchronous engine needs the TRUE mirror surface (per-wire split at the pair cores); the reference
        # absorber's own path is unaffected by this choice when it is called directly with HQP_D_ENGINE=ref
        split_mode = os.environ.get("HQP_D_SYNC_SPLIT", "core2").strip().lower()
    exact_c = None
    if swap_at == "centre" and involution:
        # SWAP_tau IN THE MIDDLE, right half relabelled:  D = Pi.E = (Pi R Pi).Pi.L  (Pi an involution) -- the same
        # operator, but the absorber now starts ON the explicit swaps and every mirrored pair sits on the SAME
        # wire labels, so it never has to DISCOVER the hidden permutation through half-absorbed SWAP gadgets
        # (the slow bond-256 thrash of the first ~100 unitaries, and the lossiest part of the measurement).
        split, sinfo = wire_split(gates, n, cut, tw, tau, centre, Eidx, mode=split_mode, log=log)
        is_left = lambda i: i <= split[gates[i][1][0]]                   # noqa: E731  (consistent on both wires)
        n_left = sum(1 for i in Eidx if is_left(i))
        info["split"] = dict(sinfo, n_left=n_left, n_right=len(Eidx) - n_left)
        if split_mode != "index":
            log(f"  excise: block {centre} split '{split_mode}': {n_left} L / {len(Eidx) - n_left} R gates "
                f"(index split {n_before} / {len(Eidx) - n_before}), clipped wires {sinfo['clipped']}, "
                f"moved {sinfo['moved']}, forced {sinfo['forced']}, bad gaps {sinfo['bad_gap']}")
        for i in Eidx:
            if is_left(i):
                t, q, p = gates[i]
                qc.cz(*q) if t == "cz" else qc.u(*p, q[0])
        for x in range(n):
            if x < tau[x]:
                qc.swap(x, tau[x])
        for i in Eidx:
            if not is_left(i):
                t, q, p = gates[i]
                qc.cz(tau[q[0]], tau[q[1]]) if t == "cz" else qc.u(*p, tau[q[0]])
        ratio = (n_left + npairs / 2.0) / max(len(Eidx) + npairs, 1)
        if split_mode != "index":
            exact_c = n_left                                             # raw-gate junction L | SWAPs+R
    else:
        for i in Eidx:
            t, q, p = gates[i]
            qc.cz(*q) if t == "cz" else qc.u(*p, q[0])
        for x in range(n):
            if x < tau[x]:
                qc.swap(x, tau[x])
        ratio = n_before / max(len(Eidx) + npairs, 1)
    info["swap_at"] = swap_at if involution else "end"
    # HQP_D_ENGINE: "sync" = synchronous two-sided absorber (syncd.py: one routing shared by both legs, twin gates in
    # lockstep, CPU complex128) with the reference absorber as the fallback; "ref" = reference absorber only.
    engine = os.environ.get("HQP_D_ENGINE", "ref").strip().lower()
    TS = None
    t0 = time.time()
    if engine in ("sync", "sync_only") and swap_at == "centre" and involution:
        try:
            import syncd
            sync_budget = min(budget, float(os.environ.get("SYNCD_BUDGET_S", "900")))
            M_, wire_at, sinfo2 = syncd.absorb(gates, n, tau, Eidx, split, tw, cutoff=cutoff, seed=seed, log=log,
                                               deadline=time.time() + sync_budget)
            # FINAL-BOND HANDOFF (SYNCD_DECODE_MAX_BOND, default 128): every block the synchronous engine ever decoded
            # correctly came out of the absorption with a final bond <= 64 (real samples 1-8, k10's corrected block 32-64,
            # the rescued s160 block 3-4). MEASURED 2026-09-24 on genadv h53 block 3020 (right tau, SWAP cores spread over
            # half the block): the lockstep left 38 insertions open, final bond 235, and the DECODE of that operator ran
            # 1350-1615 s single-threaded before failing structurally -- which, being "structural", also suppressed the
            # reference worker. A high final bond means the SCHEDULE could not cancel the block, not that the cut is
            # wrong: hand the block to the reference absorber at once (as a "gave no D", so the parent launches it).
            _fb = int((sinfo2 or {}).get("final_bond") or 0)
            _fb_cap = int(os.environ.get("SYNCD_DECODE_MAX_BOND", "128"))
            if _fb > _fb_cap:
                info["engine"] = "sync"
                info["sync"] = sinfo2
                info["sync_failed"] = f"final bond {_fb} > {_fb_cap}: the lockstep could not cancel this block"
                log(f"  excise: block {centre} sync engine: final bond {_fb} > {_fb_cap} after absorption -> reference absorber")
                if engine == "sync_only":
                    return None, None, info
                raise RuntimeError(info["sync_failed"])
            TS = syncd.to_TS(M_, n)
            siteR = {i: wire_at[i] for i in range(n)}
            siteL = dict(siteR)
            pR = pL = [wire_at.index(w) for w in range(n)]
            info["engine"] = "sync"
            info["sync"] = sinfo2
        except Exception as e_:                                   # noqa: BLE001
            TS = None
            info["sync_failed"] = f"{type(e_).__name__}: {str(e_)[:120]}"
            log(f"  excise: block {centre} sync engine failed ({info['sync_failed']}) -> reference absorber")
    if TS is None and engine == "sync_only":
        info.setdefault("sync_failed", "not applicable (tau is not an involution or the swaps are not at the centre)")
        info["engine"] = "sync_only"
        return None, None, info
    if TS is None and engine == "sync":
        # fall back to the reference absorber EXACTLY as validated (its own split and consolidation), not on the
        # circuit built for the synchronous engine
        _prev = os.environ.get("HQP_D_ENGINE")
        os.environ["HQP_D_ENGINE"] = "ref"
        try:
            return measure_D(gates, n, centre, tau, window, to_backend, log, budget=budget, cutoff=cutoff, seed=seed,
                             cut_override=cut_override)
        finally:
            if _prev is None:
                os.environ.pop("HQP_D_ENGINE", None)
            else:
                os.environ["HQP_D_ENGINE"] = _prev
    if TS is None:
        info["engine"] = "ref"
        if exact_c is not None:
            # consolidate each half on its own: a global Collect2qBlocks pass re-emits the circuit in a lexicographic
            # topological order, in which no instruction index is the mirror surface any more
            circ, _idx = _consolidate_sections(qc, [exact_c])
            ratio = int(_idx[0])
            info["split"]["C"] = ratio
            info["split"]["blocks"] = len(circ.data)
        else:
            circ = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(qc)
        t0 = time.time()
        mpo_compress_unswap = _absorber()
        mpo, ll, lr, _ = mpo_compress_unswap(
            circ, seed=seed, to_backend=to_backend, cutoff=cutoff, max_bond=D_MAXBOND,
            unswap_threshold=1e6, center_ratio=ratio, equal=False, flip_freq=None, max_its=20,
            early_stopping_gates=0, hows=("both", "left", "right"),
            deadline=time.time() + budget, allow_abandon=False, adapt_stop=False)
        left_ops = sum(sum(v for k, v in lay.count_ops().items() if k not in ("measure", "barrier")) for lay in ll)
        right_ops = sum(sum(v for k, v in lay.count_ops().items() if k not in ("measure", "barrier")) for lay in lr)
        if left_ops or right_ops:
            log(f"  excise: block {centre} D absorption INCOMPLETE (leftover {left_ops}/{right_ops}) -> not excised")
            return None, None, info

        def frame(layers):
            meas = [lay for lay in layers if "measure" in dict(lay.count_ops())]
            perm = list(range(n))
            if not meas:
                return perm
            for ins in meas[-1]:
                if ins.operation.name == "measure":
                    perm[ins.clbits[0]._index] = ins.qubits[0]._index
            return perm
        pR, pL = frame(lr), frame(ll)
        siteR = {pR[w]: w for w in range(n)}
        siteL = {pL[w]: w for w in range(n)}
        TS = []
        for i in range(n):
            t = mpo[mpo.site_tag(i)]
            if isinstance(t, tuple):
                t = t[0]
            d = t.data if torch.is_tensor(t.data) else torch.tensor(np.asarray(t.data), device="cuda")
            TS.append((d.to(torch.complex128), list(t.inds)))

    def amp(outv, inv_):
        env = None; ei = None
        for i in range(n):
            d, inds = TS[i]
            vo = torch.tensor(outv[siteR[i]], dtype=torch.complex128, device=d.device)
            vi = torch.tensor(inv_[siteL[i]], dtype=torch.complex128, device=d.device)
            A = torch.tensordot(d, vo.conj(), dims=([inds.index(f"k{i}")], [0]))
            i2 = [x for x in inds if x != f"k{i}"]
            A = torch.tensordot(A, vi, dims=([i2.index(f"b{i}")], [0]))
            i3 = [x for x in i2 if x != f"b{i}"]
            if env is None:
                env, ei = A, i3
            else:
                sh = [x for x in ei if x in i3]
                env = torch.tensordot(env, A, dims=([ei.index(s) for s in sh], [i3.index(s) for s in sh]))
                ei = [x for x in ei if x not in sh] + [x for x in i3 if x not in sh]
        return complex(env.reshape(-1)[0])
    Z = np.array([1, 0], complex); O = np.array([0, 1], complex)
    bas = lambda ones: [O if w in ones else Z for w in range(n)]
    if os.environ.get("D_DECODE", "ref").strip().lower() != "legacy":
        # REFERENCE-FREE decode (d_decode.py). The legacy decode below reads everything against <0..0|D|0..0>, which
        # is EXACTLY zero as soon as one wire carries an X-type 1q block: MEASURED on d3_s1 block 1435 (628 CZ) the
        # absorbed operator was product-like (bond 2-16, 10^-0.10 of the norm lost) and was still rejected with
        # "unitary 0.00, F0 0.000". Reference = dominant output string of D|0..0>, from its MPS marginals.
        import d_decode
        mps = []
        for i in range(n):
            d, inds = TS[i]
            v0 = torch.tensor(Z, dtype=torch.complex128, device=d.device)
            A = torch.tensordot(d, v0, dims=([inds.index(f"b{i}")], [0]))
            ai = [x for x in inds if x != f"b{i}"]
            left = [x for x in ai if i > 0 and x in TS[i - 1][1]]
            right = [x for x in ai if i < n - 1 and x in TS[i + 1][1]]
            perm = [ai.index(x) for x in left] + [ai.index(f"k{i}")] + [ai.index(x) for x in right]
            A = A.permute(*perm)
            Dl = int(np.prod([A.shape[j] for j in range(len(left))])) if left else 1
            Dr = int(np.prod([A.shape[len(left) + 1 + j] for j in range(len(right))])) if right else 1
            mps.append(A.reshape(Dl, 2, Dr).cpu().numpy())
        site_bits = d_decode.dominant_output(mps)
        ref_out = [0] * n
        for i in range(n):
            ref_out[siteR[i]] = site_bits[i]            # site i's ket index is output wire siteR[i]
        amp_bits = lambda xo, xi: amp([O if xo[w] else Z for w in range(n)], [O if xi[w] else Z for w in range(n)])   # noqa: E731
        if info.get("engine") == "sync":
            # basis-state amplitudes straight from the numpy site tensors: MEASURED, the generic torch.tensordot path
            # costs 0.155 s per amplitude on the CPU without thread caps (230k tiny tensordots = 250 s per decode)
            _T = [np.asarray(M_.T[s_]) * math.sqrt(2.0) for s_ in range(n)]
            _pos = {wire_at[s_]: s_ for s_ in range(n)}

            def amp_bits(xo, xi):                                  # noqa: F811
                v = _T[0][:, xo[wire_at[0]], xi[wire_at[0]], :]
                for s_ in range(1, n):
                    v = v @ _T[s_][:, xo[wire_at[s_]], xi[wire_at[s_]], :]
                return complex(v[0, 0])
        corr, dinfo = d_decode.decode(amp_bits, n, tau, ref_out, log=log, centre=centre)
        # The decode's ansatz (1q blocks + DIAGONAL two-body phases) is a LOSSY projection of the
        # measured operator: MEASURED 2026-09-21, it keeps 0.984-1.000 of D on blocks the pipeline gets
        # right and only 0.294 on the one block that makes d3s1_k5 answer wrong. Extract what it drops
        # as a few extra two-qubit gates; the rest of the pipeline just sees a longer gate list.
        if info.get("engine") == "sync" and os.environ.get("HQP_D_RESID", "1") != "0":
            try:
                _resid_rounds(M_, wire_at, cut, corr, dinfo, n, cutoff, log, centre)
            except Exception as e_:                                # noqa: BLE001
                # never let this stage break a measurement that already succeeded
                log(f"  excise: block {centre} residual correction failed "
                    f"({type(e_).__name__}: {str(e_)[:110]}) -> ansatz only")
                dinfo["resid_failed"] = f"{type(e_).__name__}"
        elif (info.get("engine") == "ref" and os.environ.get("HQP_D_RESID", "1") != "0"
              and os.environ.get("HQP_D_RESID_REF", "1") != "0"):
            # REFERENCE-PATH residual correction (HQP_D_RESID_REF=1, 0 disables). The reference absorber is the
            # fallback that decides exactly the hard blocks (the sync engine gave up on them), and until now its
            # decode was ansatz-only: a hard block's D was emitted at 0.2-0.4 overlap and the answer was wrong (the
            # 2026-09-24 validator failure). Its operator carries TWO wire frames (siteR on the output legs, siteL on
            # the input legs; MEASURED they differ on 5-37 wires in 11 of 12 saved operators), so before the SAME
            # rounds loop as the sync path the input legs are sorted into the output frame with one-leg SWAPs
            # (ref_align.py). The realignment is a truncated operation: if it discards more than HQP_RESID_MAX_LOST
            # the correction is refused, exactly as dresid refuses an unreliable R.
            try:
                import ref_align
                _t = time.time()
                M_, wire_at, _ainfo = ref_align.from_frames(TS, n, siteR, siteL, log=log)
                dinfo["realign"] = {k_: v_ for k_, v_ in _ainfo.items()}
                _max_lost = float(os.environ.get("HQP_RESID_MAX_LOST", "1e-3"))
                if _ainfo["lost"] > _max_lost:
                    log(f"  excise: block {centre} realignment discarded {_ainfo['lost']:.2e} (> {_max_lost:g}, bond "
                        f"{_ainfo['bond_before']} -> {_ainfo['bond_after']}) -> no residual correction")
                    dinfo["resid_failed"] = "realign truncation"
                else:
                    _resid_rounds(M_, wire_at, cut, corr, dinfo, n, cutoff, log, centre)
                dinfo["realign"]["total_secs"] = time.time() - _t
            except Exception as e_:                                # noqa: BLE001
                log(f"  excise: block {centre} reference-path residual correction failed "
                    f"({type(e_).__name__}: {str(e_)[:110]}) -> ansatz only")
                dinfo["resid_failed"] = f"{type(e_).__name__}"
        unitary_ok = dinfo["unitary"] > float(os.environ.get("D_MIN_UNITARY", "0.7"))
        info.update(dinfo)
        info.update({"unitary_ok": unitary_ok, "cz_excised": audit["cz"], "secs": time.time() - t0, "cutoff": cutoff,
                     "seed": seed, "decode": "ref"})
        log(f"  excise: block {centre} D structure: {dinfo['two_on_pair']} pi-terms on tau-pairs, {dinfo['two_off_pair']} other "
            f"two-body, non-diagonal 1q blocks on wires {dinfo['nondiag_wires']}, F0={dinfo['F0']:.3f}, "
            f"reference |amp|^2={dinfo['Fref']:.3f} ({dinfo['ref_ones']} flipped outputs), phase side '{dinfo['phase_side']}'")
        ok = unitary_ok and dinfo["three_body"] <= MAX_3BODY and dinfo["model_err"] <= MAX_MODEL_ERR
        # RESCUE (HQP_RESID_RESCUE=0.99, 0 disables): the gate above judges the product ANSATZ, but what the
        # readout emits is the ansatz PLUS the residual correction. MEASURED on the exact n=10 control with a
        # full random SU(4) slipped into the cut (clean/t_e2e_resid2.py, strict D_MIN_UNITARY): the rescue takes
        # the measurable instances from 3/12 to 10/12 and every one of the 8 corrections is BETTER, none worse --
        # blocks the gate rejects at model err 0.12-0.76 correct to |<I,R>| 1.00000 and reduce to L1 3e-15..5e-3.
        # The two it still refuses are exactly the two whose correction only reaches 0.78-0.81 and which stay
        # wrong (L1 ~0.6), so the post-correction overlap separates them cleanly at this threshold. Without the
        # rescue such a block is simply not excised, v13 declines, and the legacy stages answer -- so the
        # downside is bounded by what we would have done anyway.
        _resc = float(os.environ.get("HQP_RESID_RESCUE", "0.99"))
        # NEVER rescue a STRUCTURAL failure (the parent's own criteria: unitarity < 0.5 or model err > 0.9): MEASURED
        # 2026-09-24 on h35 block 3460 with a wrong tau (reference path, validator flags): the absorbed operator had
        # unitarity 0.00 and model err 1.000 -- not an operator at all -- yet the residual fit "reproduced" it to
        # |<I,R>| 0.9996 (93 gates) and the block was excised with D err 1.0; the other view then re-used the file and
        # the two views "agreed". A rescue is only meaningful for a genuine (near-)unitary D whose product ANSATZ is
        # poor (the validated cases: model err 0.12-0.76, unitary operators with a slipped-in SU(4)).
        # NOTE (2026-09-24 15:30): dinfo["unitary"] is the singular-value ratio of the FITTED per-wire factors -- an
        # ansatz-quality number that reads 0.00 whenever the product ansatz is noise (w20 block 3200: unitary 0.00,
        # model err 0.4, and its rescue to 0.9987 is CORRECT) -- so it must not gate the rescue: MIN_UNITARY defaults
        # to 0 (off); only the model-error criterion is kept (no validated rescue exceeds 0.76).
        _min_uni = float(os.environ.get("HQP_RESID_RESCUE_MIN_UNITARY", "0"))
        # MEASURED 2026-09-24 (h35 3460, reference path, wrong tau, validator flags): model err 1.000, overlap 0.9996,
        # answer RIGHT (hamming 0, z 9.0) -> no model-error criterion either; the overlap >= HQP_RESID_RESCUE is the gate.
        _max_err = float(os.environ.get("HQP_RESID_RESCUE_MAX_MODEL_ERR", "2"))
        _structural = dinfo["unitary"] < _min_uni or dinfo["model_err"] > _max_err
        if not ok and _structural and _resc > 0 and (dinfo.get("resid_fid1") or 0.0) >= _resc:
            log(f"  excise: block {centre}: STRUCTURAL failure (unitary {dinfo['unitary']:.2f}, model err {dinfo['model_err']:.3f}) "
                f"-- the residual fit's overlap {dinfo['resid_fid1']:.4f} is not evidence on a non-operator -> NOT rescued")
        if not ok and not _structural and _resc > 0 and (dinfo.get("resid_fid1") or 0.0) >= _resc:
            log(f"  excise: block {centre} fails the ansatz gate (unitary {dinfo['unitary']:.2f}, model err "
                f"{dinfo['model_err']:.3f}) but the EMITTED D reproduces the measured one to "
                f"{dinfo['resid_fid1']:.4f} >= {_resc} -> excised anyway")
            ok = True
        log(f"  excise: block {centre}: {audit['cz']} CZ, D model err {dinfo['model_err']:.3f}, 3-body {dinfo['three_body']:.1e}, "
            f"unitary {dinfo['unitary']:.2f} -> {'EXCISE' if ok else 'keep (gate failed)'} [{time.time()-t0:.0f}s]")
        if os.environ.get("D_SAVE_MPO"):                       # experiments: keep the absorbed operator for re-decoding
            try:
                import pickle as _pk
                _pk.dump({"TS": [(d_.cpu().numpy(), list(i_)) for d_, i_ in TS], "pR": pR, "pL": pL, "tau": list(tau)},
                         open(os.path.join(os.environ["D_SAVE_MPO"], f"mpo_{centre}_{seed}.pkl"), "wb"))
            except Exception as e_:                               # noqa: BLE001
                log(f"  excise: could not save the MPO ({type(e_).__name__})")
        return (cut, corr, info) if ok else (None, None, info)
    Aw = {}; sing = {}
    for w in range(n):
        M = np.array([[amp(bas(set()), bas(set())), amp(bas(set()), bas({w}))],
                      [amp(bas({w}), bas(set())), amp(bas({w}), bas({w}))]], complex)
        U_, s_, Vh_ = np.linalg.svd(M)
        sing[w] = (float(s_.max()), float(s_.min()))
        Aw[w] = U_ @ Vh_
    model = lambda xo, xi: np.prod([Aw[w][xo[w], xi[w]] for w in range(n)])
    base_m = amp(bas(set()), bas(set())); base_p = model([0] * n, [0] * n)
    scale = base_m / base_p if abs(base_p) > 1e-12 else 1.0
    two = []
    for a_, b_ in itertools.combinations(range(n), 2):
        xo = [0] * n; xo[a_] = 1; xo[b_] = 1
        m = amp(bas({a_, b_}), bas({a_, b_})); pm = model(xo, xo) * scale
        if abs(m) > 1e-14 and abs(pm) > 1e-14:
            ph = float(np.angle(m / pm))
            if abs(ph) > 0.05:
                two.append((a_, b_, ph))
    rng = np.random.default_rng(0); worst3 = 0.0
    for _ in range(120):
        a_, b_, c_ = sorted(rng.choice(n, 3, replace=False))
        xo = [0] * n
        for q in (a_, b_, c_): xo[q] = 1
        m = amp(bas({a_, b_, c_}), bas({a_, b_, c_}))
        pm = model(xo, xo) * scale * np.exp(1j * sum(p for (x_, y_, p) in two if {x_, y_} <= {a_, b_, c_}))
        if abs(m) > 1e-12 and abs(pm) > 1e-12:
            worst3 = max(worst3, abs(np.angle(m / pm)))
    errs = []
    for _ in range(40):
        xi = [int(rng.random() < 0.3) for _ in range(n)]
        xo = list(xi)
        if rng.random() < 0.5:
            f = int(rng.integers(n)); xo[f] = 1 - xo[f]
        m = amp(bas({w for w in range(n) if xo[w]}), bas({w for w in range(n) if xi[w]}))
        pm = model(xo, xi) * scale * np.exp(1j * sum(p for (x_, y_, p) in two if xi[x_] and xi[y_]))
        errs.append(abs(m - pm) / max(abs(m), abs(pm), 1e-15))
    med_err = float(np.median(errs))
    # WHICH SIDE DO THE TWO-BODY PHASES SIT ON?  The model above keys them on the INPUT bits (D = A . Phi:
    # phases first in time, then the 1q blocks); the emitters historically wrote the 1q blocks FIRST. The two
    # differ exactly when a wire of a phase pair carries a NON-diagonal 1q block (an X-type A pushes a Z onto
    # its partner), which the generic error sample above almost never probes. Measure both conventions on
    # elements that flip such a wire and record the better one; the emitters follow info["phase_side"].
    pair_wires = sorted({w for (x_, y_, _p) in two for w in (x_, y_)})
    hot = [w for w in pair_wires if max(abs(Aw[w][0, 1]), abs(Aw[w][1, 0])) > 0.1]
    e_in, e_out = [], []
    for w in hot:
        partners = [y_ if x_ == w else x_ for (x_, y_, _p) in two if w in (x_, y_)]
        for _rep in range(6):
            xi = [int(rng.random() < 0.3) for _ in range(n)]
            for pw in partners:
                xi[pw] = 1
            xo = list(xi); xo[w] = 1 - xo[w]
            m = amp(bas({q for q in range(n) if xo[q]}), bas({q for q in range(n) if xi[q]}))
            base_pm = model(xo, xi) * scale
            pm_in = base_pm * np.exp(1j * sum(p_ for (x_, y_, p_) in two if xi[x_] and xi[y_]))
            pm_out = base_pm * np.exp(1j * sum(p_ for (x_, y_, p_) in two if xo[x_] and xo[y_]))
            den = max(abs(m), abs(base_pm), 1e-15)
            e_in.append(abs(m - pm_in) / den); e_out.append(abs(m - pm_out) / den)
    phase_side = "in"
    if e_in and float(np.median(e_out)) < float(np.median(e_in)):
        phase_side = "out"
    info["phase_side"] = phase_side
    info["phase_side_errs"] = (float(np.median(e_in)) if e_in else None, float(np.median(e_out)) if e_out else None, hot)
    if hot:
        log(f"  excise: block {centre} phase side: non-diagonal 1q block on phase-pair wires {hot}; model err with phases on "
            f"input {info['phase_side_errs'][0]:.3f} / on output {info['phase_side_errs'][1]:.3f} -> '{phase_side}'")
    worst_uni = min(s[1] / max(s[0], 1e-30) for s in sing.values())
    # a mildly rank-deficient wire is survivable (d3_s2 B1 sat at 0.83 and still solved); only a
    # badly singular block means the cut is really broken
    unitary_ok = worst_uni > float(os.environ.get("D_MIN_UNITARY", "0.7"))
    # structural diagnostics (logged, not gated): every D that solved had its pi-terms on tau-pairs
    # and at most 1-2 wires with a non-diagonal block. Recorded so a wrong-but-product-like D
    # can be recognised after the fact.
    on_pair = sum(1 for a_, b_, ph in two if tau[a_] == b_ and abs(abs(ph) - math.pi) < 0.05)
    off_pair = len(two) - on_pair
    nondiag = [w for w in range(n) if max(abs(Aw[w][0, 1]), abs(Aw[w][1, 0])) > 0.1]
    info.update({"model_err": med_err, "three_body": worst3, "unitary_ok": unitary_ok,
                 "cz_excised": audit["cz"], "secs": time.time() - t0,
                 "two_on_pair": on_pair, "two_off_pair": off_pair, "nondiag_wires": nondiag,
                 "F0": float(abs(base_m) ** 2), "cutoff": cutoff, "seed": seed})
    log(f"  excise: block {centre} D structure: {on_pair} pi-terms on tau-pairs, {off_pair} other two-body, "
        f"non-diagonal 1q blocks on wires {nondiag}, F0={abs(base_m)**2:.3f}")
    ok = unitary_ok and worst3 <= MAX_3BODY and med_err <= MAX_MODEL_ERR
    log(f"  excise: block {centre}: {audit['cz']} CZ, D model err {med_err:.3f}, 3-body {worst3:.1e}, "
        f"unitary {worst_uni:.2f} -> {'EXCISE' if ok else 'keep (gate failed)'} [{time.time()-t0:.0f}s]")
    if not ok:
        return None, None, info
    corr = {}
    corr[("phase_side",)] = phase_side
    for w in range(n):
        corr[("A", w)] = [[Aw[w][i][j].real, Aw[w][i][j].imag] for i in range(2) for j in range(2)]
    for a_, b_, ph in two:
        corr[("cz", a_, b_)] = None if abs(abs(ph) - math.pi) < 1e-3 else float(ph)
    return cut, corr, info


def solve_excision(circ, log, to_backend, deadline=None):
    """Full stage: discover blocks, measure each D, rebuild, solve with the canonical ladder.

    Returns (bits, info) with bits=None when the stage declines (caller falls back)."""
    import excise as X
    import mps_torch
    from qiskit import QuantumCircuit
    n = circ.num_qubits
    info = {"method": "excision", "n_qubits": n, "confident": False, "trusted": False}
    if not EXCISE_ON or n < MIN_Q:
        return None, info
    try:
        gates = _circ_to_gates(circ)
    except ValueError as e:
        log(f"excision: {e}")
        return None, info
    t0 = time.time()
    found = X.discover_centres(gates, n, log=lambda m: log("  " + m[:160]))
    if not found:
        log("excision: no mirror blocks discovered -> fall back")
        return None, info
    half = int(os.environ.get("EXC_TAU_HALF", "600"))
    blocks = []
    for c, _, _ in found:
        tau, _, fr = X.discover_tau(gates, n, c, half)
        blocks.append((c, tau, fr))
    cs = [b[0] for b in blocks]
    log(f"excision: {len(blocks)} blocks at {cs} (twin fractions {[round(b[2],2) for b in blocks]}), "
        f"discovery {time.time()-t0:.1f}s")
    info["blocks_found"] = len(blocks)
    max_rungs = n_rungs(len(blocks))
    log(f"excision: draw schedule {max_rungs} rungs/block "
        f"{[(f'{c:g}', sd) for c, sd in ATTEMPTS[:max_rungs][:6]]}"
        f"{' ...' if max_rungs > 6 else ''} (time-bounded)")
    # ---- measurement state: one D per block, plus which ladder rung produced it ----
    Dstate = {c: None for c, _, _ in blocks}          # centre -> (cut, corr, binfo)
    rung = {c: 0 for c, _, _ in blocks}                # next ATTEMPTS index to try per block
    total_cz = sum(1 for t, _, _ in gates if t == "cz")
    excise_deadline = None
    if deadline is not None:
        # leave the rest of the wall to the fallbacks (d2-shaped circuits still need unswap)
        excise_deadline = time.time() + MAX_FRAC * (deadline - time.time())

    def left():
        return (excise_deadline - time.time()) if excise_deadline is not None else float("inf")

    def measure_block(c, tau, window, share=1):
        """One ladder rung for block c. Returns True if it now has a passing D.

        `share` = how many blocks still have to be paid for out of the remaining time. A rung that
        grabs the whole window starves every later block, and a circuit excised at only SOME of its
        mirror blocks still contains a whole block -> the chi ladder then reads a truncation argmax.
        """
        if rung[c] >= max_rungs:
            return False
        cut_off, sd = attempt_at(rung[c]); rung[c] += 1
        budget = min(D_BUDGET, max(D_BUDGET_MIN, (left() - CANON_RESERVE) / max(share, 1)))
        log(f"  excise: block {c} rung {rung[c]}/{max_rungs} (cutoff {cut_off:g}, seed {sd}), "
            f"budget {budget:.0f}s, {left():.0f}s of excision time left")
        try:
            cut, corr, binfo = measure_D(gates, n, c, tau, window, to_backend, log,
                                         budget=budget, cutoff=cut_off, seed=sd)
        except Exception as e:  # noqa: BLE001  one failed attempt must not abort the whole stage
            log(f"  excise: block {c} rung {rung[c]} FAILED ({type(e).__name__}: {str(e)[:120]})")
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
            return False
        if cut is not None:
            prev_d = Dstate[c]
            # Under consensus voting every PASSING draw is taken, even one that fits worse than the
            # incumbent: it already cleared the quality gate, and it is an INDEPENDENT sample, which
            # is the whole point of voting. Keeping the incumbent would re-run the ladder on
            # identical input and produce no new vote.
            if (CONSENSUS <= 1 and prev_d is not None
                    and prev_d[2].get("model_err", 9) <= binfo.get("model_err", 9)):
                log(f"  excise: block {c} kept the earlier D (model err "
                    f"{prev_d[2].get('model_err', -1):.3f} <= {binfo.get('model_err', -1):.3f})")
                return False        # nothing improved -> the caller should try elsewhere
            Dstate[c] = (cut, corr, binfo)
            return True
        return False

    def ladder():
        """Rebuild from the current D's and run the chi ladder.

        Returns (logical, margin, chi, w0, trusted, used, ncz, order_verdicts); the verdict dict is
        what votes -- one vote per gate order that actually reached a stable, above-floor answer.
        """
        cuts = [Dstate[c][0] for c, _, _ in blocks if Dstate[c]]
        corrs = [Dstate[c][1] for c, _, _ in blocks if Dstate[c]]
        used = [c for c, _, _ in blocks if Dstate[c]]
        red, Pinv, dropped = X.reduce_multi(gates, n, cuts, corrections=corrs)
        ncz = sum(1 for t, _, _ in red if t == "cz")
        log(f"excision: excised {used}; reduced {len(gates)}->{len(red)} gates, CZ {total_cz}->{ncz}")
        rq_raw = QuantumCircuit(n)
        for t, q, p in red:
            rq_raw.cz(*q) if t == "cz" else rq_raw.u(*p, q[0])
        # Truncated MPS evolution is gate-order sensitive and the better order is circuit-dependent
        # (E84: DAG order finds d3_s1's peak at every rung where insertion order needs chi 1024;
        # insertion order gives d3_s2 margins 1.95 vs 1.71). 'dag' = the order every validated run
        # used (qasm2.load + remove_final_measurements = a DAG rebuild); 'raw' = insertion order.
        from qiskit.converters import circuit_to_dag, dag_to_circuit
        variants = {"dag": lambda: dag_to_circuit(circuit_to_dag(rq_raw)), "raw": lambda: rq_raw}
        rung_results = []; out_of_time = False; order_verdicts = {}
        for order in ORDERS:
            if out_of_time:
                break
            if order not in variants:
                log(f"  excise: unknown gate order '{order}' ignored"); continue
            rq = variants[order]()
            prev = None
            prev_dt = None
            for chi in CHIS:
                # Admit a rung only if it can FINISH. A flat 120 s floor let a chi-1024 rung (~17 min
                # measured on 48q) start with two minutes left: it then overran the excision
                # deadline by a quarter of an hour, produced nothing, and squeezed the fallbacks.
                # mps_torch.evolve has no internal clock check, so this is the only place to stop it.
                need = max(120.0, (prev_dt or 0.0) * RUNG_GROWTH)
                if left() < need:
                    log(f"excision: next chi rung needs ~{need:.0f}s but only {left():.0f}s "
                        f"of excision time remains -> stop ladder")
                    out_of_time = True; break
                try:
                    mps, dt = mps_torch.evolve(rq, chi, log_every=0)
                    prev_dt = dt
                    cands = mps_torch.topk(mps, beam=int(os.environ.get("HQP_EXCISE_BEAM", "512")), k=8)
                    del mps
                except Exception as e:  # noqa: BLE001  (OOM at a high rung must not lose the lower rungs)
                    log(f"  excise chi={chi}: FAILED ({type(e).__name__}: {str(e)[:100]}) -> stop ladder")
                    try:
                        import torch; torch.cuda.empty_cache()
                    except Exception:
                        pass
                    break
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception:
                    pass
                top0, w0 = cands[0]
                w1 = cands[1][1] if len(cands) > 1 else 0.0
                if not (w0 == w0 and w1 == w1):          # NaN weights are not a result
                    log(f"  excise chi={chi}: NaN beam weights -> rung discarded"); prev = None; continue
                margin = w0 / max(w1, 1e-300)
                stable = top0 == prev
                logical = X.map_bits_to_original(top0, Pinv)
                log(f"  excise [{order}] chi={chi}: {dt:.0f}s w0={w0:.3e} margin={margin:.2f} stable={stable} top={logical[:24]}...")
                rung_results.append((stable, chi, margin, logical, w0))
                if stable and margin >= MARGIN and order not in order_verdicts:
                    order_verdicts[order] = (logical, margin, chi, w0)
                    log(f"  excise: gate order '{order}' proposes {logical[:24]}... (margin {margin:.2f})")
                    break          # this order has its answer; cross-check it against the next order
                prev = top0
        # --- decide, using agreement between gate orders as the real trust signal ---
        if order_verdicts:
            answers = {v[0] for v in order_verdicts.values()}
            if len(order_verdicts) >= 2 and len(answers) == 1:
                logical, margin, chi, w0 = max(order_verdicts.values(), key=lambda v: v[1])
                log(f"  excise: {len(order_verdicts)} gate orders AGREE -> trust (margin {margin:.2f})")
                return logical, margin, chi, w0, True, used, ncz, order_verdicts
            if len(answers) > 1:
                log(f"  excise: gate orders DISAGREE ({[v[0][:16] for v in order_verdicts.values()]}) "
                    f"-> not trusted; re-measure a D instead")
                logical, margin, chi, w0 = max(order_verdicts.values(), key=lambda v: v[1])
                return logical, margin, chi, w0, False, used, ncz, order_verdicts
            # only one order got a verdict: trust it only if no other order could still run
            logical, margin, chi, w0 = list(order_verdicts.values())[0]
            only_order = len(ORDERS) < 2 or out_of_time or left() < 120
            if only_order or not REQUIRE_AGREEMENT:
                # Without a cross-order check the ONLY discriminator left is the margin -- and
                # MARGIN (1.15) sits inside the measured noise band: chi-stable WRONG argmaxes
                # reached 1.17 (E20), 1.05-1.15 (E52, E80) and 1.00-1.06 on d2x, versus solving
                # margins 1.18-1.21 (d3_s1) and 1.47-1.71 (d3_s2). A lone verdict at 1.15-1.17 is
                # therefore not evidence, so demand a margin clear of the whole noise band before
                # stamping it trusted. The bitstring is returned either way (best-effort still
                # submits it); trusted=False just keeps the retry loop working if any time is left.
                if margin < SOLO_MARGIN:
                    log(f"  excise: single gate order verdict at margin {margin:.2f} < "
                        f"{SOLO_MARGIN:.2f} (inside the measured noise band) -> kept, not trusted")
                    return logical, margin, chi, w0, False, used, ncz, order_verdicts
                log(f"  excise: single gate order verdict accepted (margin {margin:.2f} >= "
                    f"{SOLO_MARGIN:.2f}, {'no time for a cross-check' if only_order else 'agreement not required'})")
                return logical, margin, chi, w0, True, used, ncz, order_verdicts
            log(f"  excise: only one gate order reached a verdict and a cross-check is possible "
                f"-> not trusted yet")
            return logical, margin, chi, w0, False, used, ncz, order_verdicts
        if not rung_results:
            return None, 0.0, 0, 0.0, False, used, ncz, {}
        # best-effort pick: a rung that repeated the previous rung's top beats a lone high margin, then
        # the highest chi (most accurate), then margin
        st, chi, margin, logical, w0 = max(rung_results, key=lambda r: (r[0], r[1], r[2]))
        return logical, margin, chi, w0, False, used, ncz, order_verdicts

    # ---- first pass: one passing D per block ----
    # BREADTH-first, not depth-first: every block gets its rung-1 draw before any block gets rung 2,
    # and each draw is budgeted at its fair share of what is left. Walking one block through its
    # whole ladder first lets an escalating block 1 eat the entire window (measured constants:
    # 3600+3600+2967 s leaves 1200 s < CANON_RESERVE+D_BUDGET_MIN) so block 2 gets ZERO attempts --
    # and a half-excised circuit still carries a full mirror block.
    windows = {}
    for c, _, _ in blocks:
        k = cs.index(c)
        gaps = ([cs[k] - cs[k - 1]] if k > 0 else []) + ([cs[k + 1] - cs[k]] if k + 1 < len(cs) else [])
        windows[c] = min(gaps) // 2 if gaps else 0
    for r in range(max_rungs):
        for c, tau, _ in blocks:
            if Dstate[c] is not None or rung[c] != r:
                continue
            if left() - CANON_RESERVE < D_BUDGET_MIN:
                break
            measure_block(c, tau, windows[c],
                          share=sum(1 for cc, _, _ in blocks if Dstate[cc] is None))
        if all(Dstate[c] is not None for c, _, _ in blocks) or left() - CANON_RESERVE < D_BUDGET_MIN:
            break
    if not any(Dstate.values()):
        log("excision: no block passed its gate -> fall back")
        return None, info

    # ---- consensus loop: independent D draws VOTE on the bitstring -------------------------
    # The arbiter is no longer one ladder run's margin but AGREEMENT ACROSS INDEPENDENT DRAWS.
    # Each gate order that reaches a stable above-floor answer casts one vote, tagged with the
    # D-draw it came from, so the same run can never vote twice. We keep drawing while time and
    # rungs remain, and stop early only on CONSENSUS agreeing votes from >= MIN_DRAWS distinct
    # draws. Whatever happens, the PLURALITY winner is what gets submitted.
    votes = Counter()
    vote_src = {}                       # bits -> {(draw signature, gate order)}
    vmeta = {}                          # bits -> (best margin, chi, w0, used, ncz)
    best_overall = None
    tries = 0
    dry = 0
    while True:
        tries += 1
        logical, margin, chi, w0, trusted, used, ncz, verdicts = ladder()
        if logical and (best_overall is None or margin > best_overall[1]):
            best_overall = (logical, margin, chi, w0, list(used), ncz)
        sig = tuple(sorted((c, rung[c]) for c, _, _ in blocks if Dstate[c]))
        for order, (lg, mg, ch, w) in verdicts.items():
            key = (sig, order)
            if key in vote_src.get(lg, set()):
                continue                # same draw + same order: not new evidence
            vote_src.setdefault(lg, set()).add(key)
            votes[lg] += 1
            if lg not in vmeta or mg > vmeta[lg][0]:
                vmeta[lg] = (mg, ch, w, list(used), ncz)
        top, nv, ndraws = None, 0, 0
        if votes:
            top, nv = votes.most_common(1)[0]
            ndraws = len({d for d, _ in vote_src[top]})
            log(f"excision: ladder run {tries} votes -> "
                f"{ {b[:12] + '…': n for b, n in votes.most_common()} }; "
                f"leader has {nv} vote(s) from {ndraws} independent draw(s), need {CONSENSUS}/{MIN_DRAWS}")
        pub, pinfo = (top, vmeta[top]) if top else (logical, (margin, chi, w0, list(used), ncz))
        if pub:
            wall_watchdog.publish(pub, {"method": "excision", "margin": pinfo[0], "chi": pinfo[1],
                                        "peak_prob": pinfo[2], "blocks_excised": pinfo[3],
                                        "reduced_cz": pinfo[4], "ladder_runs": tries,
                                        "votes": nv, "draws": ndraws},
                                  trusted=(nv >= CONSENSUS and ndraws >= MIN_DRAWS),
                                  score=nv + min(pinfo[0], 9.0) / 10.0,
                                  stage=f"excision/vote#{tries}", prio=2)
        if top and nv >= CONSENSUS and ndraws >= MIN_DRAWS:
            mg, ch, w, usd, nz = vmeta[top]
            info.update({"confident": True, "trusted": True, "margin": mg, "chi": ch, "peak_prob": w,
                         "blocks_excised": usd, "reduced_cz": nz, "ladder_runs": tries,
                         "votes": nv, "draws": ndraws})
            log(f"excision: CONSENSUS -- {nv} votes from {ndraws} independent draws agree "
                f"(margin {mg:.2f}) -> trust")
            return top, info
        # keep drawing: pick a block with rungs left, worst-fitting D first
        cands = [(c, tau) for c, tau, _ in blocks if rung[c] < max_rungs]
        if not cands or left() - CANON_RESERVE < D_BUDGET_MIN:
            log("excision: no rungs or time left for another measurement")
            break
        def worst_key(ct):
            c = ct[0]
            return (Dstate[c] is not None, Dstate[c][2].get("model_err", 0.0) * -1 if Dstate[c] else 0.0)
        improved = False
        for c, tau in sorted(cands, key=worst_key):
            if left() - CANON_RESERVE < D_BUDGET_MIN:
                break
            if measure_block(c, tau, windows[c]):
                improved = True
                break
        if improved:
            dry = 0
        else:
            dry += 1
            if dry >= MAX_DRY_SWEEPS:
                log(f"excision: no measurement produced a new D in {dry} sweep(s) -> stop drawing")
                break
            log(f"excision: dry sweep {dry}/{MAX_DRY_SWEEPS} (no D improved) -> keep drawing")
    # ---- deadline: submit the PLURALITY winner, not a lone argmax ----
    if votes:
        top, nv = votes.most_common(1)[0]
        ndraws = len({d for d, _ in vote_src[top]})
        mg, ch, w, usd, nz = vmeta[top]
        info.update({"margin": mg, "chi": ch, "peak_prob": w, "blocks_excised": usd, "reduced_cz": nz,
                     "ladder_runs": tries, "votes": nv, "draws": ndraws, "best_effort": True})
        log(f"excision: no consensus after {tries} ladder runs; submitting the plurality winner "
            f"({nv} vote(s) from {ndraws} draw(s), margin {mg:.2f})")
        return top, info
    if best_overall is not None:
        logical, margin, chi, w0, used, ncz = best_overall
        info.update({"margin": margin, "chi": chi, "peak_prob": w0, "blocks_excised": used,
                     "reduced_cz": ncz, "ladder_runs": tries, "best_effort": True})
        log(f"excision: no trusted rung after {tries} ladder runs (best margin {margin:.2f}) -> "
            f"best-effort candidate kept, caller falls back")
        return logical, info
    return None, info
