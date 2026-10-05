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

"""v13: whole-block excision (loose cut tolerance) + concurrent D workers + multi-opinion readout.

solve_v13(qasm_path, log, deadline, publish=None, truth="") -> (bits | None, info)

What changed against v9/v12 and why (all MEASURED 2026-09-19, scratchpad clean/):
  * CUT TOLERANCE. The big block's outer ~45 % is an angle-SWEPT mirror (mirror-position thetas differ by
    1e-6..1e-4). At the exact tolerance it carried no twin evidence, so half the block stayed in the "reduced"
    circuit as a deep shell (686 vs 426 CZ on d3_s1) and a cut reaching into it closed asymmetrically
    (d3_s2 B1: D model err 0.887). Cut evidence now walks a tolerance LADDER (first that passes the audit
    and whose D measures); discovery stays at the exact 1e-6.
  * D WORKERS. One process per block, concurrently: a D-measurement is latency-bound (one core, tiny tensors).
  * READOUT (v12_stage). Forward ladder, pooled multi-order ensemble, Pauli-path gate, optional operator-level
    readout; the decision needs independent opinions to agree.
Every knob has a default and an env override.
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("EXC_MIN_EVIDENCE", "3")
os.environ.setdefault("EXC_PEEL_MODE", "close")
# D workers inherit the environment. Device-adaptive absorber ON by default: MEASURED 1.5-1.7x faster per worker when
# several share the GPU, identical two-body terms, model err 0.0013-0.022 on the four d3_s1 cuts measured with it.
os.environ.setdefault("HQP_DEV_ADAPT", "1")

READOUT_RESERVE = float(os.environ.get("V13_READOUT_RESERVE", "6000"))
# CUT OPTIONS, best first: "tol:extend". The edge-extended cut measures cleanest on the REAL blocks (d3_s1: model err
# 0.001-0.003 on both blocks, two views) but on a synthetic block with structural masking at its rim it paired rim
# gates wrongly and the D came out non-product (h10 block 4560: unitarity 0.00) -> fall back to the plain cut, then
# to tighter tolerances. A failed option costs one D measurement, so the order matters.
CUT_OPTIONS = [(float(o.split(":")[0]), int(o.split(":")[1])) for o in
               os.environ.get("V13_CUT_OPTIONS", "1e-3:1,1e-3:0,1e-4:0,1e-6:0").split(",") if o.strip()]
VIEW_NET_GUARD = os.environ.get("V13_VIEW_NET_GUARD", "1") == "1"   # demote net-negative (over-extended) cut options in the view ladder
MAX_DIST = int(os.environ.get("V13_MAX_DIST", "1500"))
D_PARALLEL = os.environ.get("V13_D_PARALLEL", "1").strip() not in ("0", "false", "False")
D_BUDGET = float(os.environ.get("V13_D_BUDGET", "4800"))
D_DRAWS = int(os.environ.get("V13_D_DRAWS", "3"))
D_VIEWS = int(os.environ.get("V13_D_VIEWS", "2"))              # independent D measurements per block (>= 1)
D_MAX_PROCS = int(os.environ.get("V13_D_MAX_PROCS", "6"))      # concurrent D workers
# v14: try the synchronous CPU engine (syncd.py) on every (block, cut option) before any reference worker is started
D_SYNC = os.environ.get("V13_D_SYNC", "1").strip() not in ("0", "false", "False")
SYNC_BUDGET_S = float(os.environ.get("V13_SYNC_BUDGET_S", "900"))   # per measurement; it needs 0.3-15 s on the public blocks;
                                                                    # 240 -> 900 with the bond cap 1024 (a rescued bond-1024 block took 615-655 s per view with 4 attempts on 16 cores, and
                                                                    # failed at 489 s under CPU contention); a hopeless block costs this before its reference worker
# v19: synchronous attempts run CONCURRENTLY in their own processes (0 = in the main process, one after another, as before).
# MEASURED 2026-09-24 on the w20 rehearsal: the attempts ran one after another while the first GPU workers were already
# taking the CPU, and the same block that needs 18 s on a quiet box needed 361 s -> the D stage dispatched its last
# worker at 1551 s instead of ~300 s.
SYNC_PROCS = int(os.environ.get("V13_SYNC_PROCS", "4"))
VIEW_CUTS = os.environ.get("V13_VIEW_CUTS", "1").strip() not in ("0", "false", "False")   # view v starts at cut option v
MIN_Q = int(os.environ.get("V13_MIN_Q", "44"))
LOOSE_DISC = os.environ.get("V13_LOOSE_DISCOVERY", "1").strip() not in ("0", "false", "False")
LOOSE_TOL = float(os.environ.get("V13_LOOSE_TOL", "1e-3"))
LOOSE_MIN_FRAC = float(os.environ.get("V13_LOOSE_MIN_FRAC", "0.33"))
LOOSE_SEP = int(os.environ.get("V13_LOOSE_SEP", "500"))
# MEASURED 2026-09-24: real fully swept blocks confirm 103-116 of their loose LCS pairs (0.58-0.59); the spurious loose
# peaks the adversarial ladder (genadv h40-h43) pushed through the old gate (30 pairs, 0.2-0.3) had 32-42 of 109-134
# (0.29-0.33). Diluted REAL blocks (s160: 51/176) are found by the exact pass since the u3-count scan window.
LOOSE_MIN_GOOD = int(os.environ.get("V13_LOOSE_MIN_GOOD", "60"))
# confirmed share of the loose LCS twin pairs a rescued block needs. MEASURED: real blocks 0.58-0.59 (d3_s1/d3_s2 with the
# exact pass disabled), block-free d1_s2 peaks 0-3 of 26-36 (< 0.1, and under LOOSE_MIN_GOOD anyway); d3_s1 + 80
# one-sided identity insertions per block: the diluted second block came in at 51/176 = 0.29 and the old 0.3 REJECTED it.
LOOSE_MIN_RATIO = float(os.environ.get("V13_LOOSE_MIN_RATIO", "0.45"))
HERE = os.path.dirname(os.path.abspath(__file__))


TAU_MODES = [m.strip() for m in os.environ.get("V13_TAU_MODES", "all,generic,gencz,gencore").split(",") if m.strip()] or ["all"]
TAU_FIX_UNTOUCHED = os.environ.get("V13_TAU_FIX_UNTOUCHED", "1") == "1"   # census: a tau whose cut leaves whole pairs untouched is also scored with those pairs as fixed points


def _choose_tau(gates, n, c, centres, log, tol=1e-6):
    """tau for the block at c: discover it with every half-window in V13_TAU_HALVES (default 600,900,1200; EXC_TAU_HALF
    alone when unset) and every counting mode in V13_TAU_MODES (default all,generic,gencz: see excise.discover_tau) and,
    when the candidates DISAGREE, keep the tau whose production cut yields the most mirror pairs in the synchronous
    schedule (tie: fewest open insertions, then the smaller window, then the earlier mode).

    MEASURED 2026-09-24 (modes): 8 known-answer blocks were wrong at EVERY window with the all-twin counts (genhard h6
    3210, h7 1610/3530, h10 4560, dense h32 1580/3375, h35 1640/3460): the mis-paired wires' chosen partners won on
    special-theta coincidences (0, pi/2, pi) alone.  The generic-only counts are right on all 8 at every window (the CZ
    refinement on top as well); all-twin counts stay the FIRST candidate so a circuit where every mode agrees is
    untouched.  tol: the twin tolerance (the loose second pass discovers at V13_LOOSE_TOL).

    MEASURED 2026-09-24 on 26 known-answer blocks (genhard h0-h14 + the dense ladder h30-h35) and the 4 real samples:
    the 600 window alone mis-pairs 4-30 wires on 6 blocks (h7 1610, h11 1035, h32 both, h35 1640) -- their blocks span
    ~2400 gates -- and every such block became un-absorbable (h32 1580: 132 open insertions, TooBig at 349 CZ); no
    fixed larger window is safe either (1500 corrupts h5 3905, h6 3210, h7 3530). The census picks the correct tau on
    every block where any window is correct (h32 1580: pairs 216 -> 331, open 132 -> 24), leaves every block where 600
    was already right untouched (all four real samples: identical census at every window), and costs nothing unless
    the windows disagree. Returns (tau, twin_fraction)."""
    import excise as X
    import excision_solver as ES
    halves = [int(x) for x in os.environ.get("V13_TAU_HALVES", "600,900,1200").split(",") if x.strip()]
    if not halves:
        halves = [int(os.environ.get("EXC_TAU_HALF", "600"))]
    cands = []                                                    # (label, tau, fraction), first = the historical choice
    for h in halves:
        for mode in TAU_MODES:
            try:
                tau, _, fr = X.discover_tau(gates, n, c, h, tol=tol, mode=mode)
            except Exception as e:                                # noqa: BLE001
                log(f"v13: block {c}: tau discovery at half-window {h} mode {mode} failed ({type(e).__name__}) -> skipped")
                continue
            cands.append((f"{h}/{mode}", [int(v) for v in tau], float(fr)))
    if not cands:
        tau, _, fr = X.discover_tau(gates, n, c, int(os.environ.get("EXC_TAU_HALF", "600")), tol=tol)
        return [int(v) for v in tau], float(fr)
    if all(t == cands[0][1] for _, t, _ in cands):
        return cands[0][1], cands[0][2]
    # the windows disagree: score every distinct tau by the census of the cut it produces
    import syncd
    gaps = [abs(c - c2) for c2 in centres if c2 != c]
    win = int(max(gaps)) if gaps else 0
    prev = (X.EXC_MAX_DIST, X.EXC_TWIN_TOL, os.environ.get("EXC_EXTEND"))
    X.EXC_MAX_DIST = MAX_DIST
    X.EXC_TWIN_TOL = CUT_OPTIONS[0][0] if CUT_OPTIONS else 1e-3
    os.environ["EXC_EXTEND"] = str(CUT_OPTIONS[0][1] if CUT_OPTIONS else 1)
    scored = []
    try:
        seen = set()
        for rank, (h, tau, fr) in enumerate(cands):
            key = tuple(tau)
            if key in seen:
                continue
            seen.add(key)
            # census key (V13_TAU_CENSUS_KEY): "net" (default) = audit ok, mirror pairs MINUS open insertions, pairs, rank;
            # "pairs" = the historical audit ok, pairs, -open, rank.  MEASURED 2026-09-24 on genadv h45 block 3015
            # (SWAP cores spread over half the block): the WRONG tau (8 wires) gave 134 pairs / 89 open on a 431-CZ
            # over-extended cut, the RIGHT one 133 pairs / 26 open on 322 CZ -- "pairs first" picked the wrong one by a
            # single pair; the net score picks the right one and agrees with every earlier census decision (h32 1580
            # 84 -> 307, h7 1610, h11 1035 137 -> 197, h35 1640, w60 1545).
            # V13_TAU_CENSUS_LADDER (default 1): score every tau by the BEST census over the whole cut ladder
            # (CUT_OPTIONS), not only its first option. MEASURED 2026-09-25 on genadv h64 block 2705 (SWAP cores spread
            # over 70% of a 13-layer block): the RIGHT tau's tol-1e-3 cut swallowed 612 foreign gates (645 CZ, pairs 159 /
            # open 189, net -30) while its tol-1e-4 cut is compact (286 CZ, 0 foreign, pairs 114 / open 30, net 84); the
            # first-option census picked a WRONG tau (8 wires, net 63) and the block was lost (D carried a residual
            # permutation, kept).  Real samples are untouched: their windows agree, so no census runs.
            _net = os.environ.get("V13_TAU_CENSUS_KEY", "net").strip().lower() != "pairs"
            _opts = list(CUT_OPTIONS) if (os.environ.get("V13_TAU_CENSUS_LADDER", "1") == "1" and CUT_OPTIONS) else \
                ([CUT_OPTIONS[0]] if CUT_OPTIONS else [(1e-3, 1)])
            best_o = None
            seen_cuts = set()
            for otol, oext in _opts:
                try:
                    X.EXC_TWIN_TOL = float(otol)
                    os.environ["EXC_EXTEND"] = str(int(oext))
                    cut, tw = X.envelope_cut(gates, n, c, tau, window=win, log=lambda m: None)
                    se_ = (tuple(cut.s), tuple(cut.e))
                    if se_ in seen_cuts:
                        continue
                    seen_cuts.add(se_)
                    aud = X.audit(cut, tw, log=lambda m: None, tol=float(otol))
                    ok = not (aud["orphans"] or aud["inconsistent"] or aud["untouched"])
                    if (not ok and TAU_FIX_UNTOUCHED and aud["untouched"] and not aud["orphans"] and not aud["inconsistent"]):
                        # V13_TAU_FIX_UNTOUCHED (default 1): the cut under this tau leaves whole tau-pairs UNTOUCHED (no gate
                        # of either wire inside E: their twins were too sparse for the envelope). E is then exactly
                        # Pi_tau' with those pairs as FIXED POINTS -- the pair's own gates and SWAP core stay in the reduced
                        # circuit as an exact mirror around their core -- so tau' is scored as one more candidate.
                        # MEASURED 2026-09-25 genadv h67 2575: the right tau leaves (24,42),(32,33) untouched (20-28 gates
                        # each) -> audit FAIL at every option -> a wrong audit-ok tau (4 wires) won; tau' passes every
                        # option (net 71 vs 66).
                        empty_ = [x for x in range(n) if cut.empty(x) and tau[x] != x]
                        tau2 = list(tau)
                        for a_ in empty_:
                            tau2[a_] = a_
                            tau2[tau[a_]] = tau[a_]
                        if tuple(tau2) not in seen and all(tau2[tau2[a_]] == a_ for a_ in range(n)):
                            cands.append((f"{h}+fixed{len(empty_)}", tau2, fr))
                    W, _ = X.wire_ops(gates, n)
                    E = sorted({W[x][k] for x in range(n) for k in range(cut.s[x], cut.e[x] + 1)})
                    split, _ = ES.wire_split(gates, n, cut, tw, tau, c, E, mode="core2", log=lambda m: None)
                    _ops, st = syncd.schedule(gates, n, tau, E, split, tw, log=None)
                    key_ = ((1 if ok else 0, st["mirror_pairs_found"] - st["open_at_end"], st["mirror_pairs_found"], -rank) if _net
                            else (1 if ok else 0, st["mirror_pairs_found"], -st["open_at_end"], -rank))
                    st = dict(st, census_tol=float(otol), census_ext=int(oext), census_cz=int(aud.get("cz", -1)))
                    if best_o is None or key_ > best_o[0]:
                        best_o = (key_, st)
                except Exception as e:                            # noqa: BLE001
                    if best_o is None:
                        best_o = ((0, -10 ** 6, -1, -rank), {"census_error": f"{type(e).__name__}: {str(e)[:60]}"})
            if best_o is None:
                best_o = ((0, -10 ** 6, -1, -rank), {})
            scored.append((best_o[0], h, tau, fr, best_o[1]))
    finally:
        X.EXC_MAX_DIST, X.EXC_TWIN_TOL = prev[0], prev[1]
        if prev[2] is None:
            os.environ.pop("EXC_EXTEND", None)
        else:
            os.environ["EXC_EXTEND"] = prev[2]
    scored.sort(key=lambda t: t[0], reverse=True)
    best = scored[0]
    first = next(t for t in scored if t[1] == cands[0][0])
    nd = sum(1 for w in range(n) if best[2][w] != cands[0][1][w])
    log(f"v13: block {c}: tau candidates (windows {halves} x modes {TAU_MODES}) disagree ({nd} wires between {cands[0][0]} "
        f"and {best[1]}) -> census picks {best[1]}: mirror pairs {first[4].get('mirror_pairs_found')} -> "
        f"{best[4].get('mirror_pairs_found')}, open insertions {first[4].get('open_at_end')} -> {best[4].get('open_at_end')}"
        f" (census cut tol {best[4].get('census_tol')} extend {best[4].get('census_ext')} {best[4].get('census_cz')} CZ;"
        f" first {first[4].get('census_tol')}/{first[4].get('census_ext')} {first[4].get('census_cz')} CZ)")
    return best[2], best[3]


def solve_v13(qasm_path, log, deadline, publish=None, truth=""):
    import excise as X
    import excision_solver as ES
    import v12_stage as STG
    from qiskit import qasm2
    from qiskit.circuit.library import UGate

    t0 = time.time()
    cache_dir = os.environ.get("D_CACHE_DIR", "")             # experiments only; empty (default) = off
    if cache_dir:
        # a MISSING cache directory used to make every synchronous measurement "crash" with FileNotFoundError after
        # doing all of its work, silently downgrading the whole run to the GPU fallback (MEASURED 2026-09-20)
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:                                  # noqa: BLE001
            log(f"v13: D cache dir {cache_dir} is unusable ({type(e).__name__}) -> caching off")
            cache_dir = ""
    custom = [qasm2.CustomInstruction('u', 3, 1, lambda t, p, l: UGate(t, p, l), builtin=True)]
    circ = qasm2.load(qasm_path, custom_instructions=custom)  # FILE order: DAG order breaks discovery (10/27 circuits)
    n = circ.num_qubits
    gates = ES._circ_to_gates(circ)
    qsha = hashlib.sha256(open(qasm_path, "rb").read()).hexdigest()[:12]
    log(f"v13: circuit {n}q {len(gates)} gates sha {qsha}")
    if n < MIN_Q:
        return None, {"method": "v13", "skipped": f"{n} qubits < {MIN_Q}"}

    # ---------------- 1. blocks (exact tolerance) ----------------
    blocks = []
    loose_centres = set()                                 # blocks the LOOSE pass added (see _block_env)
    forced = os.environ.get("V13_FORCE_BLOCKS", "")           # TESTS ONLY: '[{"centre": c, "tau": [...]}, ...]' (small-n
    if forced:                                                # integration circuits are below the discovery's scale)
        for b in json.loads(forced):
            blocks.append((int(b["centre"]), [int(v) for v in b["tau"]], 1.0))
    else:
        found = X.discover_centres(gates, n, log=lambda m: None)
        for c, _, _ in found:
            tau, fr = _choose_tau(gates, n, int(c), [int(c2) for c2, _, _ in found], log)
            blocks.append((int(c), [int(v) for v in tau], float(fr)))
        # SECOND PASS at the loose tolerance: a block can be angle-swept THROUGHOUT (no exact-angle core at all) --
        # MEASURED 2026-09-19 on the public d1 samples: invisible at 1e-6 (peak twin fraction 0.12-0.22, below the
        # 0.25 gate), a clear block at 1e-3 (0.41-0.46; d1_s1: 476-CZ cut, circuit 684 -> 208 CZ). The public d3
        # samples have none (their loose peaks coincide with the exact ones; the remainder scans flat at <= 0.2),
        # but a validator circuit that does would keep a whole mirror block in the "reduced" circuit.
        if LOOSE_DISC:
            try:
                found2 = X.discover_centres(gates, n, tol=LOOSE_TOL, min_frac=LOOSE_MIN_FRAC, log=lambda m: None)
            except Exception as e:                                # noqa: BLE001
                found2 = []
                log(f"v13: loose discovery failed ({type(e).__name__}: {str(e)[:80]})")
            for c2, _, fr2 in found2:
                if all(abs(int(c2) - b[0]) > LOOSE_SEP for b in blocks):
                    tau2, fr2b = _choose_tau(gates, n, int(c2), [b[0] for b in blocks] + [int(c3) for c3, _, _ in found2],
                                             log, tol=LOOSE_TOL)
                    if not all(tau2[tau2[x]] == x for x in range(n)):
                        continue
                    # The loose twin FRACTION alone is not enough: MEASURED, block-free d1_s2 shows loose peaks of
                    # 0.37-0.46 that are pure statistics (26-36 LCS twin pairs, 0-3 of them with >= 3 angle-twin
                    # neighbours), while the real fully-swept block of d1_s1 has 179 LCS pairs / 103 confirmed and a
                    # real d3 block 122 / 71. A false block costs a whole D measurement before its gate rejects it.
                    tw2 = X.cz_twins(gates, n, int(c2), tau2, lo=0, hi=len(gates))
                    probe2 = X.Cut(gates, n, int(c2), tau2)
                    good2 = sum(1 for r_, l_ in tw2.items() if X.evidence(probe2, l_, r_, LOOSE_TOL) >= 3)
                    if good2 >= LOOSE_MIN_GOOD and good2 >= LOOSE_MIN_RATIO * max(len(tw2), 1):
                        blocks.append((int(c2), [int(v) for v in tau2], float(fr2)))
                        loose_centres.add(int(c2))
                        log(f"v13: loose-tolerance discovery adds a block at {int(c2)} (twin fraction {fr2:.2f} at tol "
                            f"{LOOSE_TOL:g}, {good2}/{len(tw2)} confirmed twin pairs)")
                    else:
                        log(f"v13: loose peak at {int(c2)} (fraction {fr2:.2f}) rejected: {good2}/{len(tw2)} confirmed twin pairs")
            blocks.sort(key=lambda b: b[0])
    cs = [b[0] for b in blocks]
    windows = {}
    for k, c in enumerate(cs):
        gaps = ([cs[k] - cs[k - 1]] if k > 0 else []) + ([cs[k + 1] - cs[k]] if k + 1 < len(cs) else [])
        windows[c] = int(max(gaps)) if gaps else 0
    X.EXC_MAX_DIST = MAX_DIST
    log(f"v13: blocks at {cs} (twin fractions {[round(b[2], 2) for b in blocks]}), windows {windows}")

    # ---------------- 2. cuts: every tolerance whose cut passes the audit, largest tolerance first ----------------
    cut_opts = {}
    view_net = {}
    if VIEW_NET_GUARD:
        import syncd
    for c, tau, _ in blocks:
        ok_tols = []
        for tol, ext in CUT_OPTIONS:
            try:
                os.environ["EXC_EXTEND"] = str(ext)
                cut, tw = X.envelope_cut(gates, n, c, tau, window=windows[c], tol=tol, log=lambda m: None)
                a = X.audit(cut, tw, log=lambda m: None, tol=tol)
            except Exception as e:                                # noqa: BLE001
                log(f"v13: block {c} tol {tol:g} extend {ext}: cut failed ({type(e).__name__})")
                continue
            good = not (a["orphans"] or a["inconsistent"] or a["untouched"])
            dup = any((list(cut.s), list(cut.e)) == se for _, _, se, _e in ok_tols)
            log(f"v13: block {c} tol {tol:g} extend {ext}: {a['cz']} CZ, orphans {a['orphans']}, inconsistent "
                f"{a['inconsistent']}, untouched {a['untouched']} -> {'ok' if good else 'audit FAIL'}"
                f"{' (same cut as an earlier option)' if good and dup else ''}")
            if good and not dup:
                ok_tols.append((tol, a["cz"], (list(cut.s), list(cut.e)), ext))
                if VIEW_NET_GUARD:
                    # V13_VIEW_NET_GUARD (default 1): a passing cut whose synchronous schedule has MORE open insertions than
                    # mirror pairs is over-extended (it swallowed a neighbour: the audit cannot see foreign gates that break
                    # no twin pair). Such options are moved BEHIND every net-positive option so the views start from a
                    # compact cut. MEASURED 2026-09-25 genadv h64 block 2705 (right tau): tol 1e-3 cuts 645 CZ with 612
                    # foreign gates (pairs 159 / open 189), tol 1e-4 286 CZ with none (114 / 30). Real samples: every
                    # passing option is net-positive (d3_s1/s2 census open 24-28 vs pairs 200-330) -> order unchanged.
                    try:
                        _prev_tol = X.EXC_TWIN_TOL
                        X.EXC_TWIN_TOL = tol
                        W_, _ = X.wire_ops(gates, n)
                        E_ = sorted({W_[x][k] for x in range(n) for k in range(cut.s[x], cut.e[x] + 1)})
                        split_, _ = ES.wire_split(gates, n, cut, tw, tau, c, E_, mode="core2", log=lambda m: None)
                        _o, st_ = syncd.schedule(gates, n, tau, E_, split_, tw, log=None)
                        X.EXC_TWIN_TOL = _prev_tol
                        view_net[(c, tol, ext)] = int(st_["mirror_pairs_found"]) - int(st_["open_at_end"])
                    except Exception as e:                        # noqa: BLE001
                        X.EXC_TWIN_TOL = _prev_tol
                        log(f"v13: block {c} tol {tol:g} extend {ext}: schedule census failed ({type(e).__name__}) -> net unknown")
        if VIEW_NET_GUARD and ok_tols:
            neg = [o for o in ok_tols if view_net.get((c, o[0], o[3]), 0) < 0]
            if neg and len(neg) < len(ok_tols):
                # V13_VIEW_NET_DROP (default 1): an over-extended option is DROPPED (not merely demoted) when a compact one
                # exists -- MEASURED 2026-09-25 h64 2705: the demoted 645-CZ view ground at bond 1024 for the whole sync
                # budget (567 s+) and would then have burnt the GPU reference budget on a cut we already know swallowed a
                # neighbour, while view 0 on the compact cut absorbed exactly in 0 s. With one option left every view
                # measures that cut (its own draw), the standing design for single-option blocks.
                keep_ = [o for o in ok_tols if view_net.get((c, o[0], o[3]), 0) >= 0]
                drop_ = os.environ.get("V13_VIEW_NET_DROP", "1") == "1"
                ok_tols = keep_ if drop_ else keep_ + neg
                log(f"v13: block {c}: view order guard: {len(neg)} over-extended option(s) (schedule net "
                    f"{[view_net.get((c, o[0], o[3])) for o in neg]}, {[o[1] for o in neg]} CZ) "
                    f"{'DROPPED' if drop_ else 'moved behind the compact ones'} -> views use {[(o[0], o[3], o[1]) for o in ok_tols]}")
            elif neg:
                log(f"v13: block {c}: view order guard: EVERY passing option is net-negative "
                    f"({[view_net.get((c, o[0], o[3])) for o in neg]}) -> ladder order kept")
        cut_opts[c] = ok_tols

    # ---------------- 2b. D: one worker per (block, VIEW), all concurrently ----------------
    # A VIEW is an independent measurement of every block's D (its own cutoff/seed rung). Two views give two
    # reduced circuits that differ only by D-measurement noise: the ensemble pools both, so "agreement" also
    # covers the one stage whose error no readout can see from inside (a wrong-but-product-like D).
    # Workers are latency-bound (one core, tiny tensors), so the extra view costs no wall time on a 24-vCPU box.
    # View v STARTS at cut option v (view 0 = edge-extended cut, view 1 = plain cut, ...): the rim of a cut is its
    # fragile part -- MEASURED on h10, the same real block that measures at model err 0.0015 inside d3_s1 came out
    # non-product (unitarity 0.01, a CNOT-like residual on one rim wire) when a neighbouring block changed the window
    # and the extension took a slightly different rim. Two different cuts measured CONCURRENTLY cost no extra wall
    # time, almost never fail together, and give the ensemble two genuinely different reductions.
    sync_tried = set()                       # (block, cut option) the synchronous engine has already been tried on
    state = {(c, v): {"tau": tau, "tols": cut_opts[c], "ti": min(v, max(len(cut_opts[c]) - 1, 0)) if VIEW_CUTS else 0,
                      "k": 0, "result": None}
             for c, tau, _ in blocks for v in range(D_VIEWS)}
    tmpdir = tempfile.mkdtemp(prefix="v13d_")

    def key(c, tol, cut_off, sd, se=None):
        sig = hashlib.sha256(json.dumps(se).encode()).hexdigest()[:8]          # the CUT itself (builders evolve)
        return os.path.join(cache_dir or tmpdir,
                            f"D13_{qsha}_c{c}_cut{sig}_tol{tol:g}_ev{os.environ['EXC_MIN_EVIDENCE']}"
                            f"_{os.environ['EXC_PEEL_MODE']}_sw{os.environ.get('HQP_D_SWAP_AT', 'centre')}_co{cut_off:g}"
                            f"_r{os.environ.get('HQP_D_RESID', '1')}_s{sd}.pkl")

    def try_sync(cv, out, tol, ext, cut_off, sd):
        """SYNCHRONOUS engine first (syncd.py, CPU, in-process): MEASURED on every public block (d2 855 CZ, d3 380-661
        CZ) it reproduces the reference absorber's D (1q blocks to fidelity >= 0.996, identical pi-terms) in 0.3-15 s
        instead of 2000-3000 s. Any failure (exception, bond/time give-up, quality gate) leaves no result file and
        the reference worker is launched exactly as before."""
        c, v = cv
        t_s = time.time()
        prev = {k_: os.environ.get(k_) for k_ in ("HQP_D_ENGINE", "EXC_EXTEND", "SYNCD_BUDGET_S")}
        try:
            os.environ["HQP_D_ENGINE"] = "sync_only"
            os.environ["EXC_EXTEND"] = str(ext)
            os.environ["SYNCD_BUDGET_S"] = str(SYNC_BUDGET_S)
            X.EXC_TWIN_TOL = tol
            msgs = []
            _live = ("syncd:", "D structure", "resid: block", "-> EXCISE", "keep (gate failed)", "fails the ansatz gate", "STRUCTURAL failure")

            def _collect(m_):                                  # phase lines at once (mirrors syncworker.py)
                msgs.append(m_)
                if any(k_ in m_ for k_ in _live):
                    log(f"v13: block {c} view {v}: [{time.time()-t_s:.0f}s] {m_.strip()[:180]}")

            cut, corr, binfo = ES.measure_D(gates, n, c, state[cv]["tau"], windows[c], None, _collect,
                                            budget=SYNC_BUDGET_S, cutoff=cut_off, seed=sd)
            # the sync path silences measure_D's running commentary (one line per block would flood the log), but a
            # CORRECTED WIRE MAP must never be silent: it changes what the reduced circuit means. MEASURED on h6,
            # whose block 3210 is only excisable because the decoder re-pairs four wires.
            for m_ in msgs:
                if "residual permutation" in m_:
                    log(f"v13: block {c} view {v}:{m_.split('decode:')[-1].strip()[:200]}")
                elif ("resid: block" in m_ or "resid: WARNING" in m_ or "residual correction failed" in m_ or
            "fails the ansatz gate" in m_ or "STRUCTURAL failure" in m_ or "-> EXCISE" in m_ or "keep (gate failed)" in m_):
                    # how much of D the product ansatz was throwing away, and what the correction recovered
                    log(f"v13: block {c} view {v}: {m_.strip()[:200]}")
            if cut is None:
                why = binfo.get("sync_failed") or (f"gate: model err {binfo.get('model_err')}, unitary {binfo.get('unitary')}, "
                                                   f"3-body {binfo.get('three_body')}")
                uni, merr = binfo.get("unitary"), binfo.get("model_err")
                structural = not binfo.get("sync_failed") and ((uni is not None and uni < 0.5) or (merr is not None and merr > 0.9))
                if structural:
                    # the operator of THIS CUT is not a product (MEASURED on h10 block 3120, edge-extended cut: the
                    # reference absorber finds the same thing, unitarity 0.008, after ~1 h). Record the failure so
                    # that the cut-option ladder moves on at once instead of re-measuring it on the GPU.
                    pickle.dump({"ok": False, "info": {k_: v_ for k_, v_ in binfo.items() if k_ != "audit"}}, open(out, "wb"))
                    log(f"v13: block {c} view {v}: sync engine: the cut's operator is not a product ({why}) [{time.time()-t_s:.1f}s]")
                    return True
                log(f"v13: block {c} view {v}: sync engine gave no D ({why}) [{time.time()-t_s:.1f}s] -> reference worker")
                return False
            # D-LEVEL CROSS-CHECK (v14, "more seeds must agree" applied to the measurement itself): re-measure the SAME
            # cut at other truncation cutoffs -- seconds with the synchronous engine. MEASURED on every public and
            # variant block: the pi-term set is identical and the worst per-wire 1q fidelity is >= 0.993 across
            # 1.2e-3 / 6e-4 / 3e-4; a cut whose D moves between cutoffs is not a cut we should excise.
            xcheck = [float(x) for x in os.environ.get("V13_SYNC_CROSSCHECK", "1.2e-3,3e-4").split(",") if x.strip()]
            # SLOW blocks (bond near the cap): a cross-check re-measures the whole cut at another cutoff, i.e. up to the full
            # budget EACH; MEASURED 2026-09-24 on genadv h53 block 3020 (right tau, spread cores): the attempt stayed silent for
            # 45 min (3 x 900 s) while the D stage waited. When the primary measurement took longer than V13_XCHECK_SLOW_S
            # (default 300 s) only the first cross-check cutoff runs, and a cross-check that returns no D on such a block counts
            # as SKIPPED (the primary gates + the two-view agreement still apply), not as "D not stable".
            slow_ = (time.time() - t_s) > float(os.environ.get("V13_XCHECK_SLOW_S", "300"))
            if slow_ and len(xcheck) > 1:
                log(f"v13: block {c} view {v}: the measurement took {time.time()-t_s:.0f}s -> only the first cross-check cutoff")
                xcheck = xcheck[:1]
            min_fid = float(os.environ.get("V13_SYNC_MIN_FID", "0.98"))
            xinfo = []
            if xcheck:
                import numpy as _np
                A_of = lambda cc, w_: _np.array([complex(e[0], e[1]) for e in cc[("A", w_)]]).reshape(2, 2)   # noqa: E731
                # compare only STRONG two-body terms: a phase near the decoder's 0.05-0.16 rad noise floor can appear
                # or vanish between cutoffs without meaning anything (MEASURED on d1_s1's fully swept block), and a
                # single term of difference is tolerated (V13_SYNC_PI_DIFF).
                pi_min = float(os.environ.get("V13_SYNC_PI_MIN", "1.0"))
                pis = lambda cc: {k_[1:] for k_, v_ in cc.items() if k_[0] == "cz" and                        # noqa: E731
                                  (v_ is None or abs(float(v_)) >= pi_min)}
                pi_diff = int(os.environ.get("V13_SYNC_PI_DIFF", "1"))
                for co2 in xcheck:
                    if abs(co2 - cut_off) < 1e-12:
                        continue
                    try:
                        c2, corr2, bi2 = ES.measure_D(gates, n, c, state[cv]["tau"], windows[c], None, lambda m: None,
                                                      budget=(min(SYNC_BUDGET_S, float(os.environ.get("V13_XCHECK_SLOW_BUDGET_S", "300"))) if slow_ else SYNC_BUDGET_S), cutoff=co2, seed=sd)
                    except Exception:                                 # noqa: BLE001
                        c2, corr2 = None, None
                    if c2 is None or corr2 is None:
                        xinfo.append((co2, None, None, None))
                        continue
                    fid = min(abs(_np.trace(A_of(corr2, w_).conj().T @ A_of(corr, w_))) / 2 for w_ in range(n))
                    nd = len(pis(corr2) ^ pis(corr))
                    # When the residual correction carried the block (the ansatz kept only a fraction of D), the
                    # ansatz's 1q factors are NOISE and comparing them across cutoffs says nothing: MEASURED on w20
                    # block 3200, correction 0.190 -> 0.9987 on BOTH views while the ansatz 1q fidelity across
                    # cutoffs was 0.040-0.054 -> "not stable" -> the cut was discarded and the next option failed
                    # structurally. What is meaningful in that regime is that the EMITTED operator reproduces the
                    # measured one at both cutoffs. Pi-term stability is still required either way.
                    # V13_XCHECK_RESID=0 restores the ansatz-only comparison.
                    if os.environ.get("V13_XCHECK_RESID", "1") != "0":
                        f1, f2, f0 = binfo.get("resid_fid1"), bi2.get("resid_fid1"), binfo.get("resid_fid0")
                        rmin_x = float(os.environ.get("V13_RESID_MIN", "0.97"))
                        if f1 is not None and f2 is not None and f0 is not None and f0 < min_fid:
                            if min(float(f1), float(f2)) >= rmin_x:
                                log(f"v13: block {c} view {v}: cross-check at cutoff {co2:g} judged on the CORRECTED "
                                    f"operator (the ansatz kept only {f0:.3f} of D, so its 1q factors are noise): "
                                    f"post-correction overlaps {float(f1):.4f} / {float(f2):.4f}")
                                fid = min(float(f1), float(f2))
                    xinfo.append((co2, float(fid), nd <= pi_diff, nd))
                good = [x for x in xinfo if x[1] is not None]
                bad = [x for x in good if x[1] < min_fid or not x[2]]
                if not good and slow_ and not bad:
                    log(f"v13: block {c} view {v}: cross-check returned no D on a slow block -> skipped (primary gates only)")
                    good = [("skipped", 1.0, True, 0)]
                log(f"v13: block {c} view {v}: D cross-check " + ", ".join(
                    f"cutoff {x[0]:g}: " + ("no D" if x[1] is None else f"1q fidelity {x[1]:.4f}, strong two-body terms "
                                            f"{'match' if x[2] else 'DIFFER'}" + (f" ({x[3]} of them)" if x[3] else "")) for x in xinfo))
                if bad or not good:
                    log(f"v13: block {c} view {v}: D is NOT stable across cutoffs -> this cut is not trusted, next cut option")
                    pickle.dump({"ok": False, "info": dict({k_: v_ for k_, v_ in binfo.items() if k_ != "audit"},
                                                           xcheck_failed=True)}, open(out, "wb"))
                    return True
            res = {"ok": True, "s": list(cut.s), "e": list(cut.e), "tau": list(state[cv]["tau"]), "corr": corr,
                   "info": dict({k_: v_ for k_, v_ in binfo.items() if k_ != "audit"},
                                xcheck=[(x[0], x[1], x[2]) for x in xinfo])}
            tmp = out + ".tmp"
            pickle.dump(res, open(tmp, "wb"))
            os.replace(tmp, out)
            sy = binfo.get("sync", {})
            log(f"v13: block {c} view {v}: sync engine: {sy.get('pairs')}/{sy.get('mirror_pairs_found')} mirror pairs in lockstep, "
                f"peak bond {sy.get('peak_bond')}, final {sy.get('final_bond')}, log10 norm2 {sy.get('log10_norm2', 0):+.3f} "
                f"[{time.time()-t_s:.1f}s]")
            return True
        except Exception as e:                                    # noqa: BLE001
            log(f"v13: block {c} view {v}: sync engine crashed ({type(e).__name__}: {str(e)[:100]}) -> reference worker")
            return False
        finally:
            for k_, v_ in prev.items():
                if v_ is None:
                    os.environ.pop(k_, None)
                else:
                    os.environ[k_] = v_

    def _block_env(c):
        """Per-block environment for a D measurement.  A block the LOOSE pass added gets NO residual rescue
        (V13_LOOSE_RESCUE=0, default): MEASURED 2026-09-24 on the adversarial ladder (genadv h40/h41/h42, exact truths),
        the loose pass added a third "block" inside the base circuit's suffix (loose twin fraction 0.47, 40/123 confirmed
        pairs) and the rescue then excised it on a garbage operator (ansatz kept 0.138 of D, model err 0.616, unitary
        0.31) because the 150-gate residual fit reached |<I,R>| 0.9946 -- a wrong reduction. A real fully swept block
        (d1_s1's) gives a clean product D and needs no rescue; a spurious region cannot pass the ansatz gate alone."""
        if int(c) in loose_centres and os.environ.get("V13_LOOSE_RESCUE", "0").strip() in ("0", "false", "no", "off", ""):
            return {"HQP_RESID_RESCUE": "0"}
        return {}

    def _spawn_sync(cv, sync_out, tol, ext, cut_off, sd):
        """Run the synchronous attempt in its OWN interpreter (syncworker.py, the same pattern as dworker.py). A forked
        child is not an option: the parent has initialised CUDA and used torch's OpenMP pool, and a forked child dies
        or deadlocks on its first torch CPU op (MEASURED 2026-09-24, validator container flags). The child speaks
        through the result file alone: present -> decided (a D or a structural failure), absent -> gave up or crashed
        and the reference worker is launched exactly as before. Returns None when the process cannot be started."""
        c, v = cv
        st = state[cv]
        # child alarm 2*budget+600: absorption + decode/resid (HQP_RESID_MAX_S) + one capped cross-check
        job = {"qasm": qasm_path, "centre": c, "view": v, "tau": st["tau"], "window": windows[c], "tol": tol, "extend": ext,
               "cutoff": cut_off, "seed": sd, "out": sync_out, "budget": SYNC_BUDGET_S, "max_dist": MAX_DIST,
               "min_ev": int(os.environ["EXC_MIN_EVIDENCE"]), "peel": os.environ["EXC_PEEL_MODE"], "t0": t0,
               "max_s": float(os.environ.get("V13_SYNC_CHILD_MAX_S", str(2 * SYNC_BUDGET_S + 600)))}
        try:
            jp = os.path.join(tmpdir, f"sjob_{c}_{v}_{st['ti']}_{st['k']}.json")
            json.dump(job, open(jp, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "syncworker.py"), jp], cwd=HERE,
                                 env={**os.environ, **_block_env(c), "HQP_D_ENGINE": "sync_only"})     # stdout/stderr: the parent's stream
        except Exception as e:                                # noqa: BLE001
            log(f"v13: block {c} view {v}: cannot start a synchronous attempt process ({type(e).__name__}: {str(e)[:80]}) -> in-process")
            return None
        p.sync, p.out = True, sync_out
        return p

    def _reap(cv, p, relaunch):
        """A finished process: a synchronous attempt hands over to the reference worker when it left no result file
        (only while the D stage is running -- a straggler that fails during the readout must not start a GPU worker
        next to it); a worker's result is collected."""
        if getattr(p, "sync", False):
            if os.path.exists(p.out):
                state[cv]["out"] = p.out
                collect(cv)
                return None
            if not relaunch:
                return None
            log(f"v13: block {cv[0]} view {cv[1]}: synchronous attempt ended with exit code {p.poll()} and no result "
                f"-> reference worker")
            p2 = launch(cv)                                   # sync_tried already holds this attempt: straight to a worker
            if p2 is None:
                collect(cv)
            return p2
        collect(cv)
        return None

    def launch(cv):
        c, v = cv
        st = state[cv]
        tol, _cz, se_opt, ext = st["tols"][st["ti"]]
        cut_off, sd = ES.attempt_at((0 if VIEW_CUTS else v) + st["k"] * (1 if VIEW_CUTS else D_VIEWS))
        out = key(c, tol, cut_off, sd, se_opt)
        st["out"] = out
        if os.path.exists(out):
            return None
        # key the attempt by (block, cut option, cutoff, seed): two views of the SAME cut differ only by their draw,
        # and each draw deserves its own (1-second) synchronous measurement rather than a 50-minute GPU worker
        if D_SYNC and (c, st["ti"], cut_off, sd) not in sync_tried:
            sync_tried.add((c, st["ti"], cut_off, sd))
            sync_out = out[:-4] + "_sync3.pkl"                 # bump the tag whenever engine or DECODE semantics change (cache!)
            if os.path.exists(sync_out):
                st["out"] = sync_out
                return None
            if SYNC_PROCS > 0:
                p = _spawn_sync(cv, sync_out, tol, ext, cut_off, sd)
                if p is not None:
                    return p                                   # pump() reaps it: result file -> collect, none -> worker
            if try_sync(cv, sync_out, tol, ext, cut_off, sd):
                st["out"] = sync_out
                return None
        if cache_dir:                                 # experiments only: reuse a v12-format entry if the CUT is identical
            legacy = os.path.join(cache_dir, f"D_{qsha}_c{c}_w{windows[c]}_md{MAX_DIST}_ev{os.environ['EXC_MIN_EVIDENCE']}"
                                             f"_{os.environ['EXC_PEEL_MODE']}_co{cut_off:g}_s{sd}.pkl")
            for cand in (legacy, legacy.replace(f"_md{MAX_DIST}_", f"_md{MAX_DIST}_tol{tol:g}_")):
                if os.path.exists(cand):
                    try:
                        blob = pickle.load(open(cand, "rb"))
                        if [blob["s"], blob["e"]] == list(st["tols"][st["ti"]][2]):
                            pickle.dump({"ok": True, "s": blob["s"], "e": blob["e"], "tau": blob["tau"],
                                         "corr": blob["corr"], "info": blob["info"]}, open(out, "wb"))
                            log(f"v13: block {c} view {v}: reusing v12 D cache entry {os.path.basename(cand)}")
                            return None
                    except Exception as e:                        # noqa: BLE001
                        log(f"v13: block {c}: legacy cache unreadable ({type(e).__name__})")
        job = {"qasm": qasm_path, "centre": c, "tau": st["tau"], "window": windows[c], "tol": tol, "max_dist": MAX_DIST,
               "cutoff": cut_off, "seed": sd, "out": out, "extend": ext,
               "budget": min(D_BUDGET, max(600.0, deadline - reserve - time.time())),
               "min_ev": int(os.environ["EXC_MIN_EVIDENCE"]), "peel": os.environ["EXC_PEEL_MODE"]}
        jp = os.path.join(tmpdir, f"job_{c}_{v}_{st['ti']}_{st['k']}.json")
        json.dump(job, open(jp, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
        lp = os.path.join(os.environ.get("V13_DLOG_DIR", tmpdir), f"dworker_{qsha}_{c}_v{v}_{st['ti']}_{st['k']}.log")
        log(f"v13: block {c} view {v}: D worker tol {tol:g} cutoff {cut_off:g} seed {sd} budget {job['budget']:.0f}s -> {lp}")
        try:
            return subprocess.Popen([sys.executable, os.path.join(HERE, "dworker.py"), jp],
                                    stdout=open(lp, "w"), stderr=subprocess.STDOUT, cwd=HERE,
                                    env={**os.environ, **_block_env(c), "HQP_D_ENGINE": "ref"})
        except Exception as e:                                    # noqa: BLE001  (no fork / read-only tmp ...)
            log(f"v13: block {c} view {v}: cannot start a worker ({type(e).__name__}: {str(e)[:80]}) -> measuring in-process")
            try:
                import torch
                tb = lambda x: torch.tensor(x, dtype=torch.complex64, device="cuda")    # noqa: E731
                X.EXC_TWIN_TOL = tol
                os.environ["EXC_EXTEND"] = str(ext)
                cut, corr, binfo = ES.measure_D(gates, n, c, st["tau"], windows[c], tb, log, budget=job["budget"],
                                                cutoff=cut_off, seed=sd)
                res = {"ok": cut is not None, "info": {k_: v_ for k_, v_ in binfo.items() if k_ != "audit"}}
                if cut is not None:
                    res.update({"s": list(cut.s), "e": list(cut.e), "tau": list(st["tau"]), "corr": corr})
                pickle.dump(res, open(out, "wb"))
            except Exception as e2:                               # noqa: BLE001
                log(f"v13: block {c} view {v}: in-process D failed ({type(e2).__name__}: {str(e2)[:80]})")
            return None

    def collect(cv):
        c, v = cv
        st = state[cv]
        try:
            res = pickle.load(open(st["out"], "rb"))
        except Exception as e:                                    # noqa: BLE001
            res = {"ok": False, "info": {}, "error": f"no result ({type(e).__name__})"}
        tol, _cz, _se, ext = st["tols"][st["ti"]]
        # defence in depth: a worker result whose operator is structurally not a D (unitarity < 0.5 or model err > 0.9)
        # is a FAILED draw whatever the worker's own verdict said (MEASURED 2026-09-24 on h35 3460: the reference worker
        # returned ok=True for unitarity 0.00 / model err 1.000 through the residual rescue, and the block was excised)
        _inf = res.get("info", {}) or {}
        _cg = os.environ.get("V13_COLLECT_STRUCTURAL_GATE", "0").strip() not in ("0", "false", "no", "off", "")   # off: see h35
        # (unitary is NOT an operator property, see excision_solver's rescue note: model error only)
        if _cg and res.get("ok") and (_inf.get("model_err") is not None and float(_inf["model_err"]) > 0.9):
            res = {"ok": False, "info": _inf}
        if res.get("ok"):
            if [res["s"], res["e"]] == list(st["tols"][st["ti"]][2]):
                os.environ["EXC_EXTEND"] = str(ext)
                cut, _tw = X.envelope_cut(gates, n, c, st["tau"], window=windows[c], tol=tol, log=lambda m: None)
                st["result"] = (cut, res["corr"], res["info"], tol)
                log(f"v13: block {c} view {v}: D ok at tol {tol:g}: {res['info'].get('cz_excised')} CZ, model err "
                    f"{res['info'].get('model_err', -1):.3f}, phase side {res['info'].get('phase_side', '?')}, "
                    f"{res['info'].get('secs', 0):.0f}s")
                return
            res = {"ok": False, "info": res.get("info", {}), "error": "cut mismatch"}
        log(f"v13: block {c} view {v}: D draw FAILED at tol {tol:g} (model err {res.get('info', {}).get('model_err')}, "
            f"{res.get('error', 'gate')})")
        st["k"] += 1
        inf = res.get("info", {}) or {}
        structural = (inf.get("unitary") is not None and inf.get("unitary") < 0.5) or \
                     (inf.get("model_err") is not None and inf.get("model_err") > 0.9) or \
                     bool(inf.get("xcheck_failed"))          # D not stable across cutoffs: the CUT is the problem
        if structural and not res.get("error"):
            # the absorbed operator is NOT a product at all (unitarity ~0 / model err ~1): the CUT is wrong, not the
            # draw -- another seed re-measures the same wrong operator (~15-60 min each). Next cut option now.
            log(f"v13: block {c} view {v}: structural failure (unitary {inf.get('unitary')}, model err {inf.get('model_err')}) "
                f"-> skipping the remaining draws of this cut")
            st["k"] = D_DRAWS
        if st["k"] >= D_DRAWS:                        # this cut does not measure: fall back to the next cut option
            st["k"] = 0
            st["ti"] += 1

    # the readout keeps at most READOUT_RESERVE, but never more than 45 % of whatever wall this call was given
    reserve = min(READOUT_RESERVE, 0.45 * max(deadline - t0, 0.0))
    soft_s = float(os.environ.get("V13_D_SOFT_S", "3600"))
    active = {}                                   # (block, view) -> Popen

    def pump():
        """Non-blocking scheduler step: reap finished workers, start pending ones. Returns True while work remains."""
        for cv, p in list(active.items()):
            if p.poll() is not None:
                del active[cv]
                p2 = _reap(cv, p, relaunch=True)
                if p2 is not None:
                    active[cv] = p2
        todo = [cv for cv in state if state[cv]["tols"] and state[cv]["result"] is None
                and state[cv]["ti"] < len(state[cv]["tols"]) and cv not in active]
        for cv in todo:
            # another view of the same block is measuring exactly this (cut option, draw) right now: wait for its file
            twin_busy = any(o != cv and o[0] == cv[0] and o in active and state[o]["ti"] == state[cv]["ti"]
                            and state[o]["k"] == state[cv]["k"] for o in state)
            if twin_busy:
                continue
            if len(active) >= (D_MAX_PROCS if D_PARALLEL else 1) or time.time() > deadline - reserve:
                break
            p = launch(cv)
            if p is None:                          # cache hit or in-process measurement: result file is there
                collect(cv)
            else:
                active[cv] = p
        return bool(active) or bool([cv for cv in state if state[cv]["tols"] and state[cv]["result"] is None
                                     and state[cv]["ti"] < len(state[cv]["tols"])])

    def have_primary():
        return all(any(state[(c, v)]["result"] for v in range(D_VIEWS)) for c in cs if cut_opts[c])

    # The readout starts as soon as EVERY block has one measurement and either all views are in or the soft limit
    # has passed; slower views (a different seed can take 2x longer -- MEASURED) keep running in the background and
    # join the ensemble through extra_views() if they finish in time.
    # GRACE: MEASURED in the production rehearsal (4 workers, idle GPU): the plain cuts finish in ~35-40 min, the
    # edge-extended cuts of the same blocks need ~2x that (their absorption stalls more). Once every block has ONE
    # measurement the stragglers get V13_D_GRACE_S more, then the readout starts without them (they still join the
    # ensemble if they make it) -- waiting the full soft limit bought +30 % w0 for +40 min of wall.
    grace_s = float(os.environ.get("V13_D_GRACE_S", "900"))
    t_primary = None
    while pump():
        if time.time() > deadline - reserve:
            break
        if have_primary():
            t_primary = t_primary or time.time()
            if time.time() > t_primary + grace_s or time.time() > t0 + soft_s:
                log(f"v13: every block has a measurement ({time.time()-t0:.0f}s); {len(active)} straggler(s) keep running "
                    f"-> readout starts, they may still join the ensemble")
                break
        time.sleep(2.0)

    def assemble():
        views_ = []
        for v in range(D_VIEWS):
            pick = []
            for c in cs:
                r = state[(c, v)]["result"] or next((state[(c, w)]["result"] for w in range(D_VIEWS)
                                                     if state[(c, w)]["result"]), None)
                if r is not None:
                    pick.append((c, r))
            sig = [(c, tuple(r[0].s), tuple(r[0].e), r[2].get("seed"), r[2].get("cutoff")) for c, r in pick]   # same cut + same draw = same view
            if pick and all(sig != s_ for s_, _ in views_):
                views_.append((sig, pick))
        return [p_ for _, p_ in views_]

    views = assemble()
    used = [c for c, _ in views[0]] if views else []
    log(f"v13: excised {used} of {cs}; {len(views)} view(s); D err "
        f"{[[round(r[2].get('model_err', -1), 3) for _, r in vw] for vw in views]}; "
        f"cut tol {[r[3] for _, r in views[0]] if views else []}; D stage {time.time()-t0:.0f}s")

    def extra_views():
        """Called by the readout right before the ensemble: whatever other views are complete by now."""
        pump_ok = True
        try:
            for cv, p in list(active.items()):
                if p.poll() is not None:
                    del active[cv]
                    _reap(cv, p, relaunch=False)
        except Exception as e:                                    # noqa: BLE001
            pump_ok = False
            log(f"v13: late view collection failed ({type(e).__name__})")
        vs = assemble() if pump_ok else views
        return [([r[0] for _, r in vw], [r[1] for _, r in vw]) for vw in vs[1:]]

    def stop_workers():
        for cv, p in list(active.items()):
            try:
                p.kill()
            except Exception:                                     # noqa: BLE001
                pass
        active.clear()

    # D self-consistency between views: per-wire fidelity of the 1q blocks and agreement of the two-body terms.
    # Two independent measurements (different cutoff/seed rung) that agree are evidence the D is RIGHT, not
    # merely product-like (the old gate only checked that a product model fits the measured MPO).
    d_agree = None
    if len(views) >= 2:
        try:
            import numpy as np
            d_agree = []
            for (c, ra), (_c2, rb) in zip(views[0], views[1]):
                if (list(ra[0].s), list(ra[0].e)) != (list(rb[0].s), list(rb[0].e)):
                    log(f"v13: block {c}: the two views use DIFFERENT cuts ({len(ra[0].cz_in())} vs {len(rb[0].cz_in())} CZ) "
                        f"-> their D's are not comparable gate by gate; agreement is judged by the readouts")
                    continue
                ca, cb = ra[1], rb[1]
                fids = []
                for w in range(n):
                    Ma = np.array([complex(e[0], e[1]) for e in ca[("A", w)]]).reshape(2, 2)
                    Mb = np.array([complex(e[0], e[1]) for e in cb[("A", w)]]).reshape(2, 2)
                    fids.append(abs(np.trace(Ma.conj().T @ Mb)) / 2)
                ta = {k_[1:]: v_ for k_, v_ in ca.items() if isinstance(k_, tuple) and k_[0] == "cz"}
                tb_ = {k_[1:]: v_ for k_, v_ in cb.items() if isinstance(k_, tuple) and k_[0] == "cz"}
                d_agree.append({"block": c, "min_1q_fid": float(min(fids)), "mean_1q_fid": float(np.mean(fids)),
                                "two_body_same": sorted(ta) == sorted(tb_), "n_two_body": (len(ta), len(tb_))})
                log(f"v13: block {c}: D views agree: 1q fidelity min {min(fids):.4f} mean {np.mean(fids):.4f}, "
                    f"two-body terms {len(ta)} vs {len(tb_)} ({'same pairs' if sorted(ta) == sorted(tb_) else 'DIFFERENT pairs'})")
        except Exception as e:                                    # noqa: BLE001
            log(f"v13: D view comparison failed ({type(e).__name__}: {str(e)[:80]})")

    if os.environ.get("V13_D_ONLY", "0") == "1":      # experiments: fill the D cache (latency-bound stage) and stop
        while active and time.time() < deadline - 60:            # D-only runs wait for EVERY view
            pump(); time.sleep(5.0)
        stop_workers()
        views = assemble()
        return None, {"method": "v13", "d_only": True, "used": used,
                      "derr": [[r[2].get("model_err") for _, r in vw] for vw in views]}

    if not used:
        # nothing was excised (no mirror blocks found, or every D failed): this solver has nothing to add over the
        # legacy stages, which own the d1/d2-shaped circuits -- say so instead of answering from the raw circuit.
        stop_workers()
        log("v13: no block excised -> handing the circuit back to the legacy stages")
        return None, {"method": "v13", "blocks": cs, "blocks_found": len(cs), "used": [], "secs": time.time() - t0}

    # ---------------- 3. readouts + decision ----------------
    cuts = [r[0] for _, r in views[0]] if views else []
    corrs = [r[1] for _, r in views[0]] if views else []
    try:
        out = STG.run(gates, n, cuts, corrs, log, deadline, truth=truth, publish=publish, extra_views=extra_views)
    finally:
        stop_workers()
    answer, verdict = STG.decide(out["candidates"], n)
    # THE D MEASUREMENT'S OWN ERROR is a trust condition of its own (V13_DERR_MAX=0 disables). The gate that decides
    # whether to excise a block at all is deliberately lax (0.35) -- a block with a poor D is still better excised
    # than left in. But MEASURED across every full-budget run: correct answers come from blocks whose decoded D fits
    # the product+2-body model to 0.000-0.025, while the one wrong answer (a real sample hardened with ~30 % more
    # insertions) came from blocks at 0.069-0.072. Note the cutoff CROSS-CHECK passes there (1q fidelity 0.996-0.999,
    # same pi-terms): a consistently slightly-wrong D is stable, so stability alone cannot catch this.
    derr_max = float(os.environ.get("V13_DERR_MAX", "0.05"))
    # MEASURED 2026-09-24 (h35, validator flags): a block on the reference path with a WRONG tau came back with model
    # err 1.000 (the product ansatz is meaningless there) but the residual correction reproduced the operator to
    # |<I,R>| 0.9996, the readout answered with z 9.0 / 8 of 8 votes and the answer was RIGHT (hamming 0) -- yet this
    # gate flagged it UNTRUSTED. model_err judges the ansatz; what is emitted is ansatz + residual, which the overlap
    # gate below scores. So (V13_DERR_SKIP_CORRECTED=1, default) blocks WITH overlap evidence are judged by the overlap
    # gate only; the model-error gate keeps guarding blocks that have no correction evidence.
    _skip_corr = os.environ.get("V13_DERR_SKIP_CORRECTED", "1").strip() not in ("0", "false", "no", "off", "")
    derrs = [r[2].get("model_err") for vw in views for _c, r in vw if r[2].get("model_err") is not None
             and not (_skip_corr and r[2].get("resid_fid1") is not None)]
    if derr_max > 0 and derrs and max(derrs) > derr_max:
        verdict = dict(verdict)
        verdict["fails"] = list(verdict.get("fails") or []) + [f"worst block D error {max(derrs):.3f} > {derr_max}"]
        verdict["why"] = f"{verdict.get('why', '')} -- UNTRUSTED: worst block D error {max(derrs):.3f} > {derr_max}"
        verdict["trusted"] = False
        log(f"v13: the worst block D error is {max(derrs):.3f} (> {derr_max}): the reduction is not reliable enough "
            f"to certify this answer")
    # THE OPERATOR OVERLAP |<I,R>| is a strictly better D-quality metric than model_err, and it is what the readout
    # actually applies. MEASURED 2026-09-21 on the exact n=10 control (clean/t_e2e_resid2.py): an instance came back
    # with model_err 0.0000 while |<I,R>| was 0.997 and the true end-to-end L1 was 0.080 -- model_err is computed from
    # a handful of amplitude ratios and is blind to a non-diagonal two-body residue; the overlap is the whole operator.
    # This reads the value AFTER the residual correction, so it scores what is emitted, not what was decoded.
    rmin = float(os.environ.get("V13_RESID_MIN", "0.97"))
    # NB dict.get(k, default) returns None when k EXISTS with value None -- the default is not used -- so this
    # cannot be written as a one-liner without risking a TypeError in min() at the very end of the solve.
    fids = []
    nblk = 0
    for _vw in views:
        for _c, _r in _vw:
            nblk += 1
            _f = _r[2].get("resid_fid1")
            if _f is None:
                _f = _r[2].get("resid_fid0")
            if _f is not None:
                fids.append(float(_f))
    if nblk and len(fids) < nblk:
        # Since 2026-09-24 the correction runs on BOTH D paths (the reference absorber's two-frame operator is
        # realigned first, ref_align.py; HQP_D_RESID_REF=0 turns that off). A block without overlap evidence is
        # one where the correction was disabled, refused (realignment or R truncation above HQP_RESID_MAX_LOST)
        # or failed -- say so rather than pass silently.
        log(f"v13: {nblk - len(fids)}/{nblk} block measurements have no operator-overlap evidence (residual "
            f"correction disabled, refused or failed on them) -- those blocks are gated on the model error alone")
    if rmin > 0 and fids and min(fids) < rmin:
        verdict = dict(verdict)
        verdict["fails"] = list(verdict.get("fails") or []) + [f"worst block |<I,R>| {min(fids):.4f} < {rmin}"]
        verdict["why"] = f"{verdict.get('why', '')} -- UNTRUSTED: worst block |<I,R>| {min(fids):.4f} < {rmin}"
        verdict["trusted"] = False
        log(f"v13: the worst block operator overlap is {min(fids):.4f} (< {rmin}): even after the residual "
            f"correction the emitted D does not reproduce the measured one well enough to certify this answer")
    info = {"method": "v13", "blocks": cs, "blocks_found": len(cs), "used": used, "views": len(views),
            "trusted": bool(verdict.get("trusted")), "margin": verdict.get("margin"), "w0": verdict.get("w0"),
            "d_agree": d_agree,
            "verdict": verdict, "candidates": [{k: v for k, v in c.items() if k != "top"} for c in out["candidates"]],
            "secs": time.time() - t0}
    return answer, info
