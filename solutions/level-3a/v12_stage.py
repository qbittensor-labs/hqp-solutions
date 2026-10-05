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

"""Readout stage for an excised d3 circuit: three structurally different opinions on one reduction.

  ladder : forward MPS of the whole reduced circuit (the v6-v9 readout) -- cheap, fragile (margin ~1.2).
  oper   : operator-level absorption of the un-excised mirror SHELL around a block junction, stopped
           adaptively when cancellation ends, the true base then evolved as a state. MEASURED on
           d3_s1: margin 2.77 / w0 2.0e-4 against the ladder's 1.2-1.7 / 5e-6..1e-5.
  pps    : Heisenberg-picture Pauli-path marginals -- a consistency GATE (agreement ratio), never an
           arbiter: marginals mis-decode bits whose non-peak part is biased, even when exact.

Why three: v6/v8/v9 all returned IncorrectFailure on the validator while passing both public samples.
Their only trust signal was agreement between draws that shared one cut AND one readout type, so a
readout that was confidently wrong (h0: every forward simulation 4-14 bits off) was certified.

Everything is env-tunable with defaults; nothing here touches the D-measurement.
"""
import math
import os
import time

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

import excise as X
import mps_torch
import op_readout as OPR
import sectioned as SEC

LADDER_CHIS = [int(x) for x in os.environ.get("V12_LADDER_CHIS", "512,1024,2048").split(",") if x.strip()]
CENTRES = [c.strip() for c in os.environ.get("V12_CENTRES", "0,1,mid").split(",") if c.strip()]
PROBE_S = float(os.environ.get("V12_PROBE_S", "600"))
PROBE_N = int(os.environ.get("V12_PROBE_N", "100"))
OP_EARLY_STOP = int(os.environ.get("V12_OP_EARLY_STOP", "100"))
OP_CUTOFF = float(os.environ.get("V12_OP_CUTOFF", "0.002"))
OP_MAXBOND = int(os.environ.get("V12_OP_MAXBOND", "2048"))
OP_FINAL_MAXBOND = int(os.environ.get("V12_OP_FINAL_MAXBOND", "2048"))
OP_MAX_S = float(os.environ.get("V12_OP_MAX_S", "5400"))
PPS_BUDGET = float(os.environ.get("V12_PPS_BUDGET_S", "1500"))
PPS_K = int(os.environ.get("V12_PPS_K", "2000000"))
RUNG_GROWTH = float(os.environ.get("V12_RUNG_GROWTH", "6.0"))     # predicted cost ratio between consecutive rungs (chi x2)
LATER_RESERVE = float(os.environ.get("V12_LATER_RESERVE", "6000"))   # wall kept for ensemble + Pauli-path after the ladder
BIG_CHI = int(os.environ.get("V12_BIG_CHI", "4096"))              # one last rung if the wall allows (0 = off)
ENS_MEMBERS = int(os.environ.get("V12_ENS_MEMBERS", "8"))        # 0 = no ensemble
# "auto" (default since v14): the largest ladder chi whose MEASURED rung time fits the ensemble budget for all
# members. The D stage used to cost ~50 min per circuit and left the ensemble ~1024; with the synchronous engine it
# costs seconds, so the members can be as accurate as the wall allows (MEASURED on d3_s2: margin 3.12 at chi 512,
# 3.29 at 1024, 3.61 at 2048, 4.18 at 4096 -- higher chi = a genuinely better opinion, not just a slower one).
ENS_CHI_ENV = os.environ.get("V12_ENS_CHI", "auto").strip().lower()
ENS_CHI = 1024 if ENS_CHI_ENV == "auto" else int(ENS_CHI_ENV)
# one worker: MEASURED on an otherwise idle RTX PRO 6000, a chi-1024 evolution saturates the GPU (163 s alone vs 370-470 s
# each when two run side by side), so a second worker buys nothing and costs memory
ENS_WORKERS = int(os.environ.get("V12_ENS_WORKERS", "1"))
ENS_POOLK = int(os.environ.get("V12_ENS_POOLK", "64"))
ENS_BUDGET = float(os.environ.get("V12_ENS_BUDGET_S", "5400"))
MAX_CENTRES = int(os.environ.get("V12_MAX_CENTRES", "1"))      # 0 = skip the operator-level readout      # full readouts to run (best probe first)


def build_reduced(gates, n, cuts, corrs):
    sections, Pinv = SEC.sectioned(gates, n, cuts, corrections=corrs) if cuts else ([list(gates)], list(range(n)))
    rq = QuantumCircuit(n)
    for sec in sections:
        for t, q, p in sec:
            rq.cz(*q) if t == "cz" else rq.u(*p, q[0])
    lens = [len(s) for s in sections]
    junctions = [sum(lens[:k + 1]) for k in range(len(lens) - 1)]
    return sections, Pinv, rq, junctions


def run(gates, n, cuts, corrs, log, deadline, truth="", publish=None, extra_views=None):
    """Returns dict(candidates=[...], sections, Pinv). Each candidate: name, answer (ORIGINAL frame),
    w0, margin, ham (if truth), extra. `publish(bits, meta)` is called as soon as any answer exists."""
    t0 = time.time()
    sections, Pinv, rq, junctions = build_reduced(gates, n, cuts, corrs)
    total = sum(len(s) for s in sections)
    log(f"v12: reduced {len(gates)}->{total} gates, sections {[len(s) for s in sections]}, "
        f"junctions {[round(j / max(total, 1), 3) for j in junctions]}")
    ham = lambda b: (sum(a != c for a, c in zip(b, truth)) if truth else -1)   # noqa: E731
    cands = []
    npub = [0]

    def _pub():
        """Publish the CURRENT decision over every opinion so far (not the latest opinion: a later opinion is
        not necessarily a better one). The counter makes each publish supersede the previous one."""
        if not publish:
            return
        try:
            a, v = decide(cands, n)
            if a:
                npub[0] += 1
                publish(a, dict(v, method="v13", n_opinions=len(cands), seq=npub[0]))
        except Exception as e:                                    # noqa: BLE001
            log(f"v12: publish failed ({type(e).__name__}: {str(e)[:80]})")

    # ---- ladder (time-aware: a rung runs only if its predicted cost leaves the later opinions their budget) ----
    rung_secs = {}

    def _rung(chi, reserve):
        left = deadline - time.time() - reserve
        pred = (max(rung_secs.values()) * RUNG_GROWTH) if rung_secs else 0.0
        if left <= 0 or pred > left:
            log(f"v12: ladder chi={chi} skipped (predicted {pred:.0f}s, {max(left, 0):.0f}s available before the reserve)")
            return False
        try:
            mps, dt = mps_torch.evolve(dag_to_circuit(circuit_to_dag(rq)), chi, log_every=0)
            top = mps_torch.topk(mps, beam=512, k=8)
            del mps
            torch.cuda.empty_cache()
        except Exception as e:                                    # noqa: BLE001
            log(f"v12: ladder chi={chi} FAILED ({type(e).__name__}: {str(e)[:100]})")
            torch.cuda.empty_cache()
            return False
        rung_secs[chi] = dt
        if not top or not all(math.isfinite(float(w_)) for _b, w_ in top[:2]):
            log(f"v12: ladder chi={chi} produced a non-finite state -> rung discarded")
            return False
        ans = X.map_bits_to_original(top[0][0], Pinv)
        c = {"name": f"ladder{chi}", "answer": ans, "w0": top[0][1],
             "margin": top[0][1] / max(top[1][1], 1e-300), "ham": ham(ans), "secs": dt,
             "top": [(X.map_bits_to_original(b, Pinv), w) for b, w in top]}
        cands.append(c)
        extra = ""
        if truth:
            hams = [ham(b) for b, _w in c["top"]]
            extra = f" top-8 hams {hams}"
        log(f"v12: ladder chi={chi}: {dt:.0f}s w0={c['w0']:.3e} margin={c['margin']:.2f} ham={c['ham']}{extra}")
        _pub()
        return True

    later = min(LATER_RESERVE, 0.5 * max(deadline - t0, 0.0))     # never reserve more than half of what this stage got
    for chi in LADDER_CHIS:
        _rung(chi, later)

    growth = {}

    def _growth(tag="ladder"):
        """Diagnostic (never a gate): how each top candidate's weight moves as chi grows. A truncated simulation
        UNDER-estimates the true peak, so the real peak's weight rises with chi while a truncation artefact
        saturates or falls. MEASURED on d3_s2: the truth went 6.2e-3 -> 9.9e-3 -> 1.6e-2 -> 2.5e-2 over
        chi 512..4096 (x4 over three doublings) while its rivals stayed within a factor of ~2."""
        rl = [c for c in cands if c["name"].startswith(tag) and c.get("top")]
        if len(rl) < 2:
            return
        rl.sort(key=lambda c: int(c["name"][6:]))
        best = rl[-1]
        for bits, _w in best["top"][:3]:
            ws = []
            for c in rl:
                w = next((float(w_) for b_, w_ in c["top"] if b_ == bits), None)
                ws.append("-" if w is None else f"{w:.2e}")
            g = None
            w0_, w1_ = next((float(w_) for b_, w_ in rl[0]["top"] if b_ == bits), None),                 next((float(w_) for b_, w_ in rl[-1]["top"] if b_ == bits), None)
            if w0_ and w1_:
                g = w1_ / w0_
            if bits == best["top"][0][0] and g is not None:
                growth[bits] = g
            log(f"v12: weight growth {'ANSWER  ' if bits == best['top'][0][0] else 'rival   '}"
                f"chi {[int(c['name'][6:]) for c in rl]}: {ws}" + ("" if g is None else f"  (x{g:.1f})"))
    _growth()
    for c in cands:
        if c["name"].startswith("ladder") and c.get("answer") is not None:
            for bits, g in growth.items():
                if X.map_bits_to_original(bits, Pinv) == c["answer"]:
                    c["growth"] = g

    # ---- ensemble of independent chain orders / gate orders ("more seeds must agree") ----
    # One truncated-state simulation is a biased coin (clean d1_s2: bond-128 argmax right ~2 times in 3);
    # the true peak is in every member's pooled top list and leads the pooled mean log-probability.
    if ENS_MEMBERS > 0 and time.time() < deadline - min(1200.0, 0.2 * max(deadline - t0, 0.0)):
        pool = None
        try:
            import collections
            import ens_pool
            flat = [x for sec in sections for x in sec]
            views = [{"qasm": X.to_qasm(n, flat), "Pinv": list(Pinv)}]
            ev = extra_views() if callable(extra_views) else (extra_views or [])
            for ecuts, ecorrs in ev:                             # other D measurements of the same blocks
                try:
                    esec, ePinv, _rq, _j = build_reduced(gates, n, ecuts, ecorrs)
                    views.append({"qasm": X.to_qasm(n, [x for sec in esec for x in sec]), "Pinv": list(ePinv)})
                except Exception as e:                            # noqa: BLE001
                    log(f"v12: extra view unusable ({type(e).__name__}: {str(e)[:80]})")
            # Members must be INDEPENDENT coins: MEASURED on d3_s1, the dag and raw gate orders of the sectioned circuit
            # give the identical state (margin 2.39, F 2.2e-4 twice), so only the CHAIN order (and the D view)
            # diversifies. One identity-chain member per view, every other member gets its own chain order.
            nv = len(views)
            # (no "optimised" chain orders: MEASURED on d3_s1 the spectral/annealed order was the WORST member --
            #  F 1.0e-4 / margin 1.69 against 1.7-3.3e-4 / 2.1-2.4 for random orders)
            seeds = [None] * nv + [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
            ens_chi = ENS_CHI
            if ENS_CHI_ENV == "auto":
                budget = min(ENS_BUDGET, max(deadline - time.time() - min(900.0, 0.15 * max(deadline - t0, 0.0)), 0.0))
                # the 1.15 head-room used to be the only slack; a rung measured under momentary load (another
                # process on the GPU) then pushed the choice a whole factor of 2 down. Members are pooled as they
                # finish and the round loop re-plans, so plan for ENS_MEMBERS but accept a chi whose FIRST HALF fits.
                need = ENS_MEMBERS * float(os.environ.get("V12_ENS_FIT_FRAC", "0.6")) * 1.15
                fits = [c for c in sorted(rung_secs) if rung_secs[c] * need <= budget]
                ens_chi = max(fits) if fits else min(rung_secs) if rung_secs else ENS_CHI
                log(f"v12: ens chi auto -> {ens_chi} (budget {budget:.0f}s, measured rungs "
                    f"{ {c: round(v) for c, v in sorted(rung_secs.items())} })")
            log(f"v12: ens: {len(views)} view(s), {ENS_MEMBERS} members at chi {ens_chi}")
            pool = ens_pool.EnsemblePool(views, n_workers=min(ENS_WORKERS, ENS_MEMBERS), threads=3, log=log)
            hard_dl = deadline - min(900.0, 0.15 * max(deadline - t0, 0.0))
            ens_end = min(hard_dl, time.time() + ENS_BUDGET)
            # MORE SEEDS UNTIL THEY AGREE (v14): a round that does not separate (z, votes) is not evidence -- add
            # another round of independent chain orders while the wall allows, pooled with the previous ones.
            rounds = int(os.environ.get("V12_ENS_ROUNDS", "3"))
            z_want = float(os.environ.get("V12_ENS_Z", "2.5"))
            vote_want = float(os.environ.get("V12_ENS_VOTE_FRAC", "0.75"))
            v, cl, L, votes = None, None, None, collections.Counter()
            for r_ in range(rounds):
                specs = [(i % nv, ens_chi, "dag", (seeds[i % len(seeds)] if r_ == 0 else 100 + 20 * r_ + i))
                         for i in range(ENS_MEMBERS)]
                pool.build(specs, poolk=ENS_POOLK, deadline=ens_end)
                if not pool.members:
                    break
                cl, mids, L = pool.score()
                v = ens_pool.verdict(cl, L)
                votes = collections.Counter(m["top"] for m in pool.members.values())
                nm = len(pool.members)
                sep = bool(v) and v.get("z", 0) >= z_want and votes.get(v["leader"], 0) >= vote_want * nm
                per = (sum(m["secs"] for m in pool.members.values()) / max(nm, 1)) / max(ENS_WORKERS, 1)
                nxt = time.time() + per * ENS_MEMBERS * 1.1
                if not sep and nxt > ens_end and r_ + 1 < rounds:
                    # no room for another round at this chi: a CHEAPER round is still worth more than nothing --
                    # members at a lower chi are independent opinions too (and the ladder measured their cost).
                    left = ens_end - time.time()
                    cheaper = [c_ for c_ in sorted(rung_secs) if c_ < ens_chi and rung_secs[c_] * ENS_MEMBERS * 1.15 <= left]
                    if cheaper:
                        log(f"v12: ens round {r_+1}: no separation and no wall at chi {ens_chi} -> {ENS_MEMBERS} more "
                            f"members at chi {max(cheaper)} (~{rung_secs[max(cheaper)] * ENS_MEMBERS:.0f}s)")
                        ens_chi = max(cheaper)
                        continue
                if sep or r_ + 1 >= rounds or nxt > ens_end:
                    if not sep and v:
                        log(f"v12: ens round {r_+1}: no separation (z {v.get('z', 0):.2f}, votes "
                            f"{votes.get(v['leader'], 0)}/{nm}) and no wall for another round")
                    break
                log(f"v12: ens round {r_+1}: z {v.get('z', 0) if v else 0:.2f}, votes {votes.get(v['leader'], 0) if v else 0}/{nm} "
                    f"-> {ENS_MEMBERS} more independent orders (~{per * ENS_MEMBERS:.0f}s)")
            if pool.members:
                if v:
                    cand = {"name": "ens", "answer": v["leader"], "w0": float("nan"), "margin": v.get("geo_margin", 0.0),
                            "z": v.get("z"), "wins": v.get("wins"), "members": v["members"],
                            "votes": votes.get(v["leader"], 0), "ham": ham(v["leader"]),
                            "member_margins": [round(m["margin"], 2) for m in pool.members.values()]}
                    cands.append(cand)
                    log(f"v12: ens: {cand['members']} members at chi {ens_chi}: leader ham={cand['ham']} z={cand['z']:.2f} "
                        f"geo-margin={cand['margin']:.2f} wins={cand['wins']:.2f} top-1 votes {cand['votes']}/{cand['members']} "
                        f"pool {len(cl)}")
                    if truth:
                        import numpy as _np
                        mean = L.mean(axis=1); order = list(_np.argsort(-mean))
                        rk = order.index(cl.index(truth)) if truth in cl else -1
                        log(f"v12: ens: TRUTH rank {rk} of {len(cl)} pooled ({'in pool' if truth in cl else 'NOT in pool'})")
                    _pub()
        except Exception as e:                                    # noqa: BLE001
            log(f"v12: ensemble failed ({type(e).__name__}: {str(e)[:140]})")
        finally:
            if pool is not None:
                try:
                    pool.close()
                except Exception:                                 # noqa: BLE001
                    pass
            torch.cuda.empty_cache()

    # ---- Pauli-path gate ----
    if cands and PPS_BUDGET > 0 and time.time() < deadline - min(600.0, 0.1 * max(deadline - t0, 0.0)):
        try:
            import pps2_gate as pps_gate                      # pp2 engine: 5-10x faster than pauli_prop, same numbers
            flat = [x for sec in sections for x in sec]
            z, done = pps_gate.marginals(pps_gate.compile_reduced(flat), n, K=PPS_K,
                                         budget_s=min(PPS_BUDGET, deadline - time.time() - 300), log=log)
            if done == n:
                sign_red = "".join("0" if v >= 0 else "1" for v in z)
                sign_ans = X.map_bits_to_original(sign_red, Pinv)
                weak = sorted(abs(float(v)) for v in z)[:6]
                log(f"v12: pps sign string: ham={ham(sign_ans)}; weakest |<Z>| {[round(w_, 4) for w_ in weak]}")
                for cand in cands:
                    cand["pps"] = pps_gate.agreement(z, X.map_truth_to_reduced(cand["answer"], Pinv))
                    log(f"v12: pps gate {cand['name']}: ratio {cand['pps']['ratio']:.3f}, "
                        f"disagreements {cand['pps']['sign_disagreements']}/{n}, strong wrong {cand['pps']['strong_wrong']}")
                if truth:
                    a = pps_gate.agreement(z, X.map_truth_to_reduced(truth, Pinv))
                    log(f"v12: pps gate TRUTH: ratio {a['ratio']:.3f}, disagreements {a['sign_disagreements']}/{n}")
        except Exception as e:                                    # noqa: BLE001
            log(f"v12: pps gate failed ({type(e).__name__}: {str(e)[:120]})")
            torch.cuda.empty_cache()
    def _operator_opinion():
        """Operator-level readout (absorb the residual mirror shell around a junction, state for the rest).
        With whole-block cuts the shells are thin, so this is a FALLBACK opinion: it costs 30-90 min."""
        # ---- centre probes ----
        centres = [c for c in CENTRES if c == "mid" or (junctions and int(c) < len(junctions))]
        kw_of = lambda c: ({"center_ratio": 0.5} if (c == "mid" or not junctions)            # noqa: E731
                           else {"junctions": junctions, "centre_junction": int(c)})
        if MAX_CENTRES > 0 and PROBE_S > 0 and len(centres) > 1:
            scored = []
            for c in centres:
                if time.time() > deadline - 2400:
                    break
                try:
                    pr = OPR.probe(rq, n_absorb=PROBE_N, seconds=PROBE_S, logger=log, **kw_of(c))
                except Exception as e:                                # noqa: BLE001
                    log(f"v12: probe centre {c} FAILED ({type(e).__name__}: {str(e)[:100]})")
                    torch.cuda.empty_cache()
                    continue
                # collapsed operator first, then more absorbed in the same time, then lower loss (a bare
                # loss-per-unitary score prefers probes that absorbed LESS -- measured on h0)
                scored.append(((0 if pr["bond"] <= 16 else 1, -pr["absorbed"], pr["loss_per_unitary"]), c))
                log(f"v12: probe centre {c}: absorbed {pr['absorbed']} in {pr['secs']:.0f}s, bond {pr['bond']}, "
                    f"log10 norm {pr['log10_norm']:.2f}")
            if scored:
                scored.sort(key=lambda t_: t_[0])
                centres = [c for _, c in scored]
                log(f"v12: centre order by probe: {centres}")

        # ---- operator-level readout ----
        for c in centres[:MAX_CENTRES]:
            left = deadline - time.time()
            if left < 1500:
                log("v12: no time left for an operator-level readout")
                break
            try:
                res = OPR.solve(rq, seed=123, cutoff=OP_CUTOFF, max_bond=OP_MAXBOND, final_maxbond=OP_FINAL_MAXBOND,
                                early_stop=OP_EARLY_STOP, deadline=time.time() + min(left - 900, OP_MAX_S),
                                logger=log, **kw_of(c))
            except Exception as e:                                    # noqa: BLE001
                log(f"v12: operator readout at centre {c} FAILED ({type(e).__name__}: {str(e)[:140]})")
                torch.cuda.empty_cache()
                continue
            res.pop("_state", None)
            torch.cuda.empty_cache()
            ans = X.map_bits_to_original(res["top"][0][0], Pinv)
            cand = {"name": f"oper@{c}", "answer": ans, "w0": res["w0"], "margin": res["margin"], "ham": ham(ans),
                    "absorbed": res["absorbed"], "total": res["total"], "rolled_back": res.get("rolled_back"),
                    "log10_norm": res["log10_norm"], "degenerate": res["degenerate"], "secs": res["secs"],
                    "top": [(X.map_bits_to_original(b, Pinv), w) for b, w in res["top"]]}
            cands.append(cand)
            log(f"v12: oper@{c}: w0={cand['w0']:.3e} margin={cand['margin']:.2f} ham={cand['ham']} "
                f"absorbed {cand['absorbed']}/{cand['total']} rolled_back={cand['rolled_back']} ({cand['secs']:.0f}s)")
            _pub()


    oper_mode = os.environ.get("V12_OPER_MODE", "fallback").strip().lower()      # fallback | always | off
    if MAX_CENTRES > 0 and oper_mode != "off":
        _a, _v = decide(cands, n)
        if oper_mode == "always" or not _v.get("trusted"):
            if time.time() < deadline - 2700:
                log(f"v12: opinions so far {'agree' if _v.get('trusted') else 'do NOT certify the answer'} -> operator-level readout")
                _operator_opinion()
            else:
                log("v12: no wall left for the operator-level fallback")
    _pub()
    if BIG_CHI and BIG_CHI not in rung_secs and rung_secs:
        _rung(BIG_CHI, 600.0)
    log(f"v12: stage done in {time.time()-t0:.0f}s with {len(cands)} candidates")
    return {"candidates": cands, "sections": sections, "Pinv": Pinv, "junctions": junctions}


def decide(cands, n, log=None):
    """Pick the answer and say whether to trust it.

    Opinions: ladder<chi> (forward MPS rungs), ens (pooled multi-order ensemble), oper@<c> (operator-level), each
    optionally carrying a Pauli-path agreement ratio.
    trusted <=> the (valid) ensemble leader equals the highest-chi rung, separates from its most dangerous rival
                (z >= V12_ENS_Z, top-1 votes >= half the members), the Pauli-path gate does not contradict it --
                or, without a valid ensemble, the two highest rungs agree with margin >= V12_LADDER_MARGIN.
    A BROKEN opinion must never outvote healthy ones: MEASURED on d3_s2, one NaN ensemble member made the pooled
    leader a Hamming-28 string with z = inf, and the first version of this rule preferred it to three agreeing rungs.
    """
    pps_min = float(os.environ.get("V12_PPS_MIN", "0.80"))
    ens_z = float(os.environ.get("V12_ENS_Z", "2.5"))
    ladder_margin = float(os.environ.get("V12_LADDER_MARGIN", "1.5"))
    oper_margin = float(os.environ.get("V12_OPER_MARGIN", "2.0"))
    fin = lambda v: v is not None and isinstance(v, (int, float)) and math.isfinite(v)      # noqa: E731
    opers = [c for c in cands if c["name"].startswith("oper") and not c.get("degenerate") and fin(c.get("margin"))]
    ladders = sorted([c for c in cands if c["name"].startswith("ladder") and fin(c.get("w0")) and fin(c.get("margin"))],
                     key=lambda c: -int(c["name"][6:]))
    ens = next((c for c in cands if c["name"] == "ens"), None)
    if ens is not None and not (fin(ens.get("z")) and fin(ens.get("margin")) and ens.get("votes", 0) >= 1):
        ens = None                                           # non-finite / voteless ensemble = no opinion
    if not opers and not ladders and not ens:
        return None, {"trusted": False}
    ratio = lambda c: (c.get("pps") or {}).get("ratio")             # noqa: E731
    pps_ok = lambda c: ratio(c) is None or ratio(c) >= pps_min      # noqa: E731
    top = ladders[0] if ladders else None
    rungs_agree = len(ladders) >= 2 and ladders[0]["answer"] == ladders[1]["answer"]
    if ens and top and ens["answer"] == top["answer"]:
        # v14 BAR (each condition env-settable; every failed one is named in the verdict):
        #   * every valid ladder rung agrees with the ensemble leader  (V12_RUNG_UNANIMITY=0 -> only the top rung)
        #   * the ensemble separates: z >= V12_ENS_Z and top-1 votes >= V12_ENS_VOTE_FRAC of the members
        #   * the best rung's margin >= V12_TRUST_MARGIN
        #   * the Pauli-path gate agrees (ratio >= V12_PPS_MIN) with no STRONG disagreement
        vote_frac = float(os.environ.get("V12_ENS_VOTE_FRAC", "0.75"))
        trust_margin = float(os.environ.get("V12_TRUST_MARGIN", "2.0"))
        strong_max = int(os.environ.get("V12_PPS_STRONG_WRONG_MAX", "0"))
        unanimity = os.environ.get("V12_RUNG_UNANIMITY", "1").strip() not in ("0", "false", "False")
        dissent = [c["name"] for c in ladders if c["answer"] != ens["answer"]]
        sw = (top.get("pps") or {}).get("strong_wrong")
        fails = []
        if unanimity and dissent:
            fails.append(f"rungs {','.join(dissent)} disagree")
        if ens["z"] < ens_z:
            fails.append(f"z {ens['z']:.2f} < {ens_z}")
        if ens.get("votes", 0) < vote_frac * ens.get("members", 1):
            fails.append(f"votes {ens.get('votes')}/{ens.get('members')} < {vote_frac:.0%}")
        if not fin(top.get("margin")) or top["margin"] < trust_margin:
            fails.append(f"margin {top.get('margin')} < {trust_margin}")
        if not pps_ok(top):
            fails.append(f"pauli-path ratio {ratio(top)}")
        if sw is not None and sw > strong_max:
            fails.append(f"pauli-path strong disagreements {sw} > {strong_max}")
        # WEIGHT GROWTH of the answer across the ladder (V12_GROWTH_MIN=0 disables). A truncated simulation
        # under-estimates the true peak, so the real peak's weight climbs steeply with chi while a peak produced by a
        # slightly WRONG reduction sits where it is. MEASURED over 512->2048 on nine full-budget runs: every correct
        # answer grew x2.08-3.12 (real samples x2.5-3.1, the hardest synthetic x2.08), while the one run with a bad D
        # (a real sample hardened with extra insertions, block D error 0.071) grew only x1.34 -- and was wrong.
        gmin = float(os.environ.get("V12_GROWTH_MIN", "1.5"))
        g_ = top.get("growth")
        if gmin > 0 and len(ladders) >= 3 and g_ is not None and g_ < gmin:
            fails.append(f"peak weight grew only x{g_:.2f} across the ladder (< x{gmin})")
        why = (f"ens == {top['name']} (z {ens['z']:.2f}, votes {ens.get('votes')}/{ens.get('members')}, "
               f"margin {top['margin']:.2f}, rungs {len(ladders) - len(dissent)}/{len(ladders)})")
        if fails:
            why += " -- UNTRUSTED: " + "; ".join(fails)
        if log:
            log(f"v12: trust check: {why}")
        return ens["answer"], {"trusted": not fails, "why": why, "margin": top["margin"], "w0": top["w0"],
                               "z": ens["z"], "pps_ratio": ratio(top), "fails": fails}
    for o in opers:
        for l in ladders:
            if o["answer"] == l["answer"]:
                return o["answer"], {"trusted": bool(pps_ok(o)), "why": f"{o['name']} == {l['name']}",
                                     "margin": o["margin"], "w0": o["w0"], "pps_ratio": ratio(o)}
    if rungs_agree:                                          # the ladder is self-consistent: it outranks a dissenting ensemble
        # v14: WITHOUT the ensemble, agreeing rungs alone are trusted only at a margin no historical failure ever had
        # (the v6/v8/v9 IncorrectFailures agreed across rungs at margin 1.2-1.6; a real d3 sits at 2.5-4.2 and a
        # trivially peaked d1/d2 circuit at 7.5-9.1). V12_LADDER_TRUST_MARGIN=0 restores the old behaviour.
        lad_trust = float(os.environ.get("V12_LADDER_TRUST_MARGIN", "5.0"))
        ok = ens is None and pps_ok(top) and top["margin"] >= max(ladder_margin, lad_trust)
        why = f"{ladders[0]['name']} == {ladders[1]['name']}" + ("" if ens is None else " (ensemble leader differs)")
        if not ok and ens is None:
            why += f" -- UNTRUSTED: no ensemble evidence and margin {top['margin']:.2f} < {max(ladder_margin, lad_trust)}"
        return top["answer"], {"trusted": bool(ok), "why": why,
                               "margin": top["margin"], "w0": top["w0"], "pps_ratio": ratio(top)}
    if ens and ens["z"] >= 1.5 and ens.get("votes", 0) * 2 >= ens.get("members", 1):
        return ens["answer"], {"trusted": False, "why": "ensemble leader (ladder rungs disagree)",
                               "margin": ens["margin"], "w0": ens.get("w0"), "z": ens["z"], "pps_ratio": ratio(ens)}
    pool = ([ens] if ens else []) + opers + ladders[:1]
    if len(pool) > 1 and all(ratio(c) is not None for c in pool):
        best = max(pool, key=ratio)
        why = f"readouts disagree; {best['name']} has the best pps ratio {ratio(best):.3f}"
    else:
        best = next((o for o in opers if o["margin"] >= oper_margin), None) or (ladders[0] if ladders else pool[0])
        why = f"single candidate {best['name']}"
    return best["answer"], {"trusted": False, "why": why, "margin": best.get("margin"), "w0": best.get("w0"),
                            "pps_ratio": ratio(best)}
