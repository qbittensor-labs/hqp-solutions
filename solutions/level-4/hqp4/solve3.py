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

"""hqp4 v3 route (2026-10-14 night): iterative excision portfolio.

    undo 0.50 + Z fold  ->  repeat { candidates = census modules (involutive) + SWAP networks (CZ-triple matching) of the
    CURRENT circuit, each decoded at a module window (--mod-win) and SWAP networks also at the whole circuit (lockstep);
    all decoded at once; after the first success the others get a grace period; the best excision wins (lowest error
    class, then most CZ removed) }  until the circuit stops shrinking or is small  ->  p9solve ladder -> readout.

Why (overnight_20261013/LOG.md): v1/v2 run a fixed module -> body sequence. On generated d4-like seeds with masked
modules the census finds no clean module, the first network to finish gave a poor cut (err 6.3e-3, 336 CZ), and the
whole-circuit lockstep then ground for its whole budget. Every excision accepted here is exact up to its measured model
error, so applying them greedily in any order preserves the peak; the loop only chooses which ones and when to stop.
d4 (module, then the SWAP-network body) is the two-round special case.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hqp4.swapnet_solve import (CENSUS, FOLD_Z, L3_SCRIPTS, MKRED, MKRED_CWD, P9SOLVE, UNDO_WRAP, WINNER_SRC, find_swapnet,
                                find_swapnets, parse_qasm, run, sha)


def ncz(q):
    return sum(1 for t, _, _ in parse_qasm(q)[1] if t == "cz")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm"); ap.add_argument("--out", required=True); ap.add_argument("--truth-dir")
    ap.add_argument("--threads", default="3"); ap.add_argument("--undo-threshold", default="0.50")
    ap.add_argument("--min-frac", type=float, default=0.3, help="census modules: twin fraction gate (involutive only)")
    ap.add_argument("--mod-win", type=int, default=1200)
    ap.add_argument("--min-cover", type=float, default=0.9)
    ap.add_argument("--par", type=int, default=6, help="decodes at once per round")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--target-cz", type=int, default=150, help="stop excising below this CZ count")
    ap.add_argument("--budget", type=float, default=1200, help="wall budget per decode (s)")
    ap.add_argument("--grace", type=float, default=60, help="after the first success, wait max(grace, 0.5 x its time)")
    ap.add_argument("--cut-policy", default="best", choices=("best", "lowest-err"),
                    help="best = lowest error class then most CZ (v3); lowest-err = smallest model error among successes")
    ap.add_argument("--p9-seed", default="7", help="p9solve sampling / routing seed")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.qasm).stem; log = out / "solve3.log"; thr = int(a.threads)
    env = dict(os.environ, WINNER_SRC=str(WINNER_SRC), PYTHONPATH=str(WINNER_SRC), OMP_NUM_THREADS=str(thr),
               OPENBLAS_NUM_THREADS=str(thr), VECLIB_MAXIMUM_THREADS=str(thr), MKL_NUM_THREADS=str(thr),
               EXC_MAX_DIST="1500", MKRED_MAX_ERR="0.1", MKRED_BUDGET=str(a.budget), SYNCD_BUDGET_S=str(a.budget))
    res = {"qasm": str(a.qasm), "sha256": sha(a.qasm), "stages": [], "route": "solve3"}
    T0 = time.time()

    def stage(name, **kw):
        kw["name"] = name; kw["t"] = round(time.time() - T0, 1); res["stages"].append(kw)
        (out / "result.json").write_text(json.dumps(res, indent=1))
        print(f"[{kw['t']:7.1f}s] {name}: " + ", ".join(f"{k}={v}" for k, v in kw.items() if k not in ("name", "t")), flush=True)

    # 1. undo + fold
    ud = out / f"{stem}_undo"; uf = out / f"{stem}_uf.qasm"
    rc, dt = run([a.python, UNDO_WRAP, L3_SCRIPTS, "--source", a.qasm, "--output", ud, "--threshold", a.undo_threshold,
                  "--allow-derived-source"], log, label="undo")
    if rc or not (ud / "reduced.qasm").exists():
        stage("undo", rc=rc, error="undo failed"); return 2
    run([a.python, FOLD_Z, ud / "reduced.qasm", uf], log, label="fold_z")
    n = parse_qasm(a.qasm)[0]
    stage("undo+fold", cz_before=ncz(a.qasm), cz_after=ncz(uf), secs=round(time.time() - T0, 1))
    cur, pinvs = uf, []

    # 2. iterative excision portfolio
    for r in range(a.rounds):
        _, g = parse_qasm(cur); N = len(g); c0 = ncz(cur)
        if c0 <= a.target_cz:
            stage(f"round {r}", note=f"{c0} CZ <= target {a.target_cz}: stop"); break
        cj = out / f"{stem}_r{r}_census.json"
        run([a.python, CENSUS, cur, cj, "--directed", "--tol", "1e-3"], log, env=env, label=f"census r{r}")
        blocks = json.load(open(cj))["blocks"] if cj.exists() else []
        inv = lambda t: all(t[t[x]] == x for x in range(len(t)))      # noqa: E731
        mods = [b for b in blocks if inv(b["tau"]) and b.get("frac", 0) >= a.min_frac]
        span_max = max(600, int(0.12 * N)); net_win = max(1200, int(0.24 * N))
        nets = find_swapnets(n, g, span_max, a.min_cover, net_win)
        if not nets:
            sw6 = find_swapnet(n, g, span_max, 0.6, net_win)
            nets = [sw6] if sw6 and "error" not in sw6 else []
        cands = []                                                    # (kind, centre, tau, window)
        for b in sorted(mods, key=lambda b: -b.get("frac", 0)):
            cands.append(("census", b["centre"], b["tau"], a.mod_win))
        for w in nets:
            cands.append(("net", w["centre"], w["tau"], a.mod_win))
            cands.append(("net-whole", w["centre"], w["tau"], N))
        for b in mods:
            near = [w for w in nets if abs(w["centre"] - b["centre"]) <= 400]
            if near:
                w = min(near, key=lambda w: abs(w["centre"] - b["centre"]))
                cands.append(("midpoint", (b["centre"] + w["centre"]) // 2, b["tau"], a.mod_win))
        uniq = []
        for c in cands:
            if all(not (abs(c[1] - u[1]) <= 20 and c[2] == u[2] and c[3] == u[3]) for u in uniq):
                uniq.append(c)
        cands = uniq[: a.par]
        stage(f"round {r} candidates", cz=c0, gates=N, census=[(b["centre"], round(b.get("frac", 0), 2)) for b in mods],
              nets=[(w["centre"], round(w["cover"], 2)) for w in nets], cands=[(k, c, w) for k, c, _, w in cands])
        if not cands:
            break
        per = max(1, thr // len(cands)); procs = []; rt0 = time.time()
        for k, (kind, c, tau, win) in enumerate(cands):
            cjk = out / f"{stem}_r{r}c{k}.json"
            cjk.write_text(json.dumps({"blocks": [{"centre": c, "tau": tau, "method": kind}]}))
            pre = out / f"{stem}_r{r}c{k}_red"
            e = dict(env, OMP_NUM_THREADS=str(per), OPENBLAS_NUM_THREADS=str(per), MKL_NUM_THREADS=str(per),
                     VECLIB_MAXIMUM_THREADS=str(per))
            if win >= N:
                e.update(EXC_MAX_DIST=str(max(1500, N)), MKRED_MAX_ERR="0.5")
            lf = open(out / f"mkred_r{r}c{k}.log", "w")
            p = subprocess.Popen([str(x) for x in (a.python, MKRED, cur, cjk, pre, f"{c}:{win}")], cwd=MKRED_CWD, env=e,
                                 stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(dict(k=k, kind=kind, c=c, win=win, pre=pre, p=p, lf=lf, t0=time.time()))
        done, first = [], None
        while procs:
            time.sleep(2)
            for d in list(procs):
                if d["p"].poll() is None:
                    continue
                procs.remove(d); d["lf"].close(); d["secs"] = round(time.time() - d["t0"], 1)
                jq = Path(str(d["pre"]) + ".json")
                if d["p"].returncode == 0 and jq.exists():
                    info = json.load(open(jq)); bl = info["blocks"][0]
                    d.update(ok=True, err=bl["model_err"], cz=bl["cz"], info=info)
                    first = first or time.time()
                else:
                    d.update(ok=False)
                done.append(d)
            if first and procs and time.time() - first > max(a.grace, 0.5 * (first - rt0)):
                for d in procs:
                    try:
                        os.killpg(d["p"].pid, 9)
                    except Exception:
                        pass
                    d["lf"].close(); d.update(ok=False, killed=True, secs=round(time.time() - d["t0"], 1)); done.append(d)
                procs = []
        ok = [d for d in done if d.get("ok")]
        cls = lambda e: 0 if e < 1e-2 else 1 if e < 0.1 else 2     # noqa: E731
        stage(f"round {r} results", results=[(d["kind"], d["c"], d["win"], d.get("ok"), d.get("err"), d.get("cz"), d["secs"])
                                            for d in sorted(done, key=lambda d: d["k"])])
        if not ok:
            break
        if a.cut_policy == "lowest-err":
            best = min(ok, key=lambda d: (d["err"], -d["cz"]))
        else:
            best = min(ok, key=lambda d: (cls(d["err"]), -d["cz"]))
        pinvs.append(best["info"]["Pinv"]); cur = Path(str(best["pre"]) + ".qasm")
        stage(f"round {r} applied", kind=best["kind"], centre=best["c"], win=best["win"], err=best["err"], cz_removed=best["cz"],
              cz_left=ncz(cur))
        if best["cz"] < 30:
            break

    # 3. p9solve ladder on the remainder; readout mapped back through every Pinv
    runs = out / "p9"; runs.mkdir(exist_ok=True)
    P = list(range(n))
    for pv in pinvs:
        P = [pv[P[w]] for w in range(n)]
    res["Pinv_total"] = P
    top, S = [], {}
    for cr in ("0.5", "0.52", "0.48"):
        tag = f"body_c{cr.replace('.', '')}"
        rc, dt = run([P9SOLVE, "--qasm", cur, "--outdir", runs, "--tag", tag, "--expected-bitstring", "", "--center-ratio", cr,
                      "--cutoff", "3e-4", "--seed", a.p9_seed, "--max-bond", "512", "--unswap-threshold", "5e5",
                      "--abort-after-no-progress-unswap-cycles", "4", "--early-stopping-gates", "100", "--no-parallel-rewire",
                      "--samples", "1000"], log, cwd=MKRED_CWD, env=env, label=f"p9solve centre {cr}")
        S = json.loads((runs / tag / "summary.json").read_text()) if (runs / tag / "summary.json").exists() else {}
        top = S.get("top_permuted_samples") or []
        stage("p9solve", centre=cr, cz=ncz(cur), rc=rc, secs=dt, termination=S.get("termination_reason"), samples=bool(top))
        if top:
            break
    if not top:
        stage("answer", error="no samples on any centre"); return 3
    bits, c1 = top[0][0], top[0][1]; c2 = top[1][1] if len(top) > 1 else 0
    ans = "".join(bits[P[w]] for w in range(n))
    res.update(answer_q0_left=ans, top1_count=c1, top2_count=c2, ratio=round(c1 / max(c2, 1), 2), answer_from="p9solve",
               termination=S.get("termination_reason"))
    stage("answer", ratio=res["ratio"])
    (out / "answer.txt").write_text(ans + "\n")
    if a.truth_dir:
        import hashlib
        mp = Path(a.truth_dir) / f"{stem}_meta.json"
        if mp.exists():
            t = json.loads(mp.read_text()).get("peaked_state", "")
            res["grade"] = "MATCH" if hashlib.sha256(ans.encode()).hexdigest() == hashlib.sha256(t.encode()).hexdigest() else "MISMATCH"
            stage("grade", result=res["grade"])
    res["total_secs"] = round(time.time() - T0, 1); (out / "result.json").write_text(json.dumps(res, indent=1))
    print(f"ANSWER (q0_left) {ans}  ratio {res['ratio']}  total {res['total_secs']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
