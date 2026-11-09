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

"""End-to-end level-4 route (2026-10-13, both d4 samples solved with it on the laptop CPU; see
level4/docs/01_HQP4_SAMPLES_20261013.md, 19:05 MDT): raw u/cz QASM -> answer (q0_left).

  1. undo inserted CZ couples at displacement threshold 0.50 (level3/scripts/undo_inserted_pairs.py through the
     register-size wrapper level4/sources/hqp4_d4_u010f/undo_wrap.py), then fold the Pauli-Z byproducts (fold_z.py);
  2. winner census, directed, tol 1e-3 (counter/_census_x.py); every block with an involutive tau and twin fraction
     >= --min-frac is excised by the winner reducer (research/priv/hard3/mkred_multi.py, lockstep measure_D) at
     window --mod-win;
  3. SWAP-network detection: wire pairs carrying three CZs within a span of <= --max-span gates; the window holding
     the most disjoint such triples gives tau2 (must be a fixed-point-free involution on >= --min-cover of the wires);
  4. lockstep excision with tau2 at the middle of the SWAP span, window = whole circuit;
  5. vendored p9solve (l4day p9 recipe) on the remainder (centre ladder 0.5 -> 0.52 -> 0.48 when a run ends without
     samples); readout mapped back through every stage's Pinv: original wire w = final qubit Pinv_k[...Pinv_1[w]];
  6. second instrument: quimb MPS state readout of the same remainder (cap --mps-bond), its top1 mapped the same way;
     reported as mps_agrees (an internal consistency check, not a second route family).
v2 (2026-10-13 night, overnight_20261013/LOG.md): census-free module fallback. Every SWAP network of the undo view is
found (find_swapnets: repeat the matcher with earlier spans blocked); when the census gives no excisable module, each
network is tried as a module (centre = its span middle, tau = its matching, window --mod-win then 1.5x). Measured on
s15: the matcher's module-1 tau equals the census tau on both samples, and the excision at the matcher centre 2101
decodes (model error 1.4e-4, 621 CZ) where the census centre 2195 read 2.4e-4 / 611. Windows scale with the circuit:
matcher window max(1200, 0.24 N), triple span max(600, 0.12 N) (the samples: 0.24 / 0.12 N, so unchanged there).
Truth (--truth-dir) is read only after the answer is written, for a hash grade.

usage (peakpilot env, from level4/):
  WINNER_SRC=<peakpilot>/research/p12/winner_run/src uv run --project <peakpilot> --extra torch \
      python -m hqpgen.counter.swapnet_solve <raw.qasm> --out <dir> [--truth-dir D] [--threads 3]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# submission layout (2026-10-13): every tool ships next to this file; HQP4_* env overrides for development runs
WINNER_SRC = Path(os.environ.get("WINNER_SRC", HERE / "winner"))
L3_SCRIPTS = Path(os.environ.get("HQP4_L3_SCRIPTS", HERE / "l3"))
UNDO_WRAP = Path(os.environ.get("HQP4_UNDO_WRAP", HERE / "undo_wrap.py"))
FOLD_Z = Path(os.environ.get("HQP4_FOLD_Z", HERE / "fold_z.py"))
CENSUS = Path(os.environ.get("HQP4_CENSUS", HERE / "census_x.py"))
MKRED = Path(os.environ.get("HQP4_MKRED", HERE / "mkred_multi.py"))
P9SOLVE = Path(os.environ.get("HQP4_P9SOLVE", "/opt/p9/bin/p9solve"))
MKRED_CWD = Path(os.environ.get("HQP4_MKRED_CWD", HERE))


def _ang(x):
    return float(eval(x.replace("pi", "math.pi"), {"__builtins__": {}}, {"math": math}))


def parse_qasm(path):
    n, g = None, []
    for ln in Path(path).read_text().splitlines():
        m = re.match(r"\s*qreg\s+\w+\[(\d+)\]", ln)
        if m:
            n = int(m.group(1)); continue
        m = re.match(r"\s*u\(([^,]+),([^,]+),([^)]+)\)\s+q\[(\d+)\];", ln)
        if m:
            g.append(("u", (int(m.group(4)),), tuple(_ang(m.group(i)) for i in (1, 2, 3)))); continue
        m = re.match(r"\s*cz\s+q\[(\d+)\],\s*q\[(\d+)\];", ln)
        if m:
            g.append(("cz", (int(m.group(1)), int(m.group(2))), None))
    return n, g


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd, log, cwd=None, env=None, label=""):
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n=== {label} ({time.strftime('%F %T')}): {' '.join(map(str, cmd))}\n"); f.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, round(time.time() - t0, 1)


def find_swapnet(n, g, max_span=600, min_cover=0.9, win=1200):
    """tau2 from CZ triples: for each pair, every run of three consecutive same-pair CZs within max_span gates is a
    candidate; the 1200-gate window with the most disjoint candidates wins (ties: the earliest)."""
    occ = collections.defaultdict(list)
    for i, (t, q, p) in enumerate(g):
        if t == "cz":
            occ[tuple(sorted(q))].append(i)
    cands = []   # (pair, (i1, i2, i3))
    for pr, ix in occ.items():
        for k in range(len(ix) - 2):
            if ix[k + 2] - ix[k] <= max_span:
                cands.append((pr, (ix[k], ix[k + 1], ix[k + 2])))
    if not cands:
        return None
    best = None
    for c in range(0, len(g), 50):
        lo, hi = c, c + win
        inside = [(pr, ix) for pr, ix in cands if lo <= ix[0] and ix[2] < hi]
        used, chosen = set(), []
        for pr, ix in sorted(inside, key=lambda t: t[1][0]):
            if pr[0] in used or pr[1] in used:
                continue
            used.update(pr); chosen.append((pr, ix))
        if best is None or len(chosen) > len(best[1]):
            best = (lo, chosen)
    lo, chosen = best
    cover = 2 * len(chosen) / n
    if cover < min_cover:
        return {"error": f"SWAP network covers only {cover:.2f} of the wires ({len(chosen)} pairs) in window [{lo},{lo+win})"}
    tau = list(range(n))
    for (a, b), ix in chosen:
        tau[a], tau[b] = b, a
    span = (min(ix[0] for _, ix in chosen), max(ix[2] for _, ix in chosen) + 1)
    return {"tau": tau, "pairs": len(chosen), "cover": cover, "span": span, "centre": (span[0] + span[1]) // 2,
            "window": [lo, lo + win], "fixed_points": [v for v in range(n) if tau[v] == v]}


def find_swapnets(n, g, max_span=600, min_cover=0.9, win=1200, kmax=4):
    """Every SWAP network: find_swapnet again with the CZs of the spans already found blocked, while cover holds."""
    used, nets = set(), []
    for _ in range(kmax):
        h = [(("u", q[:1], (0, 0, 0)) if (t == "cz" and i in used) else (t, q, p)) for i, (t, q, p) in enumerate(g)]
        r = find_swapnet(n, h, max_span, min_cover, win)
        if not r or "error" in r:
            break
        nets.append(r)
        used |= {i for i in range(*r["span"]) if g[i][0] == "cz"}
    return sorted(nets, key=lambda r: r["span"][0])


def mps_readout(qasm, bond, samples=400, seed=7):
    """quimb MPS state of a u/cz circuit from |0..0>; top-2 sample counts (bit i = qubit i, the p9solve convention)."""
    import quimb.tensor as qtn
    n, g = parse_qasm(qasm)
    c = qtn.CircuitMPS(n, gate_opts={"max_bond": bond, "cutoff": 1e-10})
    for t, q, p in g:
        c.apply_gate("U3", *p, q[0]) if t == "u" else c.apply_gate("CZ", q[0], q[1])
    cnt = collections.Counter(c.sample(samples, seed=seed)).most_common(2)
    top = ["".join(map(str, b)) if not isinstance(b, str) else b for b, _ in cnt]
    return {"top1": top[0], "c1": cnt[0][1], "c2": cnt[1][1] if len(cnt) > 1 else 0, "max_bond": int(c.psi.max_bond())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("qasm"); ap.add_argument("--out", required=True); ap.add_argument("--truth-dir")
    ap.add_argument("--threads", default="3"); ap.add_argument("--undo-threshold", default="0.50")
    ap.add_argument("--min-frac", type=float, default=0.5); ap.add_argument("--mod-win", type=int, default=1200)
    ap.add_argument("--max-span", type=int, default=0, help="0 = max(600, 0.12 N)")
    ap.add_argument("--min-cover", type=float, default=0.9)
    ap.add_argument("--net-win", type=int, default=0, help="matcher window; 0 = max(1200, 0.24 N)")
    ap.add_argument("--mps-bond", type=int, default=0, help="0 = no MPS second readout (v1: off, slow on the remainder)")
    ap.add_argument("--mod-par", type=int, default=4, help="module decode candidates run at once")
    ap.add_argument("--python", default=sys.executable, help="python of the peakpilot env (undo / fold / census / mkred)")
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.qasm).stem; log = out / "swapnet_solve.log"
    env = dict(os.environ, WINNER_SRC=str(WINNER_SRC), PYTHONPATH=str(WINNER_SRC), OMP_NUM_THREADS=a.threads,
               OPENBLAS_NUM_THREADS=a.threads, VECLIB_MAXIMUM_THREADS=a.threads, MKL_NUM_THREADS=a.threads)
    env.setdefault("EXC_MAX_DIST", "1500"); env.setdefault("MKRED_MAX_ERR", "0.1")
    # decode wall budgets (winner default 900 s): s16 module 1 needed 431-842 s on the laptop at 3 threads and the
    # reversed s16 585 s, so 900 s is too close on a slower validator CPU (overnight_20261013/LOG.md)
    env.setdefault("MKRED_BUDGET", "2400"); env.setdefault("SYNCD_BUDGET_S", "2400")
    res = {"qasm": str(a.qasm), "sha256": sha(a.qasm), "stages": [], "started": time.strftime("%F %T %Z")}
    T0 = time.time()

    def stage(name, **kw):
        kw["name"] = name; kw["t"] = round(time.time() - T0, 1); res["stages"].append(kw)
        (out / "result.json").write_text(json.dumps(res, indent=1)); print(f"[{kw['t']:7.1f}s] {name}: " + ", ".join(f"{k}={v}" for k, v in kw.items() if k not in ("name", "t")), flush=True)

    # 1. undo + fold
    ud = out / f"{stem}_undo{a.undo_threshold.replace('.', '')}"; uf = out / f"{stem}_u{a.undo_threshold.replace('.', '')}f.qasm"
    rc, dt = run([a.python, UNDO_WRAP, L3_SCRIPTS, "--source", a.qasm, "--output", ud, "--threshold", a.undo_threshold, "--allow-derived-source"], log, label="undo")
    u = ud / "reduced.qasm"   # the undo tool writes a directory: reduced.qasm, certificate.json, summary.json
    if rc or not u.exists():
        stage("undo", rc=rc, error="undo failed"); return 2
    rc, dt2 = run([a.python, FOLD_Z, u, uf], log, label="fold_z")
    n, g0 = parse_qasm(a.qasm); n, g = parse_qasm(uf)
    stage("undo+fold", cz_before=sum(1 for t, _, _ in g0 if t == "cz"), cz_after=sum(1 for t, _, _ in g if t == "cz"), secs=dt + dt2, sha=sha(uf)[:16])

    # 2. census + module excision
    cj = out / f"{stem}_census_1e-3.json"
    rc, dt = run([a.python, CENSUS, uf, cj, "--directed", "--tol", "1e-3"], log, env=env, label="census")
    blocks = json.load(open(cj))["blocks"] if cj.exists() else []
    mods = [b for b in blocks if b.get("involution") and b.get("frac", 0) >= a.min_frac]
    stage("census", rc=rc, secs=dt, blocks=[(b["centre"], round(b.get("frac", 0), 2), bool(b.get("involution"))) for b in blocks], excise=[b["centre"] for b in mods])
    cur, pinvs = uf, []
    N = len(g)
    span_max = a.max_span or max(600, int(0.12 * N)); net_win = a.net_win or max(1200, int(0.24 * N))
    nets = find_swapnets(n, g, span_max, a.min_cover, net_win)
    stage("swapnets (undo view)", n=len(nets), nets=[(r["pairs"], round(r["cover"], 2), r["span"], r["centre"]) for r in nets],
          span_max=span_max, net_win=net_win)
    excised = False
    # module stage = parallel portfolio of decodes (2026-10-13 night): the decodable centre differs per circuit
    # (s16: census centre 2210 decodes in 431-842 s, matcher centre 2088 REJECTED at 1259 s; reversed s16: census 2920 /
    # 2950 REJECTED at ~1200 s, matcher 2778 decodes in 585 s) and a failing decode costs its whole budget, so the
    # candidates run at once and the first one that excises wins (the others are killed).
    cands = []
    for b in mods:
        cands.append(("census", b["centre"], b["tau"]))
    near = lambda r: any(abs(r["centre"] - b["centre"]) <= 400 for b in mods)      # noqa: E731
    for r in nets:
        if not mods or near(r):
            cands.append(("network", r["centre"], r["tau"]))
    for b in mods:
        rs = [r for r in nets if abs(r["centre"] - b["centre"]) <= 400]
        if rs:
            r = min(rs, key=lambda r: abs(r["centre"] - b["centre"]))
            cands.append(("midpoint", (b["centre"] + r["centre"]) // 2, b["tau"]))
    uniq = []
    for c in cands:
        if all(not (abs(c[1] - u[1]) <= 20 and c[2] == u[2]) for u in uniq):
            uniq.append(c)
    cands = uniq[: a.mod_par]

    def portfolio(cands, win_size, tag):
        """run every candidate at once; first excision wins; returns (winner tuple or None, failures)"""
        per = max(1, int(a.threads) // len(cands))
        procs = []
        for k, (kind, c, tau) in enumerate(cands):
            cjk = out / f"{stem}_{tag}{k}_census.json"
            cjk.write_text(json.dumps({"blocks": [{"centre": c, "tau": tau, "method": kind}]}))
            pre = out / f"{stem}_m0red_{tag}{k}"
            envk = dict(env, OMP_NUM_THREADS=str(per), OPENBLAS_NUM_THREADS=str(per), MKL_NUM_THREADS=str(per),
                        VECLIB_MAXIMUM_THREADS=str(per))
            lf = open(out / f"mkred_{tag}{k}.log", "w")
            pr = subprocess.Popen([str(x) for x in (a.python, MKRED, cur, cjk, pre, f"{c}:{win_size}")], cwd=MKRED_CWD,
                                  env=envk, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append((k, kind, c, pre, pr, lf, time.time()))
        stage(f"module portfolio {tag}", candidates=[(kind, c) for kind, c, _ in cands], window=win_size, threads_each=per)
        win, fails = None, []
        while procs and win is None:
            time.sleep(2)
            for t in list(procs):
                k, kind, c, pre, pr, lf, t0 = t
                if pr.poll() is None:
                    continue
                procs.remove(t); lf.close()
                if pr.returncode == 0 and Path(str(pre) + ".qasm").exists():
                    win = t
                    break
                fails.append((kind, c, pr.returncode, round(time.time() - t0, 1)))
        for k, kind, c, pre, pr, lf, t0 in procs:
            try:
                os.killpg(pr.pid, 9)
            except Exception:
                pass
            lf.close()
        return win, fails

    if cands:
        win, fails = portfolio(cands, a.mod_win, "mod")
        if win is None:        # v2: every candidate failed at the base window -> one more round at 1.5x
            win, fails2 = portfolio(cands, int(1.5 * a.mod_win), "modw")
            fails += fails2
        if win:
            k, kind, c, pre, pr, lf, t0 = win
            info = json.load(open(str(pre) + ".json")); pinvs.append(info["Pinv"]); cur = Path(str(pre) + ".qasm"); excised = True
            stage("module excision", via=kind, centre=c, secs=round(time.time() - t0, 1), blocks=info.get("blocks"),
                  cz=sum(1 for t_, _, _ in parse_qasm(cur)[1] if t_ == "cz"), failed=fails)
        else:
            stage("module excision", error="no candidate decoded", failed=fails)

    # 3. SWAP network(s) of the remaining body. v2: candidates in order until one excises: the best network at
    # --min-cover, every other network on this circuit, then the best at cover 0.6 (fixed points allowed).
    n, g = parse_qasm(cur)
    bcands, seen = [], set()
    sw = find_swapnet(n, g, span_max, a.min_cover, net_win)
    if sw and "error" not in sw:
        bcands.append(("best", sw))
    for r in find_swapnets(n, g, span_max, a.min_cover, net_win):
        bcands.append(("other", r))
    sw6 = find_swapnet(n, g, span_max, 0.6, net_win)
    if sw6 and "error" not in sw6:
        bcands.append(("cover0.6", sw6))
    uniq = []
    for kind, r in bcands:
        key = (r["centre"] // 20, tuple(r["tau"]))
        if key not in seen:
            seen.add(key); uniq.append((kind, r))
    if not uniq:
        stage("swapnet", error=(sw or {}).get("error", "no CZ triples"))
    for k, (kind, sw) in enumerate(uniq):
        stage("swapnet", cand=k, via=kind, pairs=sw["pairs"], cover=round(sw["cover"], 2), span=sw["span"], centre=sw["centre"], fixed=sw["fixed_points"])
        cj2 = out / f"{stem}_swapnet{k}_census.json"
        cj2.write_text(json.dumps({"blocks": [{"centre": sw["centre"], "tau": sw["tau"], "method": f"swapnet {kind}", "span": sw["span"]}]}))
        pre = out / f"{stem}_body{k}"
        env2 = dict(env, EXC_MAX_DIST=str(max(1500, len(g))), MKRED_MAX_ERR="0.5")
        rc, dt = run([a.python, MKRED, cur, cj2, pre, f"{sw['centre']}:{len(g)}"], log, cwd=MKRED_CWD, env=env2, label=f"mkred swapnet {k}")
        q3, j3 = Path(str(pre) + ".qasm"), Path(str(pre) + ".json")
        if rc or not q3.exists():
            stage("swapnet excision", cand=k, rc=rc, error="mkred failed (see log)")
            continue
        info = json.load(open(j3)); pinvs.append(info["Pinv"]); cur = q3
        stage("swapnet excision", cand=k, rc=rc, secs=dt, blocks=info.get("blocks"), gates=len(parse_qasm(cur)[1]), cz=sum(1 for t, _, _ in parse_qasm(cur)[1] if t == "cz"))
        break

    # 4. p9solve on the remainder
    runs = out / "p9"; runs.mkdir(exist_ok=True)
    P = list(range(n))
    for pv in pinvs:
        P = [pv[P[w]] for w in range(n)]
    res["Pinv_total"] = P
    top = []
    for cr in ("0.5", "0.52", "0.48"):
        tag = f"body_c{cr.replace('.', '')}"
        rc, dt = run([P9SOLVE, "--qasm", cur, "--outdir", runs, "--tag", tag, "--expected-bitstring", "", "--center-ratio", cr,
                      "--cutoff", "3e-4", "--seed", "7", "--max-bond", "512", "--unswap-threshold", "5e5", "--abort-after-no-progress-unswap-cycles", "4",
                      "--early-stopping-gates", "100", "--no-parallel-rewire", "--samples", "1000"], log, cwd=MKRED_CWD, env=env, label=f"p9solve centre {cr}")
        S = json.loads((runs / tag / "summary.json").read_text()) if (runs / tag / "summary.json").exists() else {}
        top = S.get("top_permuted_samples") or []
        stage("p9solve", centre=cr, rc=rc, secs=dt, termination=S.get("termination_reason"), samples=bool(top))
        if top:
            break
    if a.mps_bond:      # second instrument on the same remainder (bounded: only small remainders)
        ncz = sum(1 for t, _, _ in parse_qasm(cur)[1] if t == "cz")   # the circuit p9solve read (after every excision)
        if ncz <= 400:
            t0 = time.time()
            try:
                M = mps_readout(cur, a.mps_bond)
                m_ans = "".join(M["top1"][P[w]] for w in range(n))
                res["mps"] = {"answer_q0_left": m_ans, "ratio": round(M["c1"] / max(M["c2"], 1), 2), "max_bond": M["max_bond"]}
                stage("mps readout", bond=a.mps_bond, secs=round(time.time() - t0, 1), ratio=res["mps"]["ratio"], max_bond=M["max_bond"])
            except Exception as e:  # noqa: BLE001
                stage("mps readout", error=f"{type(e).__name__}: {e}")
        else:
            stage("mps readout", error=f"skipped: remainder has {ncz} CZ > 400")
    if not top:
        if res.get("mps", {}).get("ratio", 0) >= 2:     # p9solve gave nothing: the MPS readout is the answer
            ans = res["mps"]["answer_q0_left"]; res.update(answer_q0_left=ans, ratio=res["mps"]["ratio"], answer_from="mps")
            (out / "answer.txt").write_text(ans + "\n")
        else:
            stage("p9solve", error="no samples on any centre"); (out / "result.json").write_text(json.dumps(res, indent=1)); return 3
    else:
        bits, c1 = top[0][0], top[0][1]; c2 = top[1][1] if len(top) > 1 else 0
        ans = "".join(bits[P[w]] for w in range(n))
        res.update(answer_q0_left=ans, top1_count=c1, top2_count=c2, ratio=round(c1 / max(c2, 1), 2), termination=S.get("termination_reason"), answer_from="p9solve")
        if "mps" in res:
            res["mps_agrees"] = res["mps"]["answer_q0_left"] == ans
        stage("answer", ratio=res["ratio"], mps_agrees=res.get("mps_agrees"))
    (out / "answer.txt").write_text(ans + "\n")
    if a.truth_dir:   # grade only, after the answer is written
        mp = Path(a.truth_dir) / f"{stem}_meta.json"
        if mp.exists():
            truth = json.loads(mp.read_text()).get("peaked_state", "")
            res["grade"] = "MATCH" if hashlib.sha256(ans.encode()).hexdigest() == hashlib.sha256(truth.encode()).hexdigest() else "MISMATCH"
            stage("grade", result=res["grade"])
    res["total_secs"] = round(time.time() - T0, 1); (out / "result.json").write_text(json.dumps(res, indent=1))
    print(f"ANSWER (q0_left) {ans}  ratio {res['ratio']}  total {res['total_secs']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
