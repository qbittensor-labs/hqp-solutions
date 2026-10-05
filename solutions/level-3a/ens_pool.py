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

"""Multi-process ensemble of cheap MPS simulations, with pooled candidates and a z-scored verdict.

WHY. MEASURED 2026-09-18 on clean d1_s2 (48q, the shape every reduced d3 circuit has): one bond-128
simulation's argmax is right only ~2 times in 3 (margin ~1.2, wrong by 2 bits otherwise), yet the true
peak is in EVERY member's top-64 and leads the pooled mean-log-probability from the third member on.
So one simulation is a biased coin, not an oracle, and the fix is many independent coins plus an
error bar -- which is also what "more seeds must agree, with a higher margin" means quantitatively.

A member costs ~60 s at bond 128 and leaves the GPU mostly idle (launch-bound), so members are built
by several worker PROCESSES at once. Workers keep their MPSs on the GPU; the master pools the beam
tops of all members (in the ORIGINAL wire frame), sends the pooled list back, and every member scores
every candidate. Views (different cuts / D seeds of the same circuit) have different reduced frames,
so each worker maps a candidate into its view's frame through that view's Pinv.
"""
import math
import os
import queue
import time
import traceback

import numpy as np


def _worker(wid, task_q, res_q, views, threads):
    try:
        # SILENCE the worker. The validator splits the container's stdout on the FIRST payload separator, and the
        # wall watchdog may emit the payload while members are still being built: one stray "[torch] done ..." line
        # from a worker inside the base64 region = a corrupt answer. Results travel through the queues only.
        if os.environ.get("ENS_WORKER_QUIET", "1") == "1":
            try:
                dn = os.open(os.devnull, os.O_WRONLY)
                os.dup2(dn, 1); os.dup2(dn, 2)
            except OSError:
                pass
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        import torch
        torch.set_num_threads(threads)
        import mps_torch
        from qiskit import qasm2
        from qiskit.circuit.library import UGate
        from qiskit.converters import circuit_to_dag, dag_to_circuit
        custom = [qasm2.CustomInstruction('u', 3, 1, lambda t, p, l: UGate(t, p, l), builtin=True)]
        circs = {}
        members = {}

        def circ_for(vi, order):
            key = (vi, order)
            if key not in circs:
                qc = qasm2.loads(views[vi]["qasm"], custom_instructions=custom)
                qc.remove_final_measurements(inplace=True)
                circs[key] = dag_to_circuit(circuit_to_dag(qc)) if order == "dag" else qc
            return circs[key]

        while True:
            msg = task_q.get()
            if msg[0] == "stop":
                break
            if msg[0] == "build":
                _, mid, vi, chi, order, seed, poolk = msg
                t0 = time.time()
                try:
                    qc = circ_for(vi, order)
                    n = qc.num_qubits
                    # seed: None -> identity chain; int -> random order; ("opt", k) -> optimised order k
                    if seed is None:
                        perm = None
                    elif isinstance(seed, (tuple, list)) and seed[0] == "opt":
                        import chain_order
                        perm, _c = chain_order.good_order(qc, seed=int(seed[1]))
                    else:
                        perm = [int(x) for x in np.random.default_rng(int(seed)).permutation(n)]
                    mps, _ = mps_torch.evolve(qc, chi, perm=perm, log_every=0)
                    top = mps_torch.topk(mps, beam=max(256, poolk), k=poolk)
                    if not top or not all(math.isfinite(float(w_)) for _b, w_ in top[:4]) or not math.isfinite(float(mps.logF)):
                        raise FloatingPointError("member state is not finite")       # never pool a NaN member
                    Pinv = views[vi]["Pinv"]
                    top_orig = [("".join(b[Pinv[w]] for w in range(n)), float(w_)) for b, w_ in top]
                    members[mid] = (vi, mps)
                    res_q.put(("built", wid, mid, vi, chi, order, seed, top_orig,
                               float(np.exp(mps.logF)), time.time() - t0, None))
                except Exception as e:                            # noqa: BLE001
                    torch.cuda.empty_cache()
                    res_q.put(("built", wid, mid, vi, chi, order, seed, [], 0.0, time.time() - t0,
                               f"{type(e).__name__}: {str(e)[:160]}"))
            elif msg[0] == "score":
                _, cands = msg
                out = {}
                for mid, (vi, mps) in members.items():
                    Pinv = views[vi]["Pinv"]
                    n = len(Pinv)
                    col = np.empty(len(cands))
                    for ci, c in enumerate(cands):
                        red = ["0"] * n
                        for w, b in enumerate(c):
                            red[Pinv[w]] = b
                        col[ci] = math.log(max(abs(mps.amp("".join(red))) ** 2, 1e-300))
                    out[mid] = col
                res_q.put(("scores", wid, out))
            elif msg[0] == "drop":
                members.clear()
                torch.cuda.empty_cache()
                res_q.put(("dropped", wid))
    except Exception:                                             # noqa: BLE001
        res_q.put(("fatal", wid, traceback.format_exc()[-600:]))


class EnsemblePool:
    def __init__(self, views, n_workers=4, threads=3, log=print):
        import torch.multiprocessing as mp
        self.ctx = mp.get_context("spawn")
        self.views = views
        self.log = log
        self.res_q = self.ctx.Queue()
        self.task_qs, self.procs = [], []
        for w in range(n_workers):
            tq = self.ctx.Queue()
            p = self.ctx.Process(target=_worker, args=(w, tq, self.res_q, views, threads), daemon=True)
            p.start()
            self.task_qs.append(tq)
            self.procs.append(p)
        self.members = {}            # mid -> dict(meta)
        self.pool = {}               # candidate (original frame) -> times proposed
        self._next = 0

    def build(self, specs, poolk=64, deadline=None):
        """specs: list of (view_idx, chi, gate_order, chain_seed|None). Blocks until all are done."""
        pending = 0
        for i, (vi, chi, order, seed) in enumerate(specs):
            mid = self._next
            self._next += 1
            self.task_qs[i % len(self.task_qs)].put(("build", mid, vi, chi, order, seed, poolk))
            pending += 1
        while pending:
            timeout = None if deadline is None else max(1.0, deadline - time.time())
            try:
                r = self.res_q.get(timeout=timeout)
            except queue.Empty:
                self.log(f"  [ens] build deadline hit with {pending} members outstanding")
                break
            if r[0] == "fatal":
                self.log(f"  [ens] worker {r[1]} died: {r[2]}")
                pending -= 1
                continue
            if r[0] != "built":
                continue
            _, wid, mid, vi, chi, order, seed, top, F, secs, err = r
            pending -= 1
            if err:
                self.log(f"  [ens] member {mid} (view {vi} chi {chi} {order} seed {seed}) FAILED: {err}")
                continue
            self.members[mid] = {"view": vi, "chi": chi, "order": order, "seed": seed, "F": F,
                                 "secs": secs, "top": top[0][0], "margin": top[0][1] / max(top[1][1], 1e-300)}
            for bits, _w in top:
                self.pool[bits] = self.pool.get(bits, 0) + 1
            self.log(f"  [ens] member {mid}: view {vi} chi {chi} {order} seed {seed} {secs:.0f}s "
                     f"margin {self.members[mid]['margin']:.2f} F {F:.1e} pool {len(self.pool)}")

    def score(self, cands=None):
        cands = list(self.pool) if cands is None else list(cands)
        for tq in self.task_qs:
            tq.put(("score", cands))
        cols, got = {}, 0
        while got < len(self.task_qs):
            r = self.res_q.get()
            if r[0] == "scores":
                cols.update(r[2])
                got += 1
            elif r[0] == "fatal":
                self.log(f"  [ens] worker {r[1]} died while scoring: {r[2]}")
                got += 1
        mids = sorted(m for m in cols if np.isfinite(cols[m]).all())      # a non-finite column would poison every mean
        if len(mids) < len(cols):
            self.log(f"  [ens] dropped {len(cols) - len(mids)} member(s) with non-finite scores")
        L = np.stack([cols[m] for m in mids], axis=1) if mids else np.zeros((len(cands), 0))
        return cands, mids, L

    def close(self):
        for tq in self.task_qs:
            tq.put(("stop",))
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


def verdict(cands, L, top_n=24):
    """Leader by mean log-probability; z against its MOST DANGEROUS rival (min z over the top_n)."""
    if L.shape[1] == 0 or not cands:
        return None
    mean = L.mean(axis=1)
    order = np.argsort(-mean)
    b = order[0]
    worst = None
    for s in order[1:top_n + 1]:
        d = L[b] - L[s]
        sd = d.std(ddof=1) if len(d) > 1 else float("nan")
        z = d.mean() / (sd / math.sqrt(len(d))) if sd and sd > 0 else float("inf")
        rec = {"rival": cands[s], "z": float(z), "geo_margin": float(math.exp(d.mean())),
               "wins": float((d > 0).mean())}
        if worst is None or rec["z"] < worst["z"]:
            worst = rec
    return {"leader": cands[b], "members": int(L.shape[1]), **(worst or {})}
