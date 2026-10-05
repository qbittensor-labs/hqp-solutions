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

"""Operator-level readout of a (reduced) peaked circuit: reference absorber + robust extraction.

Refactor of scratchpad hybrid/hybrid.py (validated: d2_s1 Hamming 0, p=0.2146, margin 7.56) into a
function, so the excision stage can call it on its reduced circuit instead of a forward-MPS ladder.
WHY: the base R>P of these circuits is itself an operator-level mirror (d2_s1's absorber swallowed
847/924 unitaries, block AND base, at bond 2-16), so cancelling it as an OPERATOR reads the peak at
p~0.1-0.2 with a margin of several, where a forward MPS of the same shape gives margin ~1.2.
All three readout fixes are kept: centre at the last site, norm taken from the last tensor, and the
carry's scale peeled into a log accumulator (complex64 underflow otherwise ties every weight at 0).
"""
import os
import time

import numpy as np
import torch

import fp32_patch  # noqa: F401
import mps_torch
from unswap_ref import mpo_compress_unswap, mpo_log10_frob2_ratio
from utils import iter_layers, merge_layers

ZIPUP_CHUNK = int(os.environ.get("HQP_ZIPUP_CHUNK", "512"))
ZIPUP_GRAM_GB = float(os.environ.get("HQP_ZIPUP_GRAM_GB", "8"))
_G4_PERM = (1, 0, 3, 2)      # qiskit little-endian 2q matrices -> (q0,q1,q0',q1'); verified by self-test


def _log_default(m):
    print(m, flush=True)


def log(m):                   # rebound per call in solve(); _zipup_gram uses it for its fallback notice
    _log_default(m)


def _zipup_gram(L, Wi, Ai, chi, t_, o_, wr_, Dp_):
    """Left singular vectors + carry for one zipup site, WITHOUT ever materialising M.

    G = M M^H is [t*o, t*o] however wide M is, and a Gram accumulates over any partition of M's
    columns, so theta is rebuilt one Dp-slice at a time. The carry is then formed as L = U_k^H M by
    direct projection -- never s^-1 * something -- so no division by a near-zero singular value
    happens anywhere.

    The catch, and it bit on the very first control: a Gram SQUARES the condition number, and these
    matrices span ~17 orders of magnitude, so cuSOLVER's eigh returned "failed to converge
    (1017)" even on a small 4096x4096 case. Three guards, in order: accumulate in complex128,
    normalise G to unit diagonal scale, and add a tiny uniform ridge (which shifts every eigenvalue
    equally and leaves the eigenVECTORS exactly unchanged). If eigh still refuses, fall back to
    svd(G) -- gesvd is markedly more robust on a Hermitian PSD matrix than eigh.
    """
    rows = t_ * o_
    dev, dt = L.device, L.dtype
    acc = torch.complex128
    G = torch.zeros((rows, rows), dtype=acc, device=dev)
    for p0 in range(0, Dp_, ZIPUP_CHUNK):
        p1 = min(p0 + ZIPUP_CHUNK, Dp_)
        th = torch.einsum("twD,wroi,DiP->torP", L, Wi, Ai[:, :, p0:p1])
        Mc = th.reshape(rows, wr_ * (p1 - p0)).to(acc)
        G += Mc @ Mc.conj().T
        del th, Mc
    G = 0.5 * (G + G.conj().T)                             # kill asymmetry from accumulation order
    gscale = float(G.diagonal().real.max())
    if not (gscale > 0.0 and np.isfinite(gscale)):
        raise RuntimeError(f"zipup Gram has non-positive scale {gscale!r}")
    G /= gscale
    G += torch.eye(rows, dtype=acc, device=dev) * 1e-14
    try:
        evals, evecs = torch.linalg.eigh(G)                # ascending
        evals = torch.flip(evals, (0,))
        evecs = torch.flip(evecs, (1,))
    except Exception as e:
        log(f"    [zipup] eigh failed ({type(e).__name__}) -> svd(G) fallback")
        evecs, evals, _ = torch.linalg.svd(G)              # already descending
    del G
    evals = (evals.real - 1e-14).clamp_min(0)
    sv = torch.sqrt(evals)
    smax = float(sv[0]) if sv.numel() else 0.0
    k = min(chi, int((sv > 1e-12 * smax).sum()) if smax > 0 else 1) or 1
    Uk = evecs[:, :k].contiguous().to(dt)
    del evecs, evals, sv
    Lnew = torch.empty((k, wr_, Dp_), dtype=dt, device=dev)
    UkH = Uk.conj().T.contiguous()
    for p0 in range(0, Dp_, ZIPUP_CHUNK):
        p1 = min(p0 + ZIPUP_CHUNK, Dp_)
        th = torch.einsum("twD,wroi,DiP->torP", L, Wi, Ai[:, :, p0:p1])
        Mc = th.reshape(rows, wr_ * (p1 - p0))
        Lnew[:, :, p0:p1] = (UkH @ Mc).reshape(k, wr_, p1 - p0)
        del th, Mc
    return Uk.reshape(t_, o_, k), Lnew




def _apply_gates(mps, circ, chi, swap4, cutoff=1e-10):
    idx = {qb: i for i, qb in enumerate(circ.qubits)}
    for inst in circ.data:
        op = inst.operation
        if op.name in ("barrier", "measure"):
            continue
        qs = [idx[q] for q in inst.qubits]
        M = torch.tensor(np.asarray(op.to_matrix()), dtype=torch.complex64, device="cuda")
        if len(qs) == 1:
            mps.apply_1q(qs[0], M)
        else:
            mps.apply_2q(qs[0], qs[1], M.reshape(2, 2, 2, 2).permute(*_G4_PERM).contiguous(), chi, cutoff, swap4)


def _zipup(W, A, chi, n, state, logscale):
    L = torch.ones((1, 1, 1), dtype=torch.complex64, device="cuda")
    B = []
    for i in range(n):
        Wi, Ai = W[i], A[i]
        if i == 0:
            Wi = Wi.unsqueeze(0)
        elif i == n - 1:
            Wi = Wi.unsqueeze(1)
        t_, o_, wr_, Dp_ = L.shape[0], Wi.shape[2], Wi.shape[1], Ai.shape[2]
        rows, cols = t_ * o_, wr_ * Dp_
        A[i] = None
        if rows * cols * 8 / 2 ** 30 <= ZIPUP_GRAM_GB:
            theta = torch.einsum("twD,wroi,DiP->torP", L, Wi, Ai)
            M = theta.reshape(rows, cols)
            U, s, Vh = torch.linalg.svd(M, full_matrices=False)
            if not torch.isfinite(s).all():
                s = torch.nan_to_num(s)
            smax = float(s[0]) if s.numel() else 0.0
            k = min(chi, int((s > 1e-12 * smax).sum()) if smax > 0 else 1) or 1
            Bi = U[:, :k].reshape(t_, o_, k)
            L = (s[:k].to(Vh.dtype).unsqueeze(1) * Vh[:k]).reshape(k, wr_, Dp_)
            del M, U, s, Vh, theta
        else:
            Bi, L = _zipup_gram(L, Wi, Ai, chi, t_, o_, wr_, Dp_)
        B.append(Bi)
        sc = float(L.abs().max())
        if sc > 0.0 and np.isfinite(sc):
            L = L / sc
            logscale[0] += float(np.log(sc))
    B[-1] = torch.einsum("tok,kwD->toD", B[-1], L).reshape(B[-1].shape[0], 2, 1)
    m = mps_torch.MPS(n)
    m.A = B
    m.pos = list(state.pos)
    m.qubit_at = list(state.qubit_at)
    m.center = n - 1                        # every B[i<n-1] is left-canonical
    nrm = float(torch.sqrt((B[-1].conj() * B[-1]).sum().real))
    if nrm > 0 and np.isfinite(nrm):
        B[-1] = B[-1] / nrm
    return m


def consolidate_sections(qc, junctions):
    """Consolidate each section of qc separately and concatenate, so no 2q block straddles a junction
    and a junction is an exact instruction index in the result.

    junctions: gate counts (in qc.data order) where a new section starts. Returns (circuit, [index]).
    A global Collect2qBlocks pass would re-emit the circuit in an arbitrary topological order and the
    junction would stop being an index at all."""
    from qiskit import QuantumCircuit
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
    from qiskit.transpiler import PassManager
    pm = lambda: PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)])   # noqa: E731
    data = [inst for inst in qc.data if inst.operation.name not in ("barrier", "measure")]
    bounds = [0] + [int(j) for j in junctions] + [len(data)]
    out = QuantumCircuit(qc.num_qubits)
    idx = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sec = QuantumCircuit(qc.num_qubits)
        for inst in data[a:b]:
            sec.append(inst.operation, [sec.qubits[qc.find_bit(q).index] for q in inst.qubits])
        sec = pm().run(sec) if b > a else sec
        out.compose(sec, inplace=True)
        idx.append(len(out.data))
    return out, idx[:-1]


def solve(qc, seed=123, cutoff=0.002, max_bond=2048, final_maxbond=4096, early_stop=100,
          center_ratio=0.5, deadline=None, beam=512, topk=8, logger=None, junctions=None, centre_junction=0):
    """qc: QuantumCircuit of 1q/2q gates (no measures). Returns dict with 'top' = [(bits, w)] in qc's frame.
    junctions + centre_junction: start absorbing exactly at that section boundary (see consolidate_sections)."""
    global log
    log = logger or _log_default
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
    from qiskit.transpiler import PassManager
    t0 = time.time()
    n = qc.num_qubits
    if junctions:
        cq, jidx = consolidate_sections(qc, junctions)
        center_ratio = int(jidx[centre_junction])
    else:
        cq = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(qc)
    T = cq.count_ops().get("unitary", 0)
    log(f"  [op-readout] {n}q, {T} two-qubit blocks, centre {center_ratio}, early_stop {early_stop}")
    tb = lambda x: torch.tensor(x, dtype=torch.complex64, device="cuda")     # noqa: E731
    mpo, ll, lr, stats = mpo_compress_unswap(
        cq, seed=seed, to_backend=tb, cutoff=cutoff, max_bond=max_bond, unswap_threshold=1e6,
        center_ratio=center_ratio, equal=False, flip_freq=None, max_its=20,
        early_stopping_gates=early_stop, hows=("both", "left", "right"), deadline=deadline,
        adapt_stop=(os.environ.get("HQP_ADAPT_STOP", "1") == "1"))
    fin = [s for s in stats if s.get("stage") == "final"]
    absorbed = fin[-1]["u_consumed_final"] if fin else max(
        (s.get("u_consumed_total", 0) for s in stats if s.get("stage") == "absorbing"), default=0)
    rolled_back = bool(fin and fin[-1].get("rolled_back"))
    core_bond = max(mpo.bond_size(i, i + 1) for i in range(n - 1))
    norm = mpo_log10_frob2_ratio(mpo)
    t_abs = time.time() - t0
    log(f"  [op-readout] absorbed {absorbed}/{T}{' (rolled back to the last good operator)' if rolled_back else ''} in {t_abs:.0f}s, core bond {core_bond}, log10 norm {norm:.2f}, "
        f"leftover layers {len(ll)}/{len(lr)}")
    swap4 = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                         dtype=torch.complex64, device="cuda").reshape(2, 2, 2, 2)
    state = mps_torch.MPS(n)
    gate_l = [l for l in ll if not ({"measure", "barrier"} & set(dict(l.count_ops())))]
    for lay in (list(iter_layers(merge_layers(gate_l).inverse())) if gate_l else []):
        _apply_gates(state, lay, final_maxbond, swap4)
    arrs = []
    for i in range(n):
        t = mpo[mpo.site_tag(i)]
        if isinstance(t, tuple):
            t = t[0]
        inds = list(t.inds)
        ki, bi = f"k{i}", f"b{i}"
        bonds = [ix for ix in inds if ix not in (ki, bi)]
        if i == 0:
            order = bonds + [ki, bi]
        else:
            prev = arrs[i - 1][1]
            order = [b for b in bonds if b == prev] + [b for b in bonds if b != prev] + [ki, bi]
        arr = t.transpose(*order, inplace=False).data
        if not torch.is_tensor(arr):
            arr = torch.tensor(np.asarray(arr), dtype=torch.complex64, device="cuda")
        arrs.append((arr, order[-3] if i < n - 1 else None))
    logscale = [0.0]
    torch.cuda.empty_cache()
    out = _zipup([a for a, _ in arrs], state.A, final_maxbond, n, state, logscale)
    final_meas = []
    for lay in lr:
        ops = dict(lay.count_ops())
        if "measure" in ops or "barrier" in ops:
            final_meas.append(lay)
        else:
            _apply_gates(out, lay, final_maxbond, swap4)
    perm = [g.qubits[0]._index for g in final_meas[-1]] if final_meas else list(range(n))
    cands = mps_torch.topk(out, beam=beam, k=topk)
    top = [("".join(bs[j] for j in perm) if len(perm) == n else bs, float(w)) for bs, w in cands]
    w0 = top[0][1]
    w1 = top[1][1] if len(top) > 1 else 0.0
    res = {"top": top, "w0": w0, "margin": w0 / max(w1, 1e-300), "absorbed": absorbed, "total": T,
           "core_bond": core_bond, "log10_norm": norm, "rolled_back": rolled_back, "secs": time.time() - t0, "abs_secs": t_abs,
           "degenerate": not (w0 > 0.0), "state_bond": out.max_bond(), "perm": perm,
           "_state": out}
    log(f"  [op-readout] w0={w0:.4e} margin={res['margin']:.2f} state bond {res['state_bond']} "
        f"({res['secs']:.0f}s total){'  DEGENERATE' if res['degenerate'] else ''}")
    return res


def probe(qc, n_absorb=100, seconds=300, seed=123, cutoff=0.002, max_bond=2048, center_ratio=0.5,
          junctions=None, centre_junction=0, logger=None):
    """Short absorption from one candidate centre; returns how well the circuit cancels from there.

    MEASURED on reduced d3_s1: started at the list midpoint the operator had lost 10^-1.64 of its norm
    by 100 absorbed unitaries; started at the junction where block 1 was excised, only 10^-0.26 -- and
    the final gap was 10^-8 against a healthy collapse. The mirror only cancels once the absorbed slice
    is symmetric about the TRUE centre, so every gate of start error must be held un-cancelled (and
    truncated) first. The norm after ~100 unitaries is therefore a sharp score for a candidate centre.
    """
    global log
    log = logger or _log_default
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
    from qiskit.transpiler import PassManager
    t0 = time.time()
    n = qc.num_qubits
    if junctions:
        cq, jidx = consolidate_sections(qc, junctions)
        center_ratio = int(jidx[centre_junction])
    else:
        cq = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(qc)
    T = cq.count_ops().get("unitary", 0)
    tb = lambda x: torch.tensor(x, dtype=torch.complex64, device="cuda")     # noqa: E731
    keep = os.environ.get("HQP_NORM_LOG")
    os.environ["HQP_NORM_LOG"] = "0"
    try:
        mpo, ll, lr, stats = mpo_compress_unswap(
            cq, seed=seed, to_backend=tb, cutoff=cutoff, max_bond=max_bond, unswap_threshold=1e6,
            center_ratio=center_ratio, equal=False, flip_freq=None, max_its=20,
            early_stopping_gates=max(T - n_absorb, 0), hows=("both", "left", "right"),
            deadline=time.time() + seconds, adapt_stop=False)
    finally:
        if keep is None:
            os.environ.pop("HQP_NORM_LOG", None)
        else:
            os.environ["HQP_NORM_LOG"] = keep
    absorbed = max((s.get("u_consumed_total", 0) for s in stats if s.get("stage") == "absorbing"), default=0)
    norm = mpo_log10_frob2_ratio(mpo)
    bond = max(mpo.bond_size(i, i + 1) for i in range(n - 1))
    del mpo
    torch.cuda.empty_cache()
    # loss per absorbed unitary, so a probe cut short by its deadline is still comparable
    rate = (-norm) / max(absorbed, 1)
    return {"absorbed": absorbed, "total": T, "log10_norm": norm, "bond": bond, "secs": time.time() - t0,
            "loss_per_unitary": rate, "centre_index": center_ratio}
