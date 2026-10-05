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

"""NaN-safe replacement extraction for the unswap pipeline.

Absorption (mpo_compress_unswap, plain gesvd) is proven sound; quimb's
mpo_to_mps extraction intermittently poisons the state with NaN. This driver
reuses the absorption and re-implements extraction with the validated torch
engine (mps_torch): gate application for leftover layers, an explicit
MPO x MPS contraction + 2-pass canonical compression, finite-checks at every
step, and the validated beam top-k.

Env: QASM (or /challenge_input), SEED (42), MAXBOND (2560), CUTOFF (0.002),
     FINAL_MAXBOND (2048), TRUTH (opt, scoring), DEADLINE_S (absorb budget).
"""
import os
import pickle
import sys
import time

# /app holds the sources baked into the image at BUILD time. Prepending it silently shadows the
# working copy: MEASURED 2026-09-17, every extract2 run had been importing the 2026-09-14 /app code,
# so HQP_UNSWAP_ALGO=greedy, HQP_CUTOFF_MODE, HQP_LOCAL_SWAPS and HQP_NORM_TRACK were all no-ops
# while appearing to work. APPEND it instead, so it stays a fallback for running outside the source
# dir but never outranks the files next to this script -- and say out loud where each module came
# from, because that is the only way this fails loudly next time.
sys.path.append("/app")
import numpy as np
import torch

import mps_torch
from unswap import mpo_compress_unswap
from utils import iter_layers, merge_layers
import unswap as _unswap_mod
import circuit_mpo as _cmpo_mod
_PROV = (f"unswap={_unswap_mod.__file__} circuit_mpo={_cmpo_mod.__file__} "
         f"greedy={hasattr(_unswap_mod, 'unswap_greedy')} "
         f"cutoff_mode={getattr(_cmpo_mod, '_CUTOFF_MODE', 'ABSENT')} "
         f"local_swaps={getattr(_cmpo_mod, '_LOCAL_SWAPS', 'ABSENT')}")

TRUTH = os.environ.get("TRUTH", "")
SEED = int(os.environ.get("SEED", "42"))
MAXBOND = int(os.environ.get("MAXBOND", "2560"))
CUTOFF = float(os.environ.get("CUTOFF", "0.002"))
FINAL_MAXBOND = int(os.environ.get("FINAL_MAXBOND", "2048"))
DEADLINE_S = float(os.environ.get("DEADLINE_S", "9000"))
# Gates left when absorption gives up. MEASURED 2026-09-16: at 100 this stopped d3_s1 at
# t_u 1328/1415, handing 119 residual layers to the extraction -- they saturated the MPS at
# bond 2048 immediately and the peak died (w0 6e-8, truth 100x BELOW uniform). Full absorption
# (greedy, 1415/1415) leaves only the 2+2 measure frames and a bond-224 core, so 0 is the
# default here; raise it only to dodge a tail livelock.
EARLY_STOP_GATES = int(os.environ.get("EARLY_STOP_GATES", "0"))
# Absorption costs ~44 min on d3 while the extraction costs ~1.5 s, so a failed extraction used to
# burn a whole run. Point CORE_CACHE at a file to save the absorbed core (+ leftover layers) and
# reuse it for every later extraction experiment. Empty = disabled (default).
CORE_CACHE = os.environ.get("CORE_CACHE", "")
# --- zipup VRAM controls ---------------------------------------------------------------------
# The zipup materialises M = theta.reshape(t*o, wr*Dp) and SVDs it. At chi=8192 with a bond-44 core
# that is [16384, 360448] complex64 = 47 GB in ONE tensor, which is what OOM'd at chi=4096 and 8192
# (the front layers themselves completed fine, reaching F_est 0.933). The left singular vectors and
# the truncation threshold can instead come from the Gram matrix G = M M^H, which is only
# [t*o, t*o] = 2.1 GB at chi=8192, accumulated over column chunks so M never exists whole.
ZIPUP_CHUNK = int(os.environ.get("HQP_ZIPUP_CHUNK", "512"))      # columns of Dp per chunk
ZIPUP_GRAM_GB = float(os.environ.get("HQP_ZIPUP_GRAM_GB", "8"))  # use the Gram path above this size
ZIPUP_OFFLOAD = os.environ.get("HQP_ZIPUP_OFFLOAD", "0") == "1"  # park finished site tensors on CPU
# The reference implementation runs in torch.complex128 throughout (its to_backend_cuda). We use
# complex64, where |amp|^2 for amp ~ 3.6e-35 underflows to EXACTLY 0 -- which is how a fully
# absorbed d3 run reported w0=0. The core MPO is bond ~50-116 by this point, so double precision
# for the READOUT costs almost nothing. Absorption keeps its own dtype.
EXTRACT_C128 = os.environ.get("HQP_EXTRACT_DTYPE", "c128").lower() in ("c128", "complex128")

t0 = time.time()


def log(m):
    print(f"[extract2 +{time.time()-t0:7.1f}s] {m}", flush=True)


# ---------- load circuit (mirror the pipeline prep exactly) ----------
import fp32_patch  # noqa: F401  (quimb sgn + torch SVD/QR fp32 robustness)
from qiskit import qasm2
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.transpiler import PassManager

QASM = os.environ.get("QASM", "/challenge_input/circuit.qasm")
qc = qasm2.load(QASM, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
qc.remove_final_measurements(inplace=True)
n = qc.num_qubits
qc = PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(qc)
log("provenance: " + _PROV)
log(f"circuit {n}q {qc.size()} gates ({qc.count_ops().get('unitary',0)} unitary blocks) | seed={SEED} maxbond={MAXBOND}")


def to_backend(x):
    return torch.tensor(x, dtype=torch.complex64, device="cuda")


# ---------- self-test: resolve 2q gate index convention vs Statevector ----------
def _gate_convention():
    """Return permute tuple for G4 so my engine matches qiskit exactly."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Statevector, random_unitary
    import random as _r
    _r.seed(3)
    tq = QuantumCircuit(3)
    for q in range(3):
        tq.u(_r.uniform(0.3, 2.8), _r.uniform(0, 6), _r.uniform(0, 6), q)
    tq.append(UnitaryGate(random_unitary(4, seed=11)), [0, 2])
    tq.append(UnitaryGate(random_unitary(4, seed=12)), [2, 1])
    sv = Statevector(tq).data
    best = None
    for perm in [None, (1, 0, 3, 2)]:
        mps = mps_torch.MPS(3)
        SWAP = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=torch.complex64,
                            device="cuda").reshape(2, 2, 2, 2)
        idx = {qb: i for i, qb in enumerate(tq.qubits)}
        ok = True
        for inst in tq.data:
            op = inst.operation
            qs = [idx[q] for q in inst.qubits]
            M = torch.tensor(np.asarray(op.to_matrix()), dtype=torch.complex64,
                             device="cuda")
            if len(qs) == 1:
                mps.apply_1q(qs[0], M)
            else:
                G4 = M.reshape(2, 2, 2, 2)
                if perm is not None:
                    G4 = G4.permute(*perm).contiguous()
                mps.apply_2q(qs[0], qs[1], G4, 64, 1e-12, SWAP)
        err = 0.0
        for j in (0, 3, 5, 7):
            bits = "".join(str((j >> q) & 1) for q in range(3))
            err = max(err, abs(mps.amp(bits) - sv[j]))
        if best is None or err < best[0]:
            best = (err, perm)
    log(f"gate convention: perm={best[1]} err={best[0]:.2e}")
    assert best[0] < 1e-5, f"no convention matched (err {best[0]})"
    return best[1]


G4_PERM = _gate_convention()
SWAP4 = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                     dtype=torch.complex64, device="cuda").reshape(2, 2, 2, 2)


def apply_circuit_gates(mps, circ, chi, cutoff=1e-10):
    idx = {qb: i for i, qb in enumerate(circ.qubits)}
    for inst in circ.data:
        op = inst.operation
        if op.name in ("barrier", "measure"):
            continue
        qs = [idx[q] for q in inst.qubits]
        M = torch.tensor(np.asarray(op.to_matrix()), dtype=torch.complex64,
                         device="cuda")
        if len(qs) == 1:
            mps.apply_1q(qs[0], M)
        elif len(qs) == 2:
            G4 = M.reshape(2, 2, 2, 2)
            if G4_PERM is not None:
                G4 = G4.permute(*G4_PERM).contiguous()
            mps.apply_2q(qs[0], qs[1], G4, chi, cutoff, SWAP4)
        else:
            raise ValueError("3q gate")
    return mps


# ---------- absorption (proven path) ----------
def _absorb():
    log("absorption starting (plain gesvd unswap)")
    return mpo_compress_unswap(
        qc, seed=SEED, to_backend=to_backend, cutoff=CUTOFF, max_bond=MAXBOND,
        unswap_threshold=1e6, center_ratio=0.5, equal=False, flip_freq=None,
        max_its=20, early_stopping_gates=EARLY_STOP_GATES, hows=("both", "left", "right"),
        deadline=time.time() + DEADLINE_S, allow_abandon=False)


mpo = ll = lr = None
if CORE_CACHE and os.path.exists(CORE_CACHE):
    # A cache built from different absorption settings would silently answer a different question,
    # so the key travels with the payload and a mismatch re-absorbs rather than being reused.
    try:
        with open(CORE_CACHE, "rb") as f:
            blob = pickle.load(f)
        key = (QASM, SEED, CUTOFF, MAXBOND, EARLY_STOP_GATES)
        if tuple(blob["key"]) == key:
            mpo, ll, lr = blob["mpo"], blob["ll"], blob["lr"]
            for t in mpo.tensors:
                t.modify(data=to_backend(t.data))
            log(f"absorption LOADED from {CORE_CACHE} (absorbed in {blob.get('secs', -1):.0f}s)")
        else:
            log(f"cache key mismatch {tuple(blob['key'])} != {key} -- re-absorbing")
    except Exception as e:
        log(f"cache unreadable ({type(e).__name__}: {e}) -- re-absorbing")

if mpo is None:
    _t_abs = time.time()
    mpo, ll, lr, _stats = _absorb()
    if CORE_CACHE:
        try:
            cpu = {i: np.asarray(t.data.cpu()) for i, t in enumerate(mpo.tensors)}
            for i, t in enumerate(mpo.tensors):
                t.modify(data=cpu[i])
            tmp = CORE_CACHE + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"key": (QASM, SEED, CUTOFF, MAXBOND, EARLY_STOP_GATES),
                             "mpo": mpo, "ll": ll, "lr": lr,
                             "secs": time.time() - _t_abs}, f, protocol=4)
            os.replace(tmp, CORE_CACHE)          # atomic: never leave a half-written cache
            log(f"absorption cached -> {CORE_CACHE}")
        except Exception as e:
            log(f"cache write failed ({type(e).__name__}: {e}) -- continuing")
        finally:
            for t in mpo.tensors:
                t.modify(data=to_backend(t.data))

core_bond = max(mpo.bond_size(i, i + 1) for i in range(n - 1))
log(f"absorption done: core bond={core_bond} left={len(ll)} right={len(lr)}")

# ---------- my extraction ----------
# 1) |0> ; apply inverted leftover front layers as gates
state = mps_torch.MPS(n)
front = list(iter_layers(merge_layers(ll[:-2]).inverse())) if len(ll) > 2 else []
for i, lay in enumerate(front):
    apply_circuit_gates(state, lay, FINAL_MAXBOND)
log(f"front layers applied: {len(front)} layers, bond={state.max_bond()} "
    f"F_est={np.exp(state.logF):.3e}")

# 2) core MPO x state, exact grouped-bond product then 2-pass compression
#    quimb MPO tensor for site i: fetch array transposed to (l, r, out, in)
arrs = []
for i, site in enumerate(mpo.sites):
    t = mpo[mpo.site_tag(i)]
    if isinstance(t, tuple):
        t = t[0]
    inds = list(t.inds)
    ki = f"k{i}"
    bi = f"b{i}"
    bonds = [ix for ix in inds if ix not in (ki, bi)]
    # order bonds: shared with previous site first
    if i == 0:
        order = bonds + [ki, bi]
    else:
        prev_bond = arrs[i - 1][1]
        first = [b for b in bonds if b == prev_bond]
        rest = [b for b in bonds if b != prev_bond]
        order = first + rest + [ki, bi]
    tt = t.transpose(*order, inplace=False)
    arr = tt.data
    if not torch.is_tensor(arr):
        arr = torch.tensor(np.asarray(arr), dtype=torch.complex64, device="cuda")
    arrs.append((arr, order[-3] if i < n - 1 else None))
W = [a for a, _ in arrs]

# Streaming zipup MPO*MPS: sweep L->R, truncating the LEFT bond to chi before moving
# right, so peak memory is one theta tensor [chi,2,wr,D'] (~GBs) not the full 69k*69k
# product. Carry L[t, wl, D]: t=truncated bond so far, wl=MPO left bond, D=MPS left bond.
logscale = [0.0]


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


def mpo_apply_zipup(W, A, chi):
    L = torch.ones((1, 1, 1), dtype=torch.complex64, device="cuda")  # [t, wl, D]
    B = []
    peak_rows = 0
    for i in range(n):
        Wi = W[i]                       # first:[wr,o,in]  mid:[wl,wr,o,in]  last:[wl,o,in]
        Ai = A[i]                       # [D, in, D']
        if i == 0:
            Wi = Wi.unsqueeze(0)        # -> [1(wl), wr, o, in]
        elif i == n - 1:
            Wi = Wi.unsqueeze(1)        # -> [wl, 1(wr), o, in]
        # theta[t,o,wr,Dp] = L[t,wl,D] * Wi[wl,wr,o,in] * Ai[D,in,Dp] -- shapes WITHOUT building it,
        # because whether we can afford to build it is exactly what we are about to decide.
        t_, o_, wr_, Dp_ = L.shape[0], Wi.shape[2], Wi.shape[1], Ai.shape[2]
        rows, cols = t_ * o_, wr_ * Dp_
        peak_rows = max(peak_rows, rows)
        # The front-layer MPS is up to ~13 GB at chi=4096 and every tensor is dead once contracted,
        # so drop the state's reference as we go rather than holding the whole thing beside theta.
        A[i] = None
        if rows * cols * 8 / 2 ** 30 <= ZIPUP_GRAM_GB:
            # Direct path -- the one d2_s1 validates end to end (Hamming 0, w0 0.2087).
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
        del Ai
        B.append(Bi)                                       # left-canonical site tensor
        if ZIPUP_OFFLOAD and i < n - 1:
            B[-1] = B[-1].to("cpu")
        # UNDERFLOW GUARD. The absorbed MPO is a product of ~1e5 truncated SVDs with no
        # renormalisation, so its overall scale is astronomically small (MEASURED on d3_s1:
        # amp ~ 3.6e-35, so |amp|^2 = 1.3e-69 -> underflows float32 to EXACTLY 0 and every beam
        # weight ties at zero, making the ranking arbitrary). Peel the scale off the carry at each
        # step and keep it in a log accumulator, so nothing in the sweep ever rides near 1e-38.
        sc = float(L.abs().max())
        if sc > 0.0 and np.isfinite(sc):
            L = L / sc
            logscale[0] += float(np.log(sc))
    if ZIPUP_OFFLOAD:
        for j in range(n - 1):
            B[j] = B[j].to("cuda")
    # absorb the final carry (k, 1, 1) into the last tensor
    B[-1] = torch.einsum("tok,kwD->toD", B[-1], L).reshape(B[-1].shape[0], 2, 1)
    m = mps_torch.MPS(n)
    m.A = B
    m.pos = list(state.pos)
    m.qubit_at = list(state.qubit_at)
    # Every B[i<n-1] is left-canonical (columns of U), so the orthogonality centre is the LAST
    # site, not the first. Marking it 0 made mps_torch.move_center_to a no-op and left the beam
    # searching in the wrong gauge -- MEASURED (n=20, planted p=0.1): a right-canonical state finds
    # the peak with beam 1-4 regardless of bond, while left-canonical needs beam proportional to the
    # bond (8/8/32/64 at D=8/16/32/64). At bond 2048 our beam=512 could not have found it at all.
    m.center = n - 1
    # The whole norm sits in B[-1] for the same reason -- the original code normalised B[0], whose
    # norm is ~1 by construction, i.e. it was a no-op.
    nrm = float(torch.sqrt((B[-1].conj() * B[-1]).sum().real))
    if nrm > 0 and np.isfinite(nrm):
        B[-1] = B[-1] / nrm
    else:
        log(f"WARNING: final zipup tensor has norm {nrm!r} -- state is unusable")
    return m


torch.cuda.empty_cache()                               # hand the zipup a defragmented pool
out = mpo_apply_zipup(W, state.A, FINAL_MAXBOND)
log(f"core applied (zipup): bond={out.max_bond()} "
    f"finite={all(bool(torch.isfinite(t.real).all()) for t in out.A)} "
    f"log10_scale_peeled={logscale[0] / np.log(10):.1f}")

# 3) right leftover layers (skip measures; capture perm)
final_meas = []
for lay in lr:
    ops = dict(lay.count_ops())
    if "measure" in ops or "barrier" in ops:
        final_meas.append(lay)
    else:
        apply_circuit_gates(out, lay, FINAL_MAXBOND)
perm = [g.qubits[0]._index for g in final_meas[-1]] if final_meas else list(range(n))
log(f"right layers applied: bond={out.max_bond()}")

# 4) beam
cands = mps_torch.topk(out, beam=512, k=8)
log("top-8 (site order mapped through perm):")
results = []
for i, (bs, w) in enumerate(cands):
    logical = "".join(bs[j] for j in perm) if len(perm) == n else bs
    ham = sum(x != y for x, y in zip(logical, TRUTH)) if TRUTH else -1
    results.append((logical, w, ham))
    print(f"  top{i}: w={w:.4e} ham={ham} {logical}", flush=True)
w0 = results[0][1]
w1 = results[1][1] if len(results) > 1 else 0
# FAIL LOUD. If the beam weights underflowed, every candidate ties at 0 and "top0" is whichever
# string the search happened to visit first -- an answer-shaped object with no information in it.
# MEASURED on d3_s1 before the zipup rescale: all eight tied at 0.0 and the reported bitstring was
# pure noise (Hamming 28/48) while still printing like a result. Say so instead.
if not (w0 > 0.0):
    print(f"DEGENERATE beam weights all <= 0 (w0={w0!r}) -- ranking is arbitrary, NOT an answer",
          flush=True)
print(f"ANSWER {results[0][0]}", flush=True)
print(f"w0={w0:.4e} margin={w0/max(w1,1e-300):.2f}", flush=True)
if TRUTH:
    print(f"HAMMING {results[0][2]}/{n}", flush=True)

    # ---- truth-rank diagnostic: is the peak recoverable-but-buried, or noise? ----
    # amp(assignment) with assignment[qubit]=bit; final logical[i]=bit of qubit perm[i],
    # so assignment[perm[i]] = truth[i]. (Validated: on d2 amp2_true must ~= w0=0.21.)
    assignment = ["0"] * n
    for i, q in enumerate(perm):
        assignment[q] = TRUTH[i]
    amp_true = out.amp(assignment)
    amp2_true = abs(amp_true) ** 2
    uniform = 2.0 ** -n
    # rank via a wide beam: how many distinct bitstrings outweigh the truth?
    wide = mps_torch.topk(out, beam=4096, k=4096)
    wide_log = []
    for bs, w in wide:
        lg = "".join(bs[j] for j in perm) if len(perm) == n else bs
        wide_log.append((lg, w))
    above = sum(1 for _, w in wide_log if w > amp2_true)
    tr = next((r for r, (lg, _) in enumerate(wide_log) if lg == TRUTH), None)
    print(f"AMP2_TRUE {amp2_true:.4e}", flush=True)
    print(f"  vs uniform 2^-{n}={uniform:.2e} -> {amp2_true/uniform:.2e}x uniform", flush=True)
    print(f"  vs top1 w0={w0:.4e} -> {amp2_true/max(w0,1e-300):.3e}x top1", flush=True)
    print(f"  RANK: >= {above+1} (candidates in top-4096 that outweigh truth; "
          f"truth {'FOUND at rank '+str(tr) if tr is not None else 'NOT in top-4096'})",
          flush=True)
