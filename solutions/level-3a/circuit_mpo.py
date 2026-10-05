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

import os
from quimb.tensor import tensor_network_1d_compress, MatrixProductOperator, MatrixProductState, Circuit

from qiskit_quimb import quimb_circuit
from qiskit import QuantumCircuit

# Memory-bounded MPO·MPO compression. quimb's apply(contract=True) stacks the two
# operators then contracts each site of the FULL bond1*bond2 product via cotengra,
# which can pick a pathological-memory path and OOM once the state-MPO bond grows
# (observed: 21 GB kill at bond ~256). "zipup"/"dm" instead sweep and truncate
# bond-by-bond, so peak memory is bounded by max_bond, never the full product.
# IDEA #13 faithful (gauged + variational) truncation. Greedy single-sweep "zipup" is only
# pseudo-canonical, so its local SVD can shed the globally peak-carrying (locally-small) singular
# direction -> a peaked state drifts to a wrong low-bond attractor (the d3 w1~1e-12 failure).
# "fit-zipup" = stable zipup guess + variational refine that re-allocates kept bond to globally
# relevant content. Absorption ALWAYS defaults to zipup: it is memory-bounded (peak mem ~ max_bond),
# whereas fit-zipup on EVERY absorbed gate OOM'd at 22GB (the variational fit holds a double layer),
# stalling at 11/1415. So faithful absorption is explicit opt-in only (HQP_COMPRESS_METHOD=fit-zipup).
# CUTOFF SEMANTICS -- verified in quimb's source, and a likely cause of the lost peak.
# tensor_network_1d_compress defaults to cutoff_mode="rsum2", i.e. "trim s.t.
# sum(s_trim**2) < sum(s**2) * cutoff". So our cutoff=2e-3 licenses every single truncation to
# throw away 0.2% of the SQUARED WEIGHT. The published method's "relative cutoff eps=2e-3" is the
# ordinary reading -- keep s_i > 2e-3 * s_0 -- whose discarded weight is ~1e-5, roughly 100x
# gentler at the same nominal number. Over the ~1e4-1e5 truncations one d3 absorption performs,
# exp(-2e-3 * 11500) = 1e-10, which is exactly the fidelity we measure; under "rel" the same run
# would land near exp(-0.115) = 0.89. Default is left at quimb's own so existing results
# reproduce; set HQP_CUTOFF_MODE=rel to switch.
_CUTOFF_MODE = os.environ.get("HQP_CUTOFF_MODE", "rsum2")

# REFERENCE PARITY. The published implementation (d-kremer/peaked-circuit-simulation, the code that
# produced the 56q/1917-gate result in 4059 s) has apply_mpo do exactly ONE thing:
#     mpo1.apply(mpo2, compress=compress, max_bond=max_bond, cutoff=cutoff, create_bond=True,
#                contract=contract)
# It never calls tensor_network_1d_compress, never names a method, and never names a cutoff_mode --
# so it gets quimb's DEFAULT compression, not zipup, and quimb's default cutoff semantics. Our fork
# replaced that with an explicit lazy-stack + zipup sweep to bound peak memory on a 21 GB card. On
# 96 GB that memory argument no longer binds, and zipup compresses in an only pseudo-canonical
# gauge, which is exactly where a tiny amplitude gets lost. HQP_REF_APPLY=1 restores the reference
# call verbatim so the deviation can be measured rather than argued about.
_REF_APPLY = os.environ.get("HQP_REF_APPLY", "0") == "1"
# HQP_FAITHFUL instead makes the FINAL extraction (mpo_to_mps) faithful -- the step that directly
# produces the peak amplitude -- which runs once, is memory-safe, and is the high-value #13 change.
_FAITHFUL_ITERS = int(os.environ.get("HQP_FAITHFUL_ITERS", "8"))
_COMPRESS_METHOD = os.environ.get("HQP_COMPRESS_METHOD", "zipup")

# ------------------------------------------------------------------
#  Constructors
# ------------------------------------------------------------------

def mpo_from_circuit(circ: Circuit):
    # add dummy rz to cover all sites
    for q in range(circ.N):
    #    circ.rz(0.0, q)
        circ.u3(0, 0, 0, q)
    tn_uni = circ.get_uni()

    # contract gates per site tag
    for st in list(tn_uni.site_tags):
        tn_uni ^= st

    # make sure bonds are simple 1D chain bonds
    tn_uni.fuse_multibonds_()  

    # cast as MatrixProductOperator
    mpo = tn_uni.view_as_(
        MatrixProductOperator,
        cyclic=False,
        L=circ.N,
    )

    mpo.ensure_bonds_exist()
    return mpo


# ------------------------------------------------------------------
#  MPO x MPO composition
# ------------------------------------------------------------------

def apply_mpo(mpo1: MatrixProductOperator, mpo2: MatrixProductOperator,
                side,
                max_bond=None,
                cutoff=0.0,
                contract=True,
                compress=True,
                **compress_opts):
    """Memory-bounded MPO·MPO product.

    Result is the same compressed product as quimb's apply(contract=True,
    compress=True), but computed by stacking the two operators LAZILY
    (contract=False -> no high-bond per-site contraction) and then compressing
    with a bounded 1D sweep (zip-up / density-matrix). Peak memory is bounded by
    max_bond rather than the full bond1*bond2 product, so it never OOMs on the
    21 GB cap as the state-MPO bond grows.
    """
    # Form the product MPO per-site (contract=True): each site contracts just the
    # two local tensors -> a clean 1-tensor/site MPO with bond bA*bB. This is
    # cheap (bounded by the local bonds, ~tens of MB here). Crucially we keep
    # compress=False so quimb does NOT run its default density-matrix compress,
    # which contracts the high-bond product against its conjugate via cotengra and
    # is what blew past the 21 GB cap.
    if _REF_APPLY:
        a, b = (mpo1, mpo2) if side == "right" else (mpo2, mpo1)
        if side not in ("right", "left"):
            raise ValueError("side must be 'left' or 'right'.")
        return a.apply(b, compress=compress, max_bond=max_bond, cutoff=cutoff,
                       create_bond=True, contract=contract, **compress_opts)

    if side == "right":
        prod = mpo1.apply(mpo2, compress=False, contract=True)
        L = len(mpo1.sites)
    elif side == "left":
        prod = mpo2.apply(mpo1, compress=False, contract=True)
        L = len(mpo2.sites)
    else:
        raise ValueError("side must be 'left' or 'right'.")

    if not compress:
        return prod

    # Compress the clean product MPO with a memory-bounded 1D sweep (zip-up):
    # a left-to-right SVD sweep whose peak memory is bounded by max_bond, never
    # the density matrix. zip-up on a proper 1-tensor/site MPO avoids the
    # permute-arrays mismatch that the lazy (2-tensor/site) structure triggers.
    _kw = {"max_iterations": _FAITHFUL_ITERS} if _COMPRESS_METHOD.startswith("fit") else {}
    try:
        out = tensor_network_1d_compress(
            prod, max_bond=max_bond, cutoff=cutoff, cutoff_mode=_CUTOFF_MODE, method=_COMPRESS_METHOD,
            optimize="auto-hq", permute_arrays=False, inplace=True, **_kw,
        )
    except Exception as _e:
        # #13 robustness: any faithful-method failure (NaN/eigh non-convergence) falls back to the
        # always-stable zipup so a run never crashes mid-absorption.
        out = tensor_network_1d_compress(
            prod, max_bond=max_bond, cutoff=cutoff, cutoff_mode=_CUTOFF_MODE, method="zipup",
            optimize="auto-hq", permute_arrays=False, inplace=True,
        )
    if not isinstance(out, MatrixProductOperator):
        out = out.view_as_(MatrixProductOperator, cyclic=False, L=L)
    out.ensure_bonds_exist()
    return out



# ------------------------------------------------------------------
#  Applying circuits to MPO
# ------------------------------------------------------------------


def apply_circuit(mpo, circ, side, max_bond=None, cutoff=0.0, contract=True, compress=True, **compress_opts):
    return apply_mpo(mpo, mpo_from_circuit(circ), side=side, max_bond=max_bond, cutoff=cutoff, contract=contract, compress=compress, **compress_opts)


# Local adjacent swaps: 1 truncation instead of 47.
# apply_swaps() below routes a SINGLE adjacent transposition through a full-length swap MPO and a
# whole-chain compress, so every bond in the 48-site chain is truncated for a move that touches two
# sites. unswap_greedy probes 3 `hows` per bond and accepts up to UNSWAP_MAX_ACCEPT=4000 of them,
# which is how one d3 absorption reaches ~1.9e6 truncations -- the count implied independently by
# the measured fidelity (rel@2e-3: AMP2_TRUE 4.2e-16 => N ~ 1.9e6).
#
# For adjacent sites the swap is EXACT and purely local: W' = W . SWAP_{i,i+1} means
# W'[.., o_i,o_j, i_i,i_j] = W[.., o_i,o_j, i_j,i_i], i.e. relabel the two input legs. So contract
# the two site tensors, rename the legs, and re-split -- one SVD, one truncated bond.
_LOCAL_SWAPS = os.environ.get("HQP_LOCAL_SWAPS", "0") == "1"


def apply_swap_local(mpo, i, side, max_bond=None, cutoff=0.0, inplace=False):
    """Exact adjacent swap on sites (i, i+1). side='right' swaps inputs, 'left' swaps outputs."""
    out = mpo if inplace else mpo.copy()
    j = i + 1
    ui, uj = out.upper_ind(i), out.upper_ind(j)
    li, lj = out.lower_ind(i), out.lower_ind(j)
    t = out[i] @ out[j]
    if side == "right":
        t.reindex_({li: "_tmp_"}); t.reindex_({lj: li}); t.reindex_({"_tmp_": lj})
    elif side == "left":
        t.reindex_({ui: "_tmp_"}); t.reindex_({uj: ui}); t.reindex_({"_tmp_": uj})
    else:
        raise ValueError(side)
    left_inds = [ix for ix in t.inds if ix in (ui, li)]
    if i > 0:
        left_inds = [out.bond(i - 1, i)] + left_inds
    tl, tr = t.split(left_inds=left_inds, max_bond=max_bond, cutoff=cutoff,
                     cutoff_mode=_CUTOFF_MODE, absorb="both", get="tensors",
                     bond_ind=out.bond(i, j))
    out[i].modify(data=tl.transpose_like_(out[i]).data if False else tl.data, inds=tl.inds)
    out[j].modify(data=tr.data, inds=tr.inds)
    return out


def apply_swaps(mpo: MatrixProductOperator, swaps_l, swaps_r, max_bond=None, cutoff=0.0, to_backend=None, inplace=False):
    N = len(mpo.sites)
    qc_swaps_l = QuantumCircuit(N)
    qc_swaps_r = QuantumCircuit(N)

    for q0, q1 in swaps_l:
        qc_swaps_l.swap(q0, q1)

    for q0, q1 in swaps_r:
        qc_swaps_r.swap(q0, q1)

    mpo_out = mpo if inplace else mpo.copy()

    if _LOCAL_SWAPS:
        adj = [(a, b) for a, b in list(swaps_l) + list(swaps_r) if abs(a - b) == 1]
        if len(adj) == len(swaps_l) + len(swaps_r) and adj:
            for a, b in swaps_l:
                mpo_out = apply_swap_local(mpo_out, min(a, b), "right", max_bond, cutoff, inplace=True)
            for a, b in swaps_r:
                mpo_out = apply_swap_local(mpo_out, min(a, b), "left", max_bond, cutoff, inplace=True)
            return mpo_out

    if len(swaps_l) > 0:
        circ_l = quimb_circuit(qc_swaps_l.inverse().decompose("swap"), Circuit, to_backend=to_backend)
        mpo_out = apply_circuit(mpo_out, circ_l, side="right", max_bond=max_bond, cutoff=cutoff)
    
    if len(swaps_r) > 0:
        circ_r = quimb_circuit(qc_swaps_r.decompose("swap"), Circuit, to_backend=to_backend)
        mpo_out = apply_circuit(mpo_out, circ_r, side="left", max_bond=max_bond, cutoff=cutoff) 

    return mpo_out

