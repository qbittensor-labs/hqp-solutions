"""Routing-SWAP as permutation (not gate) — the fix for swap_thrash accumulation.

FINDING (2026-07-02): the absorb loop applies routing SWAP gates via _swap_mpo +
apply_mpo (contract + compress). A SWAP is a permutation, but this materializes it as a
bond-2 MPO and contracts it with truncation. Measured cost on a bond-128 MPO:
  - 1 SWAP as gate:      bond 128 -> 378   (grows!)
  - wall of 10 SWAPs:    bond 128 -> 1106, size 656k -> 8.3M   (12x blowup)
Applying the SAME swap as an index permutation (relabel the physical indices) is EXACT
(operator error ~3e-15 vs the gate result) and BOND-PRESERVING (128 -> 128, even over a
10-swap wall). This is the per-cycle inefficiency behind s1's swap_thrash: each stall
cycle absorbs routing SWAP layers as gates, inflating the MPO, which then needs unswap,
which reroutes into more SWAPs — net-negative every cycle.

A one-sided routing SWAP(i,i+1) on the MPO's ket side (S.M) is exactly: swap the upper
(ket) physical index LABELS of sites i and i+1. This leaves the MPS chain and all bonds
untouched (cost O(1), zero SVD, exact). It permutes the site<->qubit mapping, which the
caller MUST track (same as the unswap perm_left/perm_right bookkeeping) and undo at
extraction.

Validated: `test_swap_permutation.py`.
"""


def apply_routing_swaps_permutation(mpo, swaps, side="left", inplace=False):
    """Apply adjacent routing SWAPs as exact, bond-preserving index permutations.

    swaps: iterable of (i, i+1) adjacent site pairs.
    side:  'left' = S.M (permute ket/upper indices); 'right' = M.S (permute bra/lower).
    Returns (mpo, perm_update) where perm_update maps site -> the qubit label now carried
    there, so the caller can compose it into its running permutation.

    The MPS chain, tensors, and all bond dimensions are untouched — only physical index
    labels are swapped. O(#swaps), no SVD, exact.
    """
    m = mpo if inplace else mpo.copy()
    phys = m.upper_ind if side == "left" else m.lower_ind
    for (a, b) in swaps:
        if abs(a - b) != 1:
            raise ValueError(f"routing SWAP must be adjacent sites, got ({a},{b})")
        i, j = (a, b) if a < b else (b, a)
        pi, pj = phys(i), phys(j)
        ti = m[i]
        tj = m[j]
        # swap the two physical index labels between the two site tensors
        ti.reindex_({pi: "_swp_tmp"})
        tj.reindex_({pj: pi})
        ti.reindex_({"_swp_tmp": pj})
    return m
