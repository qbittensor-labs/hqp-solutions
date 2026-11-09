# Vendored: p9solver (MPO + unswap solver)

| Field | Value |
|---|---|
| Source | public repo `peaked-mpo-solver` (`github.com/alexgalda-m/peaked-mpo-solver`, local checkout `../peaked-mpo-solver`) |
| Commit | `47c8057dae1809eb18a391a548362e20660221e4` (2026-08-07, "retention: survive gesvd nonconvergence in tracked_svd") |
| Licence | Apache-2.0 (`LICENSE`, unchanged) |
| Vendored on | 2026-09-28, refactor phase P4 |
| Export command | `git -C ../peaked-mpo-solver archive 47c8057 src tests pyproject.toml uv.lock LICENSE \| tar -x -C vendor/p9solver` |
| Local-change marker | `p9solver.vendor_info.LOCAL_CHANGES = "peakpilot-p4"` (written into `summary.json` under `effective_config.local_changes`) |

The public repo is frozen and was not modified. Fork-only code was ported from
`attic/portal/peaked-mpo-solver-fable/src/p9solver/` (the "fable" fork, based on public
commit `3bcdc1e`). Ported code is marked `[peakpilot port]` in comments; other local
edits are marked `[peakpilot]`.

Run: `uv sync --project vendor/p9solver`, then
`uv run --project vendor/p9solver p9solve --qasm <file> ...`.
Tests: `uv run --project vendor/p9solver pytest -q vendor/p9solver/tests` (all fast; no
test runs a full P9 solve).

## Default behaviour versus upstream

With no new flags, the solver follows the upstream path except for:

1. `--early-stopping-gates` defaults to 100. Upstream hard-coded 0. Pass
   `--early-stopping-gates 0` for the upstream path. The remaining work gates are applied
   during MPS extraction (`mpo_to_mps`, `max_bond=4096`, same `--cutoff`).
2. Non-finite values raise `FloatingPointError` where upstream replaced them with zeros.
3. `summary.json` gains an `effective_config` block, and the same block is printed to
   stderr at start.

## Local changes

| File | Change | Reason |
|---|---|---|
| `pyproject.toml` | `readme = "README.md"` changed to `"VENDORED.md"` | README.md was not vendored; hatchling refuses to build without the file |
| `pyproject.toml`, `uv.lock` | added `[dependency-groups] dev = ["pytest>=8"]`; lock re-resolved. Only pytest and its dependencies were added (exceptiongroup, iniconfig, pluggy, pytest, tomli); runtime pins unchanged | `uv run --project vendor/p9solver pytest` must work |
| `src/p9solver/cli.py` | new flag `--early-stopping-gates` (default 100), passed to `mpo_compress_unswap`, replacing the hard-coded `early_stopping_gates=0` | Plan P4.2: upstream forced the early stop off |
| `src/p9solver/cli.py` | new flags `--unswap-apply-cutoff`, `--swap-cutoff`, `--absorb-swaps-as-perm` (all off by default); passed to the pipeline; the routing perms from the `route_perms` stats row are passed to `mpo_to_mps` | Plan P4.1: port from the fable fork |
| `src/p9solver/cli.py` | effective-config block printed to stderr before any compute; `make_summary` writes `effective_config` | Plan P4.4 |
| `src/p9solver/pipeline.py` | `unswap(..., unswap_apply_cutoff=None)`: kept swaps are applied at that cutoff when set; full-probe reuse disabled when set | port of fork `--unswap-apply-cutoff` |
| `src/p9solver/pipeline.py` | `mpo_compress_unswap(..., unswap_apply_cutoff, absorb_swaps_as_perm, swap_cutoff)`: `_eff_cutoff` for SWAP-only layers; perm-tracked absorb via `remap_work_layer`; per-side routing perm updated on each consumed layer; `route_perms` stats row when the flag is on | port of fork `--swap-cutoff`, `--absorb-swaps-as-perm` |
| `src/p9solver/pipeline.py` | `mpo_to_mps(..., route_perm_left=None, route_perm_right=None)`: leftover layers remapped through the perm, and the perm composed into the measurement permutation; warns when leftover work layers remain | port of fork `--absorb-swaps-as-perm` |
| `src/p9solver/pipeline.py` | `set_nonfinite_context(...)` at each absorb-loop iteration (work gates consumed) and at extraction | lets a non-finite error name the gate position |
| `src/p9solver/mpo.py` | `_bound_scales` (torch only): `torch.nan_to_num` + "RUN TAINTED" warning replaced by `raise_nonfinite("mpo._bound_scales ...")`. The finite rescale (divide by max when max > 1e3) is kept | Plan P4.3: loud NaN. The rescale changes only the global operator scale, which argmax, sampling and the relative (`rsum2`) cutoff ignore |
| `src/p9solver/torch_hardening.py` | `rsvd` (complex64): `torch.nan_to_num` replaced by `raise_nonfinite("torch_hardening.rsvd ...")`. The finite rescale (max > 1e4) is kept | Plan P4.3; same reasoning for the rescale |
| `src/p9solver/retention.py` | `tracked_svd`: non-finite input raises for every backend (before, torch input went through `nan_to_num`, and numpy input fell into the gesvd -> QR -> jitter fallback chain with `check_finite=False`); non-finite singular values raise after the SVD. The torch finite rescale (max > 1e4) and the finite-input fallback chain are kept | Plan P4.3 |
| `src/p9solver/nonfinite.py` (new) | `set_context`, `raise_nonfinite(site, x)` (message names site, phase, work gates consumed, shape, dtype), `is_finite_array` | shared loud-NaN helper |
| `src/p9solver/vendor_info.py` (new) | vendored commit, local-change marker, `build_effective_config(args)` | Plan P4.4 |
| `src/p9solver/perm_absorb.py` (new) | copied unchanged from the fable fork | Plan P4.1 |
| `src/p9solver/swap_permutation.py` (new) | copied unchanged from the fable fork | Plan P4.1 |
| `tests/test_swap_permutation.py`, `tests/test_swap_perm_principle.py` (new) | copied unchanged from the fable fork | tests for the ported modules |
| `tests/test_peakpilot_local_changes.py` (new) | NaN/inf raise in `tracked_svd` (and in the torch paths when torch is installed); `--early-stopping-gates` default 100; fork flags default off; quimb reaches the SVD with `cutoff_mode=4` (`rsum2`); tiny 6-qubit mirror run writes `effective_config` and returns the known peak, qubit 0 leftmost | Plan P4.5 |

## Fork items not ported, or ported differently

- `--cutoff-schedule`: upstream already has it (commit `6edf6ce`, same `frac:cutoff`
  contract). The upstream version is kept; the fork copy was not ported.
- `--save-mpo`: upstream already has it (writes `mpo_dump.pkl` plus factor-bundle
  metadata). The upstream version is kept. The fork version (`final_mpo.pkl`,
  `final_mps.pkl`) was not ported.
- The fork's change to the `partial run` sampling condition (sample a capped run that
  completed) was not ported, because it would change default behaviour.
- The fork's `tests/oracle_pipeline.py` (a differential oracle script, not a pytest test)
  was not ported.

## Known interactions

- `--absorb-swaps-as-perm` is a diagnostic. The fork recorded it as P9-negative
  (`entanglement_blowup` at gate 286/1885). The perm is not propagated through the
  absorb look-ahead or post-unswap rewiring, the same as in the fork.
- With `--no-track-fid10` on numpy, quimb's own `svd_truncated_numpy` is used. It raises
  `ValueError("array must not contain infs or NaNs")` on non-finite input, which is still
  loud, but it is not a `FloatingPointError`.
