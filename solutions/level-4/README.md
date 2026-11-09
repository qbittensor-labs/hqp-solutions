# Level 4 solver — SWAP-network matching + iterative excision portfolio

A winning submission for Hardening Quantum Proof Level 4 (difficulty 4, solved
2026-10-14). Credited to the competition handle **Alexey**. CPU-only.

A difficulty-4 circuit is a shallow trained peaked circuit obfuscated two ways:
**inserted CZ "couples"** (cancellable CZ pairs whose intervening 1-qubit gates
are slightly detuned to hide the cancellation) and **mirror blocks**
`U · SWAP_τ · U†` (as an operator, a wire permutation `Π_τ` times a small
residual `D`). The solver peels both off to recover a small circuit a
tensor-network simulator can read, then reads the peak off the remainder.

- **De-obfuscate** (`hqp4/undo_wrap.py`, `hqp4/l3/`, `hqp4/fold_z.py`) — undo the
  inserted CZ couples at a displacement threshold, then fold the Pauli-Z
  byproducts.
- **Find the mirrors — two detectors.** The twin-counting census
  (`hqp4/census_x.py`, with d4-loosened tolerances) wraps the Level-3a engine;
  where the census finds no clean module, **`hqp4/swapnet_solve.py`** recovers the
  mirror involution `τ` directly from CZ geometry — a wire pair carrying three CZs
  in a short span is a candidate, and the densest window of disjoint such triples
  gives a fixed-point-free involution. This SWAP-network detector is the core
  Level-4 contribution.
- **Excise — a portfolio** (`hqp4/solve3.py`, `hqp4/mkred_multi.py`) — measure
  each block's residual `D`, replace the block with (relabel + measured `D`), and
  iterate: try many candidate cuts, keep the lowest-error / most-CZ-removed, and
  repeat until the circuit stops shrinking. Each excision is exact up to measured
  model error, so the greedy loop is safe.
- **Read the peak** (`p9solver/`, vendored) — an MPO + "unswap" readout on the
  shrunken remainder, with a centre-ratio retry ladder; the answer is remapped
  back through every excision's permutation.
- **Orchestrate** (`hardening_quantum_proof.py`) — runs a portfolio of attempts
  (varying undo threshold, cut policy, readout seed) under a thread cap and a
  wall-clock watchdog, reporting on best-effort early stop.

The excision backend in **`hqp4/winner/`** is the published Level-3a solver
(credited to **Charlie**), carried here with CPU patches (CPU torch device, an
SVD fallback chain) — per the Enigma design, winners are open-sourced so
successors start from the frontier. See `../../NOTICE.md` for the full
component-by-component attribution and licensing.

## Build / run

```bash
docker build -t hqp-level4 .
docker run --rm --network none hqp-level4   # CPU-only; reads /challenge_input/
```

The image builds two isolated `uv` virtual environments — `pp` (the solver:
torch CPU, Qiskit, quimb) and `p9` (the vendored `p9solver` readout) — on
Python 3.10, with a Python 3.12 stdlib-only entry wrapper. The validated solver
configuration is baked into the image (`HQP_WATCHDOG_AT`, `HQP_DEADLINE`, and the
per-stage thresholds), so no custom env is required. The container runs non-root
as `miner` under `/tmp` and needs no network at run time (network is used only at
build for package installs).

## Third-party code

This solver **vendors** one third-party component, `p9solver/` (the MPO + unswap
readout), from Alexey's public [`peaked-mpo-solver`](https://github.com/alexgalda-m/peaked-mpo-solver)
project under **Apache-2.0**. It is kept under its original license, with its
provenance and local changes documented in `p9solver/VENDORED.md`; its license
text is also surfaced under `licenses/`. All other source in this tree is custom
(relicensed to AGPL-3.0). See `../../docs/RELICENSING.md`.
