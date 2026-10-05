# Level 3 solver — τ-block excision + concurrent D-measurement (v13)

A winning submission for Hardening Quantum Proof Level 3 (difficulty 3, solved
2026-09-26). Credited to the competition handle **Charlie**.

A difficulty-3 circuit is a shallow trained peaked circuit with **mirror blocks**
`U · SWAP_τ · U†` inserted. As an operator each block is a wire permutation `Π_τ`
times a small residual `D`. This solver **excises** the blocks — replacing each by
its relabelling and re-inserting the *measured* `D` — which turns the d3 circuit
back into the d1-shaped circuit it was built from, and that shape is directly
simulable by the canonical forward-MPS ladder.

- **`excision_solver.py` / `v13_solver.py`** — detect the τ-blocks, choose the cut
  tolerance, excise, and re-insert the measured residual. Builds on the published
  Level 1/2 MPS + canonical beam-search (argmax, not sampling) approach.
- **`syncd.py` / `unswap.py`** — measure each block's residual `D` by absorbing its
  two halves onto the MPO chain; the synchronous two-sided absorber routes both
  legs together to avoid the bond-blowup the independent-routing reference hits.
- **`ens_pool.py`** — a multi-process ensemble of cheap MPS runs with pooled
  candidates and a z-scored verdict: one bond-limited simulation is a biased coin,
  so many independent runs plus an error bar decide the reported peak.
- **`dworker.py` / `syncworker.py`** — concurrent CPU workers that measure the
  per-block residuals in parallel while the GPU MPS ladder runs.
- Small circuits (≤ 30 qubits) short-circuit to exact statevector simulation.
- **Best-effort reporting** — a confidently-wrong answer scores no better than no
  answer on a binary grader, so the solver reports only when its ensemble agrees.

The source builds on the published Level 1/2 solutions, per the Enigma design
(winners are open-sourced so successors start from the frontier).

## Build / run

```bash
docker build -t hqp-level3a .
docker run --rm --network none --gpus all hqp-level3a   # reads /challenge_input/
```

The validated solver configuration is baked into the image (`HQP_US_ANTILIVELOCK`,
`HQP_US_SMART_ORDER`, `HQP_US_EARLY_ABANDON`, `HQP_FAITHFUL`,
`HQP_COMPRESS_METHOD=direct`, `HQP_BEST_EFFORT`; `WALL_TIME=14400`), so no custom
env is required. On a larger GPU, raise `HQP_US_MAXBOND` (default 1024 is safe on
24 GB; 2048–4096 suits a 48–96 GB card).

Dependencies (PyTorch cu130, Qiskit, quimb, qiskit-quimb, cotengra, autoray,
NumPy, SciPy) are pip-installed at image-build time; nothing third-party is
vendored in this tree.
