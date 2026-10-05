# Level 3 solver — structured-MPO peaked-circuit engine

A winning submission for Hardening Quantum Proof Level 3 (difficulty 3, solved
2026-09-26). Credited to the competition handle **Alexey**. This is one of two
independent co-winning Level 3 solutions in this repository (see `../level-3a/`
and the root `README.md`).

A difficulty-3 circuit hides its peak behind inserted mirror blocks. This solver
attacks it with a **structured matrix-product-operator (MPO) engine** that
recognises the circuit's structure, routes the contraction to keep bond dimension
down, and reduces the problem before extracting the peak — a different
architecture from `level-3a`'s excision approach, arriving at the same solve.

- **`hqp_driver.py`** — the orchestration driver: budgets the wall clock
  (`HQP_WALL_BUDGET`/`HQP_RESERVE`), runs the pipeline, checkpoints, and reports.
- **`hqp_structure.py`** — circuit structure analysis (Qiskit `Collect2qBlocks` /
  `ConsolidateBlocks`) that feeds the reduction.
- **`enigma_peaked/`** — the engine package: MPO construction (`engine/mpo.py`),
  generator/migration planning (`engine/generator.py`, `engine/plan.py`), unswap
  (`engine/unswap.py`), extraction (`engine/extraction.py`), frame/permutation
  bookkeeping (`frames/`), and structural gadget / Pauli-marginal detection
  (`structure/`).
- **`src/peaked/`** — the numerical core: Gram-SVD compression (`gram_svd.py`),
  bond routing (`bond_route.py`, `horizon_route.py`), the MPS/MPO primitives
  (`mps.py`, `core_mpo.py`), and a torch/numpy backend shim (`torch_np.py`).
- **`src/rt/`** — a runtime contract layer (`contracts.py`) and a source ledger
  (`source_ledger.py`) that records provenance; `schemas/` holds the JSON schemas
  for its artifact/claim/experiment records.
- **`scripts/`** — checkpoint, readout, and reduced-QASM tooling used by the driver.

## Build / run

```bash
docker build -t hqp-level3b .
docker run --rm --network none --gpus all hqp-level3b   # reads /challenge_input/
```

The validated configuration is baked into the image (`WALL_TIME=14400`,
`HQP_WALL_BUDGET=13400`, `HQP_RESERVE=500`, `HQP_DEVICE=cuda:0`,
`HQP_DTYPE=complex64`), so no custom env is required.

Dependencies are pinned in `requirements.txt` (PyTorch cu128, Qiskit, quimb,
qiskit-quimb, cotengra, autoray, numba, NumPy, SciPy, …) and installed with `uv`
at image-build time; nothing third-party is vendored in this tree.
