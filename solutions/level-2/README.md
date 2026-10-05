# Level 2 solver — self-certifying GPU MPS peak finder (v3)

Winning submission for Hardening Quantum Proof Level 2 (solved 2026-06-20).

Builds on the published Level 1 approach (GPU TEBD matrix-product-state
evolution at escalating bond dimension, canonical beam-search **argmax**
extraction — not sampling) and hardens it:

- **`unswap.py`** — the new technique: sheds bond dimension without destroying
  the peak, keeping truncation error concentrated in the random background.
- **Self-certification** — exact verification of a candidate peak
  (`<s|U|0>`) is infeasible at this treewidth, so the solver certifies its
  answer with independent signals and only reports success when confident
  (a confidently-wrong answer scores worse than no answer on a binary grader).
- **`fp32_patch.py`** — runtime numerical hardening: routes complex64/128 SVD/QR
  through cuSOLVER's robust `gesvd` path with float64 fallback.
- Small circuits (≤ 30 qubits) short-circuit to exact statevector simulation.

The source credits the milestone-1 solution in its comments; the two levels
were won by different keys (see `NOTICE.md`).

## Build / run

```bash
docker build -t hqp-level2 .
docker run --rm --network none --gpus all hqp-level2   # reads /challenge_input/
```

Dependencies (PyTorch cu130, Qiskit, quimb, qiskit-quimb, cotengra, autoray,
NumPy, SciPy) are pip-installed at image build time; nothing third-party is
vendored in this tree.
