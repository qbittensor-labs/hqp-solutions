# Hardening Quantum Proof — Solver Submission

Classical peak-finder for BlueQubit peaked circuits (Enigma SN63). Given an
OpenQASM circuit, it outputs the single bitstring with the highest measurement
probability (the "peak").

## Approach

Peaked circuits embed one heavy output (peak weight δ_s ≈ 0.1) inside a
random-looking, densely-entangled circuit (`u` + `cz`, all-to-all). They are
built to defeat 1D-MPS and low-treewidth tensor contraction
(arXiv:2510.25838).

The key observation: a **truncated** matrix-product state, evolved with proper
mixed-canonical-form truncation, *denoises* away the exponentially-small random
background and concentrates amplitude on the embedded peak. Extracting the
**argmax** (via canonical beam search) — not sampling — then recovers the peak
even at modest bond dimension. (The reference solver fails because it samples
10k shots, which miss a low-probability peak at low fidelity.)

`mps_gpu.py` implements a custom GPU TEBD MPS:
- mixed-canonical-form truncation (optimal local SVD truncation),
- a swap network for the all-to-all long-range `cz` gates,
- canonical beam search (left partial-norm = true prefix marginal) for top-K.

`solve_hqp.py` climbs a bond-dimension ladder (χ = 48 → 8192) within the 4 h
wall budget, stops early once the argmax is stable across 3 consecutive levels
at χ ≥ 1024 (or the bond saturates / budget runs out), and reports the argmax.

**Verification.** Exact single-amplitude verification is infeasible here — these
circuits have treewidth ≈ qubit count, so contracting one amplitude `<s|U|0>`
costs ~2⁵⁸ flops (cotengra-optimized width 50). Instead the solver uses two
independent signals:
1. **Convergence** — the argmax is stable as χ grows (and its MPS weight grows).
2. **Independent cross-check** — it re-runs at the top χ with different qubit
   orderings (independent truncation errors via different swap networks) and
   takes the answer from the most-peaked run (highest MPS weight). Agreement
   across orderings is reported as `consensus` in `solve_info.json`
   (d1 samples: **3/3**). Graceful degradation under OOM / budget.

## Validated results (validator-parity Docker, RTX PRO 6000)

| Circuit | Qubits | 2q gates | Result |
|---------|--------|----------|--------|
| `d0_s0_trivial` | 5  | 0   | ✅ exact |
| `d1_s1_4043cafb` | 46 | 684 | ✅ exact at χ=64 (rank 0), confirmed χ=128, 256 |
| `d1_s2_adeddcf3` | 48 | 853 | ✅ exact at χ=64 (rank 0), confirmed χ=128, 256 |

Both difficulty-1 samples are cracked exactly in ~75–155 s; the production
solver climbs higher (χ≈1024, ~1 h) for safety margin on unseen circuits.

## Files

- `solve_hqp.py` — entrypoint (reads `/challenge_input/`, stdout protocol).
- `mps_gpu.py`   — GPU TEBD MPS engine + canonical beam search.
- `qload.py`     — OpenQASM 2/3 loader (`u`+`cz`).
- `Dockerfile`   — `python:3.12-slim` + qiskit + cupy + CUDA 12 pip wheels.

## Build / test

```bash
# Build (validator/workbench vendors enigma_challenges into the context)
docker build --platform linux/amd64 -t hqp-solver .

# Local workbench test
python -m workbench test hardening-quantum-proof \
  --circuit d1_s1_4043cafb --solution ./ --mode docker
```

Tunables via env: `HQP_CHI_LADDER`, `HQP_WALL_BUDGET`, `HQP_BEAM`.
