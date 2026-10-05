# NOTICE — Attribution and Third-Party Components

## Custom components

The custom solver components in this repository are © 2026 qBitTensor Labs and are
licensed under the GNU Affero General Public License v3.0 (see `LICENSE`).

Under the published [Enigma Official Competition Rules](https://www.qbittensorlabs.com/enigma/rules)
(§5.1 assignment of submissions; §5.3 open-source release), winning submissions are
released under AGPL-3.0 and the intellectual property in custom components is
assigned to qBitTensor Labs. Original authorship is credited as follows:

| Level | Original author | Winning key (public record) |
|---|---|---|
| 1 | an anonymous competition participant | `5Gx9dtqdQZ1MRp9VhyBftCPSYhr5YSYKawQMKRwbzwsqEw94` |
| 2 | an anonymous competition participant | `5F48zgsHT98rSSDunsVPt6GpbA1qsZ1GvkrBPKsyg6sUXeRZ` |

> Winning keys are from the public competition record
> (challenges.qbittensorlabs.com). The two levels were won by **different keys**.
> Neither submission has been confirmed to name its author, so both are credited
> anonymously unless the authors come forward. The Level 2 solver builds on the
> published Level 1 solution and credits it in its source comments; per the
> Enigma design, building on published winners is expected and does not imply
> shared authorship.

We credit these authors for their work. Relicensing to AGPL-3.0 reflects the
participation terms; it does not diminish original authorship.

## Third-party components

**Neither solution bundles third-party code in this repository.** Both are
pure-source submissions whose Dockerfiles install their dependencies from
upstream package indexes at image-build time, so those projects' licenses
accompany their own distributions. For reference, the solvers build on:

| Component | Role in the solvers | License |
|---|---|---|
| **Qiskit** (+ qasm3-import) | OpenQASM parsing, circuit handling | Apache-2.0 |
| **CuPy** + NVIDIA CUDA wheels | GPU linear algebra (Level 1 MPS engine) | MIT / NVIDIA EULA |
| **PyTorch** | GPU tensor engine (Level 2 TEBD MPS) | BSD-3-Clause |
| **quimb** + qiskit-quimb | tensor-network circuit tooling (Level 2) | Apache-2.0 |
| **cotengra**, **autoray** | contraction planning / array dispatch (Level 2) | Apache-2.0 |
| **NumPy**, **SciPy** | numerics | BSD-3-Clause |

The Level 2 `fp32_patch.py` is an original runtime wrapper around PyTorch/quimb
entry points (not copied upstream source) and is a custom component.

## Corrections

These solvers are user-submitted artifacts, preserved as judged. Classification
and attribution are done on a best-effort basis (see the verification record in
`docs/RELICENSING.md`). If you believe any file is misclassified, misattributed,
or includes your code without proper credit or license, contact
**support@qbittensorlabs.com** and we will correct or remove it.
