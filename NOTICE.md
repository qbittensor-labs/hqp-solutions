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
| 3a | Charlie | `5F48zgsHT98rSSDunsVPt6GpbA1qsZ1GvkrBPKsyg6sUXeRZ` |
| 3b | Alexey | `5HmQDNh8BrbDeT1bjgqXZ3KGAEb9n6doozNL2mQiJ9rYmuqh` |
| 4 | Alexey | `5FvHcFfsdqMyKSWjtcmAGsXss4hx5AtX5AHYejk7VjgBzuTz` |

> Winning keys are from the public competition record
> (challenges.qbittensorlabs.com). Every level was won by a **different key** —
> except that the key behind Level 2 also submitted one of the two Level 3
> co-winners (`level-3a`). **Level 3 has two co-winning solutions**, from two
> different keys, each of which passed all three validation runs on the Level 3
> circuits; they are preserved side by side as `level-3a` and `level-3b`. Level 4
> was won by a new key, distinct from every prior level, and passed all five of
> its required validation runs.
>
> Levels 1 and 2 name no author in their sources and are credited anonymously.
> No Level 3 or Level 4 submission names its author in-source either; the credited
> handles — **Charlie** (`level-3a`) and **Alexey** (`level-3b`, `level-4`) — are
> the competition-community (Discord) identities of the respective winning-key
> owners, recorded here at the operator's direction (see `docs/RELICENSING.md`).
> They are community handles, not verified legal names. The **Level 3b and Level 4
> winning keys are different keys both attributed to the handle Alexey**; one
> person may compete under more than one key. The Level 2/3a solver builds on the
> published Level 1 solution, and the Level 4 solver reuses the published Level-3a
> engine, each crediting the prior work in-source; per the Enigma design, building
> on published winners is expected and does not imply shared authorship.

We credit these authors for their work. Relicensing to AGPL-3.0 reflects the
participation terms; it does not diminish original authorship.

## Third-party components

### Vendored (bundled in this repository)

One solution bundles third-party source. **Level 4** vendors `p9solver/` — the
MPO + "unswap" readout used for the final peak extraction — from Alexey's public
project `peaked-mpo-solver`:

| Component | Path | Origin | License |
|---|---|---|---|
| **p9solver** (MPO + unswap readout) | `solutions/level-4/p9solver/` | `github.com/alexgalda-m/peaked-mpo-solver` (commit `47c8057`, with documented local changes) | **Apache-2.0** |

It is kept under its original Apache-2.0 license and is **not** relicensed; its
`LICENSE` travels with the vendored package, its text is also surfaced under
`solutions/level-4/licenses/LICENSE.p9solver-Apache-2.0.txt`, and its provenance
and local changes are documented in `solutions/level-4/p9solver/VENDORED.md`.
Apache-2.0 permits use, modification, and redistribution (including in commercial
and closed-source products) **provided the license and attribution are retained**;
it must not be relicensed as a qBitTensor Labs component or folded into a
commercial license. The commercial-license offer in the top-level `README.md`
covers only the custom (AGPL) components, **not** `p9solver/`.

### Installed from upstream (not bundled)

All other solutions are pure-source submissions whose Dockerfiles install their
dependencies from upstream package indexes at image-build time (Level 4 likewise
installs everything except `p9solver` this way), so those projects' licenses
accompany their own distributions. For reference, the solvers build on:

| Component | Role in the solvers | License |
|---|---|---|
| **Qiskit** (+ qasm3-import) | OpenQASM parsing, circuit handling | Apache-2.0 |
| **CuPy** + NVIDIA CUDA wheels | GPU linear algebra (Level 1 MPS engine) | MIT / NVIDIA EULA |
| **PyTorch** | tensor engine (Level 2 TEBD MPS; both Level 3 solvers; Level 4 CPU backend) | BSD-3-Clause |
| **quimb** + qiskit-quimb | tensor-network circuit tooling (Level 2, Level 3a, Level 4) | Apache-2.0 |
| **cotengra**, **autoray** | contraction planning / array dispatch (Level 2, Level 3a, Level 4) | Apache-2.0 |
| **NumPy**, **SciPy** | numerics | BSD-3-Clause |

The Level 2 `fp32_patch.py` is an original runtime wrapper around PyTorch/quimb
entry points (not copied upstream source) and is a custom component. The
`enigma_challenges/` package shipped in the Level 3 and Level 4 solvers is
qBitTensor Labs' own challenge-contract code (relicensed to AGPL-3.0); all other
files in `level-3a`/`level-3b`/`level-4` — plus Level 3b's JSON schemas and
`requirements.txt`, and Level 4's `requirements-*.txt` — are original participant
work, except the vendored `level-4/p9solver/` noted above. The Level 4 solver's
`hqp4/winner/` subtree is the published Level-3a solver (Charlie), carried as a
custom AGPL component with CPU patches, per the Enigma build-on-winners design.

## Corrections

These solvers are user-submitted artifacts, preserved as judged. Classification
and attribution are done on a best-effort basis (see the verification record in
`docs/RELICENSING.md`). If you believe any file is misclassified, misattributed,
or includes your code without proper credit or license, contact
**support@qbittensorlabs.com** and we will correct or remove it.
