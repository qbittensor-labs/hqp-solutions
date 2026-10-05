# Relicensing policy and procedure

This repository normalizes the licensing of winning Hardening Quantum Proof
submissions to **AGPL-3.0** for the custom components, while leaving bundled
third-party software under its original license. This document states the
policy, the automated classification, and the verification performed before
public release. It follows the same policy proven on the
[breaking-rsa-solutions](https://github.com/qbittensor-labs/breaking-rsa-solutions)
archive.

## Policy

1. **Custom components → AGPL-3.0.** Files authored by the participant (or by
   qBitTensor Labs) — the solver entrypoint, custom simulation kernels, search
   drivers, pipeline scripts — get a uniform AGPL-3.0 header with a
   `© 2026 qBitTensor Labs` copyright and an "Original author" credit line. This is
   authorized by the Enigma rules: winning submissions are AGPL-3.0 and IP in
   custom components is assigned to qBitTensor Labs.
2. **Third-party components → unchanged.** Vendored libraries (quantum
   simulation frameworks such as Qiskit / Aer, tensor-network libraries, CUDA
   toolkit components, …) keep their own license. Their headers are **not**
   touched, and their license texts are preserved under
   `solutions/level-<n>/licenses/`.
3. **Patches to third-party code inherit the third-party license.** A `.diff` or a
   modified upstream source file is a derivative of that upstream and stays under
   its license, with the modification noted.
4. **Submitted binaries are preserved as submitted.** The submission is the
   artifact of record; we keep exactly what each submission included so it
   builds and runs as judged. A binary cannot carry a text header, so it is not
   "relicensed": a **custom** binary is a build artifact of AGPL-licensed source
   shipped alongside it, and a **third-party** binary retains its upstream
   license (see `NOTICE.md`).

## Automated classification (`tools/relicense.py`)

The tool walks a solution tree and labels each file:

- **CUSTOM** — a text source file (`.py .cu .cuh .c .h .sh`, `Dockerfile`) that is
  **not** under a vendored path and does not match a known third-party binary/name.
  Its existing header (`All rights reserved` / MIT / etc.) is replaced with the
  AGPL header for the extension's comment style.
- **THIRD_PARTY** — anything under `vendor/`, `licenses/`, or matching known
  third-party names. The per-challenge third-party name patterns are reviewed
  when each solution is imported. Left untouched.
- **BINARY** — no text header; skipped.
- **QBTL_PKG** — the `enigma_challenges/` contract package (qBitTensor Labs' own);
  normalized to the AGPL header.

```
python tools/relicense.py --src /path/to/extracted/level-2 --report            # dry run
python tools/relicense.py --src /path/to/extracted/level-2 \
    --dst solutions/level-2 --author "<verified author>" --apply              # write
```

`--author` is **required** and must be verified against the submission itself —
its own copyright/authorship headers, or the identity of the winning key's owner.
Do **not** infer authorship from code similarity to another level: Enigma
publishes winning solutions precisely so later competitors can build on them
(the Level 2 winner here started from the published Level 1 code), so shared
code lineage does not imply shared authorship. When the submission does not
name its author, use `--author "an anonymous competition participant"`.

## Verification record

The following was verified before public release (2026-09-06):

- **Authority to relicense.** The published
  [Competition Rules](https://www.qbittensorlabs.com/enigma/rules) (§5.1
  irrevocable IP assignment of all submissions; §5.3 AGPL-3.0 release of winning
  submissions; §5.5 participant warranty that every included component is theirs
  to assign or compatibly licensed) give qBitTensor Labs ownership of the custom
  components and the right to release them under AGPL-3.0 and to dual-license.
  Confirmed in force before these submissions (operator, 2026-09-02, for the
  breaking-rsa release under the same rules).
- **Artifacts of record.** Both winning submissions were retrieved directly from
  the platform submission store (the artifacts as judged) and relicensed from
  those originals.
- **Attribution.** Neither submission names its author anywhere in its sources —
  both are credited "an anonymous competition participant". Cross-checked
  against the public winning-key record (challenges.qbittensorlabs.com):
  Level 1 won 2026-06-13 by `5Gx9dtqd…Ew94`; Level 2 won 2026-06-20 by
  `5F48zgsH…XeRZ`. Distinct keys. The Level 2 source credits the milestone-1
  approach in three files; those credits are preserved verbatim.
- **Custom/third-party split.** Every file in both trees was individually
  reviewed (13 + 15 files): all solver sources are participant-original; the
  `enigma_challenges/` contract package is qBitTensor Labs' own; `fp32_patch.py`
  (Level 2) is an original runtime wrapper, not copied upstream source. Neither
  submission vendors third-party code — all dependencies are pip-installed from
  upstream at image-build time (see `NOTICE.md`).
- **Hygiene.** Both trees swept for secrets, credentials, endpoints, and
  embedded paths; no binaries present; all relicensed Python files re-parsed
  clean.

The artifact retrieval, file reviews, winning-key cross-check, and secrets sweep
were performed by AI (Claude) under qBitTensor Labs' direction, following the
process used for the breaking-rsa-solutions release. Approved for release by
qBitTensor Labs, 2026-09-06.

### Level 3 — `level-3a` — added 2026-09-29

Level 3 (challenge milestone `18065ddd-…`, difficulty 3, marked Complete
2026-09-26) was won by **more than one submission**. This record covers the first
solution published, `solutions/level-3a`; the second co-winner (`level-3b`) is
recorded in the entry that follows.

- **Artifact of record.** The `level-3a` package (submission `5a2754a9-…`, key
  `5F48zgsH…XeRZ` — the Level 2 key) was retrieved directly from the platform
  submission store via the admin review endpoints and relicensed from that
  original. Accepted, 3/3 validation runs Success (our SN63 validator
  `5EZ52JMq…7kR8`, which ran twice, and Rizzo `5GzjAcUc…dTs63`). 38 files.
- **Attribution.** The submission names no author in its sources. The credited
  handle **Charlie** is the competition-community (Discord) identity of the
  winning key's owner, supplied by the operator. This provenance is the Discord
  identity of the winning-key owner, **not** an in-source authorship header; it is
  recorded as such and not represented as self-identification. The
  `enigma_challenges/` package is qBitTensor Labs' own code and is credited to
  qBitTensor Labs. Building on the published Level 1/2 solutions is expected under
  the Enigma design and does not imply shared authorship.
- **Custom/third-party split.** Every file was reviewed (38 files). All solver
  sources are participant-original; the `enigma_challenges/` contract package is
  qBitTensor Labs' own (relicensed to AGPL). The submission vendors no third-party
  code — all dependencies are pip-installed from upstream at image-build time (see
  `NOTICE.md`).
- **Hygiene.** The tree was swept for secrets, credentials, endpoints, and embedded
  local paths — clean (no platform endpoints, credentials, or presigned URLs). No
  binaries. The Dockerfile is network-isolated at run time (network only at build
  for pip installs) and runs non-root as `miner` under `/tmp`. All relicensed
  Python re-parsed clean; file-set parity with the original verified.

Performed by AI (Claude) under qBitTensor Labs' direction, following the same
process. Approved for release by qBitTensor Labs, 2026-09-29.

### Level 3 — `level-3b` — added on release

This is the second Level 3 co-winner referenced above, published a week after
`level-3a` once its on-chain payout transaction had been issued. Level 3 was won
**twice**: two independent submissions, from two different keys, each passed all
three required validation runs on the Level 3 circuits. This is a departure from
the one-winner-per-milestone norm and resulted from a platform bug — a second
full-pass submission was accepted for validation after the milestone had already
been marked Complete, then auto-Rejected on status even though all three of its
validation runs succeeded. Rather than discard a genuine solve, both are released
as co-winners.

- **Artifact of record.** The `level-3b` package (submission `a783ae7a-…`, key
  `5HmQDNh8…rYmuqh` — a new key, distinct from every prior level) was retrieved
  directly from the platform submission store via the admin review endpoints and
  relicensed from that original. Status Rejected-but-3/3-Success (the milestone bug
  above): all three validation runs Success — our SN63 validator (`5EZ52JMq…7kR8`,
  which ran twice) and Rizzo (`5GzjAcUc…dTs63`). 62 files.
- **Attribution.** The submission names no author in its sources. The credited
  handle **Alexey** is the competition-community (Discord) identity of the winning
  key's owner, supplied by the operator (the Discord profile reports the name
  "Alexey Galda"; it is held to the handle "Alexey" here pending explicit consent
  to publish a full legal name). This provenance is the Discord identity of the
  winning-key owner, **not** an in-source authorship header, and is recorded as
  such. The `enigma_challenges/` package is qBitTensor Labs' own code and is
  credited to qBitTensor Labs.
- **Custom/third-party split.** Every file was reviewed (62 files). All solver
  sources are participant-original; the `enigma_challenges/` contract package is
  qBitTensor Labs' own (relicensed to AGPL); the JSON schemas and `requirements.txt`
  are original config/data (classified OTHER, copied verbatim, no header injected).
  The submission vendors no third-party code — dependencies are pinned in
  `requirements.txt` and installed from upstream at image-build time (see
  `NOTICE.md`). The source was submitted under an `hqp_mpo/` package root; it was
  flattened to the `level-3b/` directory so the tree matches the other levels and
  the Dockerfile's build context is preserved.
- **Hygiene.** The tree was swept for secrets, credentials, endpoints, and embedded
  local paths — clean (the only URLs are build-time package indexes and JSON-schema
  `$id` placeholders; no platform endpoints, credentials, or presigned URLs). No
  real binaries (two empty marker files only). The Dockerfile is network-isolated
  at run time (network only at build for pip/uv installs) and runs non-root as
  `miner` under `/tmp`. All relicensed Python re-parsed clean; file-set parity with
  the original verified.

Performed by AI (Claude) under qBitTensor Labs' direction, following the same
process. Approved for release by qBitTensor Labs on `level-3b` publication.

**A note on best effort.** These solvers are user-submitted artifacts, preserved
as judged. Classifying every line of third-party heritage in submissions we did
not write is inherently best-effort; the participants' warranty (Rules §5.5)
covers what we cannot see. If you believe any file is misclassified,
misattributed, or includes your code without proper credit or license, contact
**support@qbittensorlabs.com** and we will correct or remove it.
