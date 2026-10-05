# Hardening Quantum Proof — Winning Solutions

Open-source archive of the winning solvers from the **Enigma / Hardening Quantum
Proof** challenge ([qbittensorlabs.com/enigma](https://www.qbittensorlabs.com/enigma)),
operated by qBitTensor Labs on Bittensor Subnet 63, developed in collaboration
with [BlueQubit](https://www.bluequbit.io/).

Hardening Quantum Proof posts increasingly hard **peaked quantum circuits** —
obfuscated circuits whose output distribution hides one disproportionately
likely bitstring — and pays a prize to the first solver that finds the peaked
state classically, in an identical hardened sandbox under a fixed wall-clock.
If classical solvers can crack these circuits, they cannot serve as a "quantum
proof" — so every winning solver directly hardens the design of verifiable
quantum advantage tests. Every winning submission is released under **AGPL-3.0**;
this repository collects them, normalizes their licensing, and preserves each
level so the next competitor — and the wider research community — can build on
what came before.

## The level ladder

| Level | Method | Winner | Solved |
|---|---|---|---|
| 1 | matrix product state simulation driven by canonical beam search | anonymous participant | 2026-06-13 |
| 2 | Level 1 approach + "unswap" technique (sheds bond dimension without destroying the peak) | anonymous participant | 2026-06-20 |
| 3a | τ-block excision + concurrent D-measurement: excises the mirror blocks to turn a difficulty-3 circuit back into its difficulty-1 base, then finds the peak with the Level 1/2 MPS beam search | Charlie | 2026-09-26 |
| 3b | structured-MPO engine: Gram-SVD compression with a bond-routing schedule and unswap, plus structural gadget / Pauli-marginal detection and two-sided checkpointing | Alexey | 2026-09-26 |

The Level 2 winner started from the published Level 1 code and credits the
Level 1 solution in its source comments. This is Enigma working as designed —
winning solutions are open-sourced so each successor starts from the frontier.
Levels 1 and 2 were won by different keys; Level 3a was submitted by the key that
also won Level 2 (see `NOTICE.md`).

**Level 3 has two winning solutions.** Two independent submissions, from different
keys, each passed all three validation runs on the Level 3 circuits. They are
preserved side by side as `level-3a/` (Charlie) and `level-3b/` (Alexey) — two
distinct solvers, not a revision of one. Both are published as co-winners.

## Layout

```
solutions/
  level-1/    # MPS + canonical beam search (anonymous participant)
  level-2/    # + unswap bond-dimension reduction (anonymous participant)
  level-3a/   # τ-block excision + concurrent D-measurement (Charlie)
  level-3b/   # structured-MPO engine: Gram-SVD + bond routing + unswap (Alexey)
tools/
  relicense.py   # classifies custom vs vendored files; normalizes headers to AGPL-3.0
docs/
  RELICENSING.md # licensing policy, custom/third-party classification, verification record
LICENSE          # GNU AGPL-3.0 (full text)
NOTICE.md        # attribution + bundled third-party components and their licenses
```

Each `solutions/level-<n>/` contains the solver source as submitted. Vendored
third-party libraries are handled per `docs/RELICENSING.md`: third-party code
retains its own license and is **not** relicensed.

## Licensing

Custom components are © qBitTensor Labs and released under **AGPL-3.0** (see
[`LICENSE`](LICENSE)). Under the Enigma rules, winning submissions are AGPL-3.0 and
the IP in custom components is assigned to qBitTensor Labs; neither solution vendors
third-party code — dependencies are pip-installed from upstream at image-build
time and retain their own licenses, summarized in [`NOTICE.md`](NOTICE.md).

### Commercial licensing

**The AGPL-3.0 is a strong copyleft license, and it has real obligations.** You are
free to use, study, modify, and run this software — including commercially — *but*
if you convey the software or **make it available to users over a network** (the
"Affero" clause closes the SaaS loophole), you must release the **complete
corresponding source code of your entire application**, including your
modifications, under the AGPL-3.0 to those users.

#### ⚠️ Know the risk before you build on AGPL code

AGPL-3.0 is one of the most aggressive open-source licenses in existence, and
underestimating it is a costly mistake. If you incorporate this code into a
product, the copyleft can reach **your entire application**:

- **It can force you to open-source your own proprietary code.** Combine AGPL code
  with your product and distribute it — or merely run it as a **backend for a
  website, API, or SaaS** — and you can be obligated to publish *all* of your
  application's source, including code you intended to keep secret. There is no
  "internal use / we never shipped a binary" escape hatch; the network clause is
  the point of AGPL.
- **Many organizations ban AGPL outright.** It routinely **fails legal and security
  review, blocks partnerships, and derails acquisition and fundraising due
  diligence** — an AGPL dependency discovered late can tank a deal or force an
  expensive rip-and-replace.
- **Non-compliance is copyright infringement.** Getting the boundary wrong exposes
  you to injunctions and to being compelled to either **disclose your source or
  remove the software** — after you've already built on it.

In short: if you are building anything you intend to keep closed, sell, embed, or
offer as a service, **AGPL is a serious legal risk you should not take on without
advice.**

**A commercial license removes that risk entirely.** qBitTensor Labs holds the
copyright to the custom components (assigned under the Enigma rules), so we can
license the same code to you under proprietary-friendly terms that **lift the
copyleft and the network-source-disclosure requirements completely**. Under a
commercial license you can **embed, modify, and ship these components in
closed-source and SaaS products with no obligation to release your source** — the
safe, worry-free path for commercial use. This is the standard dual-licensing
model: **open source under AGPL-3.0, or a commercial license from us — your
choice.**

To make commercial use safe and remove all copyleft obligations on the components
we own, reach out to **support@qbittensorlabs.com**.

> **Scope.** A commercial license from qBitTensor Labs covers the **custom
> components** we own (assigned under the Enigma rules). It does **not** cover
> bundled third-party libraries (quantum simulation frameworks and their
> dependencies), which remain under their own licenses (see
> [`NOTICE.md`](NOTICE.md)); you are responsible for complying with those
> separately. Nothing here is legal advice; the governing terms are the
> [`LICENSE`](LICENSE) text and any commercial agreement you sign with us.

## Reproducing a solve

Each solution builds to a single Docker image and finds the peaked state of an
OpenQASM circuit at its target difficulty on one high-end GPU + CPU node
(validator parity: RTX PRO 6000 96 GB, 24 vCPU, 85 GB RAM, 4 h wall clock,
network-isolated). Per-solution build and run instructions live in each
`solutions/level-<n>/` directory.

## Credit

Levels 1 and 2 were submitted by anonymous competition participants; the Level 2
solver builds on the published Level 1 solution and credits it in its source. The
two Level 3 co-winners are credited to their competition handles — **Charlie**
(`level-3a`) and **Alexey** (`level-3b`). See [`NOTICE.md`](NOTICE.md) for the
public winning-key record and how these handles were established.
