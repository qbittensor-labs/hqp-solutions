#!/usr/bin/env python3
# Copyright (C) 2026 qBitTensor Labs.
# Original author: Alexey (Enigma / Hardening Quantum Proof competition).
# IP in custom components assigned to qBitTensor Labs under the Enigma rules.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details. You should have received a copy of the license with this program;
# if not, see <https://www.gnu.org/licenses/>.

"""Identify and undo CZ pairs that were inserted into d3_s2 and hidden by detuning.

Hypothesis under test: the obfuscator added cancellable CZ pairs to a base circuit and then
slightly detuned the few 1q gates caught between each pair, so the cancellation is not visible.

Evidence that it holds (see docs/33):
  * worst-blocker-displacement per couple has an EMPTY BAND in (0.10, 0.50) rad -- 45 couples
    below, 599 above, nothing between, across a band 4x wider than the populated one.
  * 45 observed below 0.10 vs 9.0 +- 2.6 for a null that shuffles the same detunings onto the
    same blocker structure: 13.9 sigma.
  * median partner separation is 12 instructions (min 3) against a 762-instruction median for
    all couples -- inserted adjacent, not spread.

The undo uses only deobfuscate_s2_certified.cancel_pair, so it modifies existing 1q angles and
emits free diagonal z byproducts; it never synthesizes new 1q gates. Selection is by worst
angle displacement, NOT by the accumulated operator-norm bound: that bound is measured against
the obfuscated circuit, which is not the reference whose peak we care about.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path

import numpy as np
from qiskit import qasm2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deobfuscate_s2_certified as C


def worst_displacement(gates, i, j):
    """Largest angle a blocker would have to move to let this couple annihilate."""
    pair = set(gates[i].wires)
    deltas = [abs(g.angles[0] - int(np.rint(g.angles[0] / math.pi)) * math.pi)
              for g in gates[i + 1:j] if g.name == "u" and g.wires[0] in pair]
    return max(deltas) if deltas else 0.0


def undo(gates, threshold):
    """Greedily cancel every couple whose every blocker moves less than `threshold`."""
    current, records = list(gates), []
    while True:
        options = sorted((worst_displacement(current, i, j), cost, i, j)
                         for cost, i, j in C.candidates(current))
        options = [o for o in options if o[0] <= threshold]
        if not options:
            break
        current, record = C.cancel_pair(current, options[0][2], options[0][3])
        records.append(record)
    return current, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.10,
                   help="max |delta theta| any single blocker may be moved (rad)")
    p.add_argument("--allow-derived-source", action="store_true",
                   help="accept a circuit that is not the authoritative d3_s2 QASM, so the undo "
                        "can be stacked on an already-reduced circuit; the input sha is recorded")
    args = p.parse_args()

    payload = Path(args.source).read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()
    if source_sha != C.SOURCE_SHA and not args.allow_derived_source:
        raise ValueError("Authoritative QASM identity mismatch")
    source = C.parse(qasm2.loads(payload.decode(),
                                 custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))
    result, records = undo(source, args.threshold)
    checks = C.audit(source, result, records)          # their audit: matrices, ownership, CZ count

    text = qasm2.dumps(C.circuit_of(result, 48)) + "\n"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reduced.qasm").write_text(text)
    (out / "certificate.json").write_text(json.dumps(records, indent=2) + "\n")

    moved = [abs(c["before"][0] - c["after"][0]) for r in records for c in r["changes"]]
    summary = {
        "source_sha256": source_sha,
        "is_authoritative_source": source_sha == C.SOURCE_SHA,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "threshold_rad": args.threshold,
        "couples_cancelled": len(records),
        **checks,
        "max_angle_displacement": max(moved) if moved else 0.0,
        "median_angle_displacement": float(np.median(moved)) if moved else 0.0,
        "incoherent_rss_error_estimate": float(np.sqrt(np.sum((np.array(moved) / 2) ** 2))),
        "reduced_qasm_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "note": ("operator_error_bound is a coherent triangle-inequality sum against the "
                 "OBFUSCATED circuit and is expected to be vacuous here; judge by peak "
                 "retention (docs/31), not by this bound"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
