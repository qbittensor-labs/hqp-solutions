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

from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from qiskit import qasm2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pc as C

def _f0(gates, i, j):
    pair = set(gates[i].wires)
    deltas = [abs(g.angles[0] - int(np.rint(g.angles[0] / math.pi)) * math.pi) for g in gates[i + 1:j] if g.name == 'u' and g.wires[0] in pair]
    return max(deltas) if deltas else 0.0

def _f1(gates, threshold):
    current, records = (list(gates), [])
    while True:
        options = sorted(((_f0(current, i, j), cost, i, j) for cost, i, j in C._f4(current)))
        options = [o for o in options if o[0] <= threshold]
        if not options:
            break
        current, record = C._f5(current, options[0][2], options[0][3])
        records.append(record)
    return (current, records)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threshold', type=float, default=0.1)
    p.add_argument('--allow-derived-source', action='store_true')
    args = p.parse_args()
    payload = Path(args.source).read_bytes()
    source_sha = hashlib.sha256(payload).hexdigest()
    if source_sha != C.SOURCE_SHA and (not args.allow_derived_source):
        raise ValueError('Authoritative QASM identity mismatch')
    qc_in = qasm2.loads(payload.decode(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    from qiskit import QuantumCircuit as _QC
    _keep = _QC(qc_in.num_qubits)
    for _it in qc_in.data:
        if _it.operation.name in ('measure', 'barrier'):
            continue
        _keep.append(_it.operation, [qc_in.find_bit(q).index for q in _it.qubits])
    qc_in = _keep
    n_qubits = qc_in.num_qubits
    source = C._f1(qc_in)
    result, records = _f1(source, args.threshold)
    checks = C._f7(source, result, records)
    _out = _QC(n_qubits)
    for g in result:
        if g.name == 'u':
            _out.u(*g.angles, g.wires[0])
        elif g.name == 'z':
            _out.u(0.0, 0.0, math.pi, g.wires[0])
        else:
            _out.cz(*g.wires)
    text = qasm2.dumps(_out) + '\n'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'reduced.qasm').write_text(text)
    (out / 'certificate.json').write_text(json.dumps(records, indent=2) + '\n')
    moved = [abs(c['before'][0] - c['after'][0]) for r in records for c in r['changes']]
    summary = {'source_sha256': source_sha, 'is_authoritative_source': source_sha == C.SOURCE_SHA, 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'threshold_rad': args.threshold, 'couples_cancelled': len(records), **checks, 'max_angle_displacement': max(moved) if moved else 0.0, 'median_angle_displacement': float(np.median(moved)) if moved else 0.0, 'incoherent_rss_error_estimate': float(np.sqrt(np.sum((np.array(moved) / 2) ** 2))), 'reduced_qasm_sha256': hashlib.sha256(text.encode()).hexdigest(), 'note': 'operator_error_bound is a coherent triangle-inequality sum against the OBFUSCATED circuit and is expected to be vacuous here; judge by peak retention (docs/31), not by this bound'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
