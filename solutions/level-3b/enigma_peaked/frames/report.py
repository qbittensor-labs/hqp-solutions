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
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
from qiskit import QuantumCircuit
from ..structure.gadgets import collect_structure
from .schedule import FrameSchedule, FrameScheduleError
from .structural import StructuralConfig, verify_module
FRAME_WARNING = 'Exact frame schedule only: every gate is preserved and the represented unitary is unchanged. This report never authorizes heuristic reduction or gate deletion.'

def _instance_id_for_hash(qasm_sha256: str) -> str | None:
    try:
        from ..evidence import find_project_root
        manifest_path = find_project_root() / 'instances' / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except Exception:
        return None
    for instance in manifest.get('instances', ()):
        if instance.get('qasm_sha256') == qasm_sha256:
            return instance.get('id')
    return None

def build_frame_report(qasm_path: str | Path, gadget_report: dict[str, Any], config: StructuralConfig) -> dict[str, Any]:
    qasm_path = Path(qasm_path)
    qasm_sha256 = hashlib.sha256(qasm_path.read_bytes()).hexdigest()
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    if gadget_report.get('num_qubits') != circuit.num_qubits:
        raise FrameScheduleError(f"gadget report is for {gadget_report.get('num_qubits')} qubits, circuit has {circuit.num_qubits}")
    instance_id = _instance_id_for_hash(qasm_sha256)
    structure = collect_structure(circuit)
    modules = [verify_module(circuit, hypothesis, config, qasm_sha256=qasm_sha256, instance_id=instance_id, structure=structure) for hypothesis in gadget_report.get('modules', ())]
    schedule = None
    if any((module.eligible for module in modules)):
        schedule = FrameSchedule.from_modules(modules, circuit.num_qubits)
    return {'qasm': str(qasm_path), 'qasm_sha256': qasm_sha256, 'instance_id': instance_id, 'num_qubits': circuit.num_qubits, 'modules': [module.to_dict() for module in modules], 'eligible_module_count': sum((module.eligible for module in modules)), 'schedule': schedule.to_dict() if schedule is not None else None, 'warning': FRAME_WARNING}

def _summary(report: dict[str, Any]) -> str:
    lines = [f"qasm={report['qasm']} sha256={report['qasm_sha256'][:16]}... instance={report['instance_id']} qubits={report['num_qubits']}"]
    for module in report['modules']:
        evidence = module['evidence']
        status = 'FRAME-ELIGIBLE' if module['eligible'] else 'REJECTED'
        lines.append(f"module {module['module_id']}: {status} support={evidence['support_fraction']:.4f} ({evidence['matched_pairs']}/{evidence['module_pair_count']} pairs, {evidence['unpaired_oneq_gates']} unpaired 1q) cz_timed={evidence['cz_timed_rate']:.4f} null_max={evidence['cz_null_max']} span=[{module['enter_layer']},{module['exit_layer']})")
        for reason in module['reasons']:
            lines.append(f'  reason: {reason}')
    lines.append('schedule: ' + ('none (no eligible modules)' if report['schedule'] is None else 'emitted'))
    lines.append('WARNING: ' + report['warning'])
    return '\n'.join(lines)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='enigma-peaked frame-report')
    parser.add_argument('qasm')
    parser.add_argument('--gadgets', required=True)
    parser.add_argument('--json-out')
    parser.add_argument('--inverse-tol', type=float, default=1e-11)
    parser.add_argument('--min-support', type=float, default=0.95)
    parser.add_argument('--cz-layer-tolerance', type=float, default=4.0)
    parser.add_argument('--null-trials', type=int, default=1000)
    parser.add_argument('--null-seed', type=int, default=7)
    return parser

def main(argv: Sequence[str] | None=None) -> int:
    args = _build_parser().parse_args(argv)
    config = StructuralConfig(inverse_tol=args.inverse_tol, min_support=args.min_support, cz_layer_tolerance=args.cz_layer_tolerance, null_trials=args.null_trials, null_seed=args.null_seed)
    gadget_report = json.loads(Path(args.gadgets).read_text(encoding='utf-8'))
    report = build_frame_report(args.qasm, gadget_report, config)
    print(_summary(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'wrote frame report JSON: {args.json_out}')
    return 0
