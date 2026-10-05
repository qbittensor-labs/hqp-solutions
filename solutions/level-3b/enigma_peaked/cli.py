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
import json
import sys
from pathlib import Path
from . import __version__
from .config import ConfigError, load_plan
from .evidence import EvidenceError, validate_evidence

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='enigma-peaked')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate-evidence')
    validate.add_argument('--root', type=Path, default=None)
    config = sub.add_parser('validate-config')
    config.add_argument('path', type=Path)
    sub.add_parser('gadget-report', add_help=False)
    sub.add_parser('pauli-marginal', add_help=False)
    sub.add_parser('frame-report', add_help=False)
    evaluate = sub.add_parser('evaluate-candidates')
    evaluate.add_argument('candidates', type=Path)
    evaluate.add_argument('--target-file', type=Path, default=None)
    evaluate.add_argument('--json-out', type=Path, default=None)
    guard = sub.add_parser('guard')
    guard.add_argument('--root', type=Path, default=None)
    run = sub.add_parser('run-generator')
    run.add_argument('plan', type=Path)
    run.add_argument('--qasm', type=Path, default=None)
    run.add_argument('--out-root', type=Path, default=None)
    resume = sub.add_parser('resume-generator')
    resume.add_argument('checkpoint', type=Path)
    resume.add_argument('--plan', type=Path, required=True)
    resume.add_argument('--qasm', type=Path, default=None)
    resume.add_argument('--out-root', type=Path, default=None)
    verify = sub.add_parser('verify-candidates')
    verify.add_argument('plan', type=Path)
    verify.add_argument('candidates', type=Path)
    verify.add_argument('--qasm', type=Path, default=None)
    verify.add_argument('--json-out', type=Path, default=None)
    sub.add_parser('engine-status')
    return parser

def main(argv: list[str] | None=None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if raw_args and raw_args[0] == 'gadget-report':
        from .structure.gadgets import main as gadget_main
        return gadget_main(raw_args[1:])
    if raw_args and raw_args[0] == 'pauli-marginal':
        from .structure.pauli_marginal import main as pauli_main
        return pauli_main(raw_args[1:])
    if raw_args and raw_args[0] == 'frame-report':
        from .frames.report import main as frame_main
        return frame_main(raw_args[1:])
    args = _parser().parse_args(raw_args)
    try:
        if args.command == 'validate-evidence':
            counts = validate_evidence(args.root)
            print(json.dumps({'status': 'valid', **counts}, sort_keys=True))
            return 0
        if args.command == 'validate-config':
            plan = load_plan(args.path)
            print(json.dumps({'status': 'valid', 'plan_id': plan['plan_id'], 'instance_id': plan['instance_id'], 'target_access': plan['target_access']}, sort_keys=True))
            return 0
        if args.command == 'evaluate-candidates':
            from .posthoc import PosthocError, run as posthoc_run
            try:
                result = posthoc_run(args.candidates, args.target_file, args.json_out)
            except PosthocError as exc:
                print(f'error: {exc}', file=sys.stderr)
                return 2
            print(json.dumps(result, indent=2))
            return 0
        if args.command == 'guard':
            from .guard import GuardError, run_guard
            try:
                result = run_guard(args.root)
            except GuardError as exc:
                print(f'error: {exc}', file=sys.stderr)
                return 2
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result['status'] == 'clean' else 1
        if args.command == 'verify-candidates':
            from .engine.generator import GeneratorError
            from .verifier.verify import VerifierError, run_verification
            try:
                report = run_verification(args.plan, args.candidates, qasm_override=args.qasm)
            except (VerifierError, GeneratorError, ConfigError) as exc:
                print(f'error: {exc}', file=sys.stderr)
                return 2
            if args.json_out:
                args.json_out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({'status': report['termination'], 'submitted_candidates': report['submitted_candidates'], 'scored': len(report['scored']), 'censored': len(report['censored']), 'top': report['scored'][:3]}, indent=2))
            return 0 if report['termination'] == 'completed' else 1
        if args.command in {'run-generator', 'resume-generator'}:
            from .engine.checkpoint import CheckpointError
            from .engine.generator import GeneratorError, resume_generator, run_generator
            try:
                if args.command == 'run-generator':
                    summary = run_generator(args.plan, qasm_override=args.qasm, out_root=args.out_root)
                else:
                    summary = resume_generator(args.checkpoint, plan_path=args.plan, qasm_override=args.qasm, out_root=args.out_root)
            except (GeneratorError, CheckpointError, ConfigError) as exc:
                print(f'error: {exc}', file=sys.stderr)
                return 2
            print(json.dumps({'status': 'completed', 'run_dir': summary['run_dir'], 'orderings': [{'seed': ordering['seed'], 'termination': ordering['counters']['termination'], 'censored': ordering['counters']['censored'], 'final_bond': ordering['final_bond'], 'top1_logical_bits': ordering['candidates'][0]['logical_bits'] if ordering['candidates'] else None} for ordering in summary['orderings']]}, indent=2))
            return 0
        if args.command == 'engine-status':
            print(json.dumps({'status': 'partial', 'phase': 2, 'active': ['localized-permutation-gadget diagnostic', 'Aer marginal probe', 'Pauli-path marginal solver', 'module-local frame semantics', 'MPO-unswap generator', 'two-sided rescorer', 'blind decoders', 'post-hoc target evaluator', 'repository target guard'], 'deferred': []}, sort_keys=True))
            return 0
    except (ConfigError, EvidenceError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    return 2
if __name__ == '__main__':
    raise SystemExit(main())
