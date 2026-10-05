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
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from jsonschema import Draft202012Validator, FormatChecker
from .config import assert_no_target_material

class EvidenceError(ValueError):
    pass

def find_project_root(start: str | Path | None=None) -> Path:
    candidates = [Path(start or Path.cwd()).resolve(), Path(__file__).resolve()]
    for candidate in candidates:
        for directory in (candidate, *candidate.parents):
            if (directory / 'schemas').is_dir() and (directory / 'evidence').is_dir():
                return directory
    raise EvidenceError('could not locate project root containing schemas/ and evidence/')

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise EvidenceError(f'{path}:{line_number}: invalid JSON: {exc}') from exc
    return records

def _validate_records(records: Iterable[dict[str, Any]], schema: dict[str, Any], label: str) -> None:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for index, record in enumerate(records, 1):
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        if errors:
            detail = '; '.join((f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors))
            raise EvidenceError(f'{label} record {index}: {detail}')

def _unique_by_id(records: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = record['id']
        if record_id in result:
            raise EvidenceError(f'duplicate {label} id: {record_id}')
        result[record_id] = record
    return result

def validate_evidence(root: str | Path | None=None) -> dict[str, int]:
    project = find_project_root(root)
    artifact_records = _read_jsonl(project / 'evidence' / 'legacy-artifacts.jsonl')
    experiment_records = _read_jsonl(project / 'evidence' / 'ledger.jsonl')
    claim_records = _read_jsonl(project / 'evidence' / 'claims.jsonl')
    _validate_records(artifact_records, _read_json(project / 'schemas' / 'artifact-v1.schema.json'), 'artifact')
    _validate_records(experiment_records, _read_json(project / 'schemas' / 'experiment-v1.schema.json'), 'experiment')
    _validate_records(claim_records, _read_json(project / 'schemas' / 'claim-v1.schema.json'), 'claim')
    artifacts = _unique_by_id(artifact_records, 'artifact')
    experiments = _unique_by_id(experiment_records, 'experiment')
    claims = _unique_by_id(claim_records, 'claim')
    evidence_ids = set(artifacts) | set(experiments)
    for artifact in artifact_records:
        if artifact['preservation'] != 'in_repository':
            continue
        local = project / artifact['source']['path']
        if not local.is_file():
            raise EvidenceError(f"{artifact['id']}: in-repository file missing: {local}")
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if digest != artifact['source']['sha256']:
            raise EvidenceError(f"{artifact['id']}: file hash {digest} does not match indexed sha256 {artifact['source']['sha256']}")
    for experiment in experiment_records:
        target_access = experiment['instance']['target_access']
        trial_ids = {trial['trial_id'] for trial in experiment['trials']}
        for artifact_id in experiment['artifacts']:
            if artifact_id not in artifacts:
                raise EvidenceError(f"{experiment['id']}: unknown artifact {artifact_id}")
        for observation in experiment['observations']:
            if observation['artifact_ref'] not in artifacts:
                raise EvidenceError(f"{experiment['id']}: observation references unknown artifact {observation['artifact_ref']}")
            trial_id = observation.get('trial_id')
            if trial_id is not None and trial_id not in trial_ids:
                raise EvidenceError(f"{experiment['id']}: unknown trial_id {trial_id}")
            if target_access == 'blind' and observation['target_dependent']:
                raise EvidenceError(f"{experiment['id']}: blind experiment contains target-dependent observation")
        if target_access == 'blind':
            assert_no_target_material(experiment['protocol']['config'])
    for claim in claim_records:
        refs = claim['supporting_evidence'] + claim['contrary_evidence']
        missing = sorted(set(refs) - evidence_ids)
        if missing:
            raise EvidenceError(f"{claim['id']}: unresolved evidence refs: {', '.join(missing)}")
        if claim['status'] in {'supported', 'partially_supported'} and (not claim['supporting_evidence']):
            raise EvidenceError(f"{claim['id']}: status requires supporting evidence")
        if claim['status'] == 'contradicted' and (not claim['contrary_evidence']):
            raise EvidenceError(f"{claim['id']}: contradicted claim requires contrary evidence")
        supersedes = claim.get('supersedes')
        if supersedes is not None and supersedes not in claims:
            raise EvidenceError(f"{claim['id']}: supersedes unknown claim {supersedes}")
    return {'artifacts': len(artifact_records), 'experiments': len(experiment_records), 'claims': len(claim_records)}
