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
import re
import tomllib
from pathlib import Path
from typing import Any

class ConfigError(ValueError):
    pass
FORBIDDEN_BLIND_KEYS = {'answer', 'expected_bitstring', 'k1', 'known_target', 'peaked_state', 'probe_bits', 'target', 'target_bits'}
_BITSTRING = re.compile('^[01]{32,}$')

def _walk(value: Any, path: tuple[str, ...]=()):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            yield (child_path, key, child)
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = (*path, str(index))
            yield (child_path, index, child)
            yield from _walk(child, child_path)

def assert_no_target_material(config: dict[str, Any]) -> None:
    violations: list[str] = []
    for path, key, value in _walk(config):
        if isinstance(key, str) and key.lower() in FORBIDDEN_BLIND_KEYS:
            violations.append('.'.join(path))
        if isinstance(value, str) and _BITSTRING.fullmatch(value):
            violations.append('.'.join(path) + ' (binary target-like value)')
    if violations:
        joined = ', '.join(sorted(set(violations)))
        raise ConfigError(f'blind plan contains target material: {joined}')

def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    required = {'schema_version', 'plan_id', 'instance_id', 'target_access', 'engine', 'execution'}
    missing = sorted(required - set(plan))
    if missing:
        raise ConfigError(f"missing required plan fields: {', '.join(missing)}")
    if plan['schema_version'] != '1.0':
        raise ConfigError('unsupported plan schema_version')
    if plan['instance_id'] not in {'s1', 's2', 'p9-control', 'local'}:
        raise ConfigError(f"unknown instance_id: {plan['instance_id']!r}")
    if plan['target_access'] not in {'blind', 'diagnostic_probe', 'control'}:
        raise ConfigError(f"invalid target_access: {plan['target_access']!r}")
    if plan['target_access'] == 'blind':
        assert_no_target_material(plan)
    return plan

def load_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    with plan_path.open('rb') as handle:
        plan = tomllib.load(handle)
    return validate_plan(plan)
