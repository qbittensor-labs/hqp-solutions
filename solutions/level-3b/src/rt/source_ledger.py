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
import ast
import hashlib
import math
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple
from .contracts import ContractError, SourceGate, WorkLedger, validate_permutation
S2_QASM_SHA256 = 'c09ba53721b60791bf888aaa3c161c5a2b7e310c30dfbcfdc79c26b475278354'
S2_SIGMA0_QASM_SHA256 = '4f39187587e370d3d199e368dc25ce7d4548f169af49eace6fd99df4dfc176a4'
S2_CANONICAL_TO_SIGMA0_SHA256 = '14870da89daf718943a5577dd2c990309291d121d0b27d8105f28d66e237ce30'
S2_CANONICAL_TO_SIGMA0 = (28, 46, 0, 44, 36, 6, 2, 4, 10, 7, 5, 26, 12, 47, 24, 32, 14, 22, 23, 37, 42, 1, 16, 18, 40, 11, 29, 8, 15, 30, 9, 13, 41, 34, 20, 3, 19, 25, 17, 43, 21, 31, 35, 33, 38, 27, 39, 45)
S2_REGIONS = (('prefix', 0, 785), ('m0', 785, 2184), ('gap', 2184, 2405), ('m1', 2405, 3816), ('suffix', 3816, 4353))
_QREG = re.compile('^qreg\\s+q\\[(\\d+)\\];$')
_GATE = re.compile('^([A-Za-z_][A-Za-z0-9_]*)(?:\\(([^)]*)\\))?\\s+q\\[(\\d+)\\](?:\\s*,\\s*q\\[(\\d+)\\])?;$')

def parse_angle(expression: str) -> float:
    try:
        root = ast.parse(expression.strip(), mode='eval')
    except (SyntaxError, ValueError) as exc:
        raise ContractError('invalid gate parameter') from exc

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == 'pi':
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = (visit(node.left), visit(node.right))
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ContractError('division by zero in gate parameter')
            return left / right
        raise ContractError('unsupported gate parameter expression')
    value = visit(root)
    if not math.isfinite(value):
        raise ContractError('non-finite gate parameter')
    return value

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def parse_qasm2(path: Path, expected_sha256: Optional[str]=None) -> Tuple[int, Tuple[SourceGate, ...]]:
    path = Path(path)
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ContractError('source QASM SHA-256 mismatch')
    n = None
    gates = []
    for line_number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = raw.split('//', 1)[0].strip()
        if not line or line.startswith('OPENQASM') or line.startswith('include'):
            continue
        qreg = _QREG.fullmatch(line)
        if qreg:
            if n is not None:
                raise ContractError('multiple qreg declarations')
            n = int(qreg.group(1))
            continue
        if line.startswith(('creg ', 'barrier ', 'measure ')):
            raise ContractError('classical or non-unitary instruction is unsupported')
        match = _GATE.fullmatch(line)
        if not match:
            raise ContractError('unsupported QASM at line {}'.format(line_number))
        name, raw_params, first, second = match.groups()
        params = tuple((parse_angle(item) for item in raw_params.split(','))) if raw_params else ()
        qubits = (int(first),) if second is None else (int(first), int(second))
        gates.append(SourceGate(len(gates), name.lower(), params, qubits, line))
    if n is None:
        raise ContractError('missing qreg declaration')
    if any((q >= n for gate in gates for q in gate.qubits)):
        raise ContractError('gate qubit is outside qreg')
    return (n, tuple(gates))

def validate_regions(total_gates: int, regions: Sequence[tuple]=S2_REGIONS) -> None:
    cursor = 0
    names = set()
    for name, start, stop in regions:
        if name in names or start != cursor or stop <= start:
            raise ContractError('source regions do not tile in order')
        names.add(name)
        cursor = stop
    if cursor != total_gates:
        raise ContractError('source regions do not cover all raw gates')

def ledger_from_consumed(source_sha256: str, total: int, consumed: Iterable[int]) -> WorkLedger:
    consumed_tuple = tuple(consumed)
    consumed_set = set(consumed_tuple)
    remaining = tuple((index for index in range(total) if index not in consumed_set))
    return WorkLedger(source_sha256, total, consumed_tuple, remaining)

def assert_ordered_subset(gate_ids: Sequence[int], allowed: Sequence[int], direction: str) -> None:
    if direction not in {'ascending', 'descending'}:
        raise ContractError('unknown work direction')
    values = tuple(gate_ids)
    expected = tuple(sorted(values, reverse=direction == 'descending'))
    if values != expected or len(set(values)) != len(values) or (not set(values) <= set(allowed)):
        raise ContractError('work window has invalid source order or ownership')

def validate_descending_interval_partition(start: int, stop: int, execution_intervals: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    if type(start) is not int or type(stop) is not int or start < 0 or (stop <= start) or (not execution_intervals):
        raise ContractError('descending interval partition bounds are invalid')
    intervals = tuple(execution_intervals)
    cursor = stop
    consumed = []
    for interval in intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2 or type(interval[0]) is not int or (type(interval[1]) is not int):
            raise ContractError('descending interval partition entry is invalid')
        left, right = interval
        if left < start or right != cursor or left >= right:
            raise ContractError('descending interval partition is gapped or overlapping')
        consumed.extend(range(left, right))
        cursor = left
    if cursor != start or len(consumed) != stop - start or set(consumed) != set(range(start, stop)):
        raise ContractError('descending interval partition does not tile the source interval')
    return tuple(consumed)

def assert_relabelled_source(canonical: Sequence[SourceGate], working: Sequence[SourceGate], canonical_to_working: Sequence[int]) -> None:
    mapping = validate_permutation(canonical_to_working, len(canonical_to_working), 'canonical_to_working')
    if len(canonical) != len(working):
        raise ContractError('canonical and working source lengths differ')
    for expected_id, (source, observed) in enumerate(zip(canonical, working)):
        expected = (source.name, source.params, tuple((mapping[q] for q in source.qubits)))
        actual = (observed.name, observed.params, observed.qubits)
        if source.raw_id != expected_id or observed.raw_id != expected_id or actual != expected:
            raise ContractError('working source is not the exact declared relabel')
