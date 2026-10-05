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
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple
LIVE_FRAME_ENCODING = 'live_l2s'
CHECKPOINT_SCHEMA = 'ckpt.v2'
_SHA256 = re.compile('^[0-9a-f]{64}$')

class ContractError(ValueError):
    pass

def validate_permutation(values: Sequence[int], n: int, field: str) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ContractError(field + ' must be an integer permutation')
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ContractError(field + ' must be an integer permutation') from exc
    if any((type(value) is not int for value in result)):
        raise ContractError(field + ' must contain integers')
    if len(result) != n or set(result) != set(range(n)):
        raise ContractError(field + ' is not a permutation of the sites')
    return result

@dataclass(frozen=True)
class BoundaryFrames:
    input_l2s: Tuple[int, ...]
    output_l2s: Tuple[int, ...]
    encoding: str = LIVE_FRAME_ENCODING

    def __post_init__(self) -> None:
        if self.encoding != LIVE_FRAME_ENCODING:
            raise ContractError('unsupported frame encoding')
        input_map = validate_permutation(self.input_l2s, len(self.input_l2s), 'input_l2s')
        output_map = validate_permutation(self.output_l2s, len(input_map), 'output_l2s')
        object.__setattr__(self, 'input_l2s', input_map)
        object.__setattr__(self, 'output_l2s', output_map)

    @property
    def n(self) -> int:
        return len(self.input_l2s)

    def to_dict(self) -> dict:
        return {'encoding': self.encoding, 'input_l2s': list(self.input_l2s), 'output_l2s': list(self.output_l2s)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'BoundaryFrames':
        if not isinstance(value, Mapping):
            raise ContractError('frames must be an object')
        required = {'encoding', 'input_l2s', 'output_l2s'}
        if set(value) != required:
            raise ContractError('frames have missing or unknown fields')
        return cls(tuple(value['input_l2s']), tuple(value['output_l2s']), value['encoding'])

@dataclass(frozen=True)
class SourceGate:
    raw_id: int
    name: str
    params: Tuple[float, ...]
    qubits: Tuple[int, ...]
    source_text: str

    def __post_init__(self) -> None:
        if type(self.raw_id) is not int or self.raw_id < 0:
            raise ContractError('raw_id must be a nonnegative integer')
        if not isinstance(self.name, str) or not re.fullmatch('[a-z][a-z0-9_]*', self.name) or (not 1 <= len(self.qubits) <= 2) or (len(set(self.qubits)) != len(self.qubits)) or any((type(q) is not int or q < 0 for q in self.qubits)) or any((type(value) not in {int, float} or not math.isfinite(value) for value in self.params)):
            raise ContractError('gate name and qubits are required')

@dataclass(frozen=True)
class WorkLedger:
    source_sha256: str
    total_raw_gates: int
    consumed_raw_ids: Tuple[int, ...]
    remaining_raw_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_sha256, str) or not _SHA256.fullmatch(self.source_sha256):
            raise ContractError('source_sha256 is invalid')
        if type(self.total_raw_gates) is not int or self.total_raw_gates < 0:
            raise ContractError('total_raw_gates is invalid')
        consumed = tuple(self.consumed_raw_ids)
        remaining = tuple(self.remaining_raw_ids)
        if any((type(x) is not int for x in consumed + remaining)):
            raise ContractError('raw gate identifiers must be integers')
        if len(set(consumed)) != len(consumed) or len(set(remaining)) != len(remaining):
            raise ContractError('raw gate identifiers contain duplicates')
        if set(consumed) & set(remaining):
            raise ContractError('consumed and remaining raw gates overlap')
        if set(consumed) | set(remaining) != set(range(self.total_raw_gates)):
            raise ContractError('raw gate ownership is incomplete')

    @property
    def complete(self) -> bool:
        return not self.remaining_raw_ids and len(self.consumed_raw_ids) == self.total_raw_gates

    def to_dict(self) -> dict:
        return {'source_sha256': self.source_sha256, 'total_raw_gates': self.total_raw_gates, 'consumed_raw_ids': list(self.consumed_raw_ids), 'remaining_raw_ids': list(self.remaining_raw_ids)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'WorkLedger':
        if not isinstance(value, Mapping):
            raise ContractError('work ledger must be an object')
        required = {'source_sha256', 'total_raw_gates', 'consumed_raw_ids', 'remaining_raw_ids'}
        if set(value) != required:
            raise ContractError('work ledger has missing or unknown fields')
        return cls(value['source_sha256'], value['total_raw_gates'], tuple(value['consumed_raw_ids']), tuple(value['remaining_raw_ids']))
