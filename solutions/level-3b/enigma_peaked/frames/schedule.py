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
from dataclasses import dataclass, field
from typing import Any
from .permutation import Permutation
_SHA256 = re.compile('^[0-9a-f]{64}$')

class FrameScheduleError(ValueError):
    pass

@dataclass(frozen=True)
class ModuleFrame:
    instance_id: str | None
    qasm_sha256: str
    module_id: int
    num_qubits: int
    enter_layer: int
    exit_layer: int
    permutation: Permutation
    evidence: dict[str, Any] = field(default_factory=dict)
    eligible: bool = False
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.qasm_sha256):
            raise FrameScheduleError(f'module {self.module_id}: qasm_sha256 must be 64 lowercase hex characters')
        if len(self.permutation) != self.num_qubits:
            raise FrameScheduleError(f'module {self.module_id}: permutation size {len(self.permutation)} does not match num_qubits {self.num_qubits}')
        if self.enter_layer < 0 or self.exit_layer <= self.enter_layer:
            raise FrameScheduleError(f'module {self.module_id}: invalid span [{self.enter_layer}, {self.exit_layer})')

    def to_dict(self) -> dict[str, Any]:
        return {'instance_id': self.instance_id, 'qasm_sha256': self.qasm_sha256, 'module_id': self.module_id, 'num_qubits': self.num_qubits, 'enter_layer': self.enter_layer, 'exit_layer': self.exit_layer, 'permutation': list(self.permutation), 'evidence': self.evidence, 'eligible': self.eligible, 'reasons': list(self.reasons)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModuleFrame:
        return cls(instance_id=payload.get('instance_id'), qasm_sha256=payload['qasm_sha256'], module_id=int(payload['module_id']), num_qubits=int(payload['num_qubits']), enter_layer=int(payload['enter_layer']), exit_layer=int(payload['exit_layer']), permutation=Permutation(payload['permutation']), evidence=dict(payload.get('evidence', {})), eligible=bool(payload.get('eligible', False)), reasons=tuple(payload.get('reasons', ())))

@dataclass(frozen=True)
class FrameEvent:
    layer: int
    kind: str
    module_id: int
    permutation: Permutation

    def __post_init__(self) -> None:
        if self.kind not in {'enter', 'exit'}:
            raise FrameScheduleError(f'unknown frame event kind {self.kind!r}')

    def seam_permutation(self) -> Permutation:
        return self.permutation if self.kind == 'enter' else self.permutation.inverse()

class FrameSchedule:

    def __init__(self, num_qubits: int, events: list[FrameEvent], qasm_sha256: str) -> None:
        if num_qubits <= 0:
            raise FrameScheduleError('num_qubits must be positive')
        if not _SHA256.fullmatch(qasm_sha256):
            raise FrameScheduleError('schedule qasm_sha256 must be 64 lowercase hex characters')
        self.num_qubits = num_qubits
        self.qasm_sha256 = qasm_sha256
        self.events = sorted(events, key=lambda event: (event.layer, event.kind == 'enter'))
        self._validate()

    def _validate(self) -> None:
        open_module: int | None = None
        open_permutation: Permutation | None = None
        for event in self.events:
            if len(event.permutation) != self.num_qubits:
                raise FrameScheduleError(f'event for module {event.module_id} has a permutation of size {len(event.permutation)}, expected {self.num_qubits}')
            if event.kind == 'enter':
                if open_module is not None:
                    raise FrameScheduleError(f'module {event.module_id} enters at layer {event.layer} while module {open_module} is still open; spans must not overlap')
                open_module = event.module_id
                open_permutation = event.permutation
            else:
                if open_module != event.module_id:
                    raise FrameScheduleError(f'exit event for module {event.module_id} does not match the open module {open_module}')
                if event.permutation != open_permutation:
                    raise FrameScheduleError(f'module {event.module_id} exit permutation differs from its entry')
                open_module = None
                open_permutation = None
        if open_module is not None:
            raise FrameScheduleError(f'module {open_module} has no exit event')

    @classmethod
    def identity(cls, num_qubits: int, qasm_sha256: str) -> FrameSchedule:
        return cls(num_qubits, [], qasm_sha256)

    @classmethod
    def from_modules(cls, modules: list[ModuleFrame], num_qubits: int) -> FrameSchedule:
        eligible = [module for module in modules if module.eligible]
        if not eligible:
            raise FrameScheduleError('no eligible modules to schedule')
        hashes = {module.qasm_sha256 for module in eligible}
        if len(hashes) != 1:
            raise FrameScheduleError('modules disagree on the circuit qasm_sha256')
        for module in eligible:
            if module.num_qubits != num_qubits:
                raise FrameScheduleError(f'module {module.module_id} qubit count {module.num_qubits} does not match circuit {num_qubits}')
        events: list[FrameEvent] = []
        for module in eligible:
            events.append(FrameEvent(module.enter_layer, 'enter', module.module_id, module.permutation))
            events.append(FrameEvent(module.exit_layer, 'exit', module.module_id, module.permutation))
        return cls(num_qubits, events, hashes.pop())

    def is_identity(self) -> bool:
        return not self.events

    def events_at(self, layer: int) -> list[FrameEvent]:
        return [event for event in self.events if event.layer == layer]

    def max_layer(self) -> int:
        return max((event.layer for event in self.events), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {'num_qubits': self.num_qubits, 'qasm_sha256': self.qasm_sha256, 'events': [{'layer': event.layer, 'kind': event.kind, 'module_id': event.module_id, 'permutation': list(event.permutation)} for event in self.events]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrameSchedule:
        events = [FrameEvent(layer=int(entry['layer']), kind=str(entry['kind']), module_id=int(entry['module_id']), permutation=Permutation(entry['permutation'])) for entry in payload.get('events', [])]
        return cls(int(payload['num_qubits']), events, payload['qasm_sha256'])
