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
from typing import Sequence
from .permutation import Permutation, PermutationError

class FrameStateError(ValueError):
    pass

class FrameState:
    __slots__ = ('_logical_to_site', '_site_to_logical')

    def __init__(self, logical_to_site: Sequence[int]) -> None:
        try:
            permutation = Permutation(logical_to_site)
        except PermutationError as exc:
            raise FrameStateError(f'invalid logical_to_site map: {exc}') from exc
        self._logical_to_site = list(permutation.mapping)
        self._site_to_logical = list(permutation.inverse().mapping)
        self._check()

    @classmethod
    def identity(cls, num_qubits: int) -> FrameState:
        return cls(range(num_qubits))

    def _check(self) -> None:
        for logical, site in enumerate(self._logical_to_site):
            if self._site_to_logical[site] != logical:
                raise FrameStateError(f'site_to_logical is not the inverse of logical_to_site at qubit {logical}')

    @property
    def num_qubits(self) -> int:
        return len(self._logical_to_site)

    @property
    def logical_to_site(self) -> tuple[int, ...]:
        return tuple(self._logical_to_site)

    @property
    def site_to_logical(self) -> tuple[int, ...]:
        return tuple(self._site_to_logical)

    def site_of(self, logical: int) -> int:
        return self._logical_to_site[logical]

    def logical_of(self, site: int) -> int:
        return self._site_to_logical[site]

    def copy(self) -> FrameState:
        return FrameState(self._logical_to_site)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameState):
            return NotImplemented
        return self._logical_to_site == other._logical_to_site

    def __repr__(self) -> str:
        return f'FrameState(logical_to_site={self._logical_to_site})'

    def is_identity(self) -> bool:
        return all((site == logical for logical, site in enumerate(self._logical_to_site)))

    def enter_seam(self, sigma: Permutation) -> None:
        if len(sigma) != self.num_qubits:
            raise FrameStateError(f'seam permutation acts on {len(sigma)} qubits, frame has {self.num_qubits}')
        old = self._logical_to_site
        self._logical_to_site = [old[sigma[q]] for q in range(self.num_qubits)]
        for logical, site in enumerate(self._logical_to_site):
            self._site_to_logical[site] = logical
        self._check()

    def exit_seam(self, sigma: Permutation) -> None:
        self.enter_seam(sigma.inverse())

    def apply_site_transposition(self, site_a: int, site_b: int) -> None:
        if site_a == site_b:
            raise FrameStateError('site transposition endpoints must differ')
        for site in (site_a, site_b):
            if site < 0 or site >= self.num_qubits:
                raise FrameStateError(f'site {site} out of range')
        logical_a = self._site_to_logical[site_a]
        logical_b = self._site_to_logical[site_b]
        self._logical_to_site[logical_a] = site_b
        self._logical_to_site[logical_b] = site_a
        self._site_to_logical[site_a] = logical_b
        self._site_to_logical[site_b] = logical_a
        self._check()

    def logical_bits_from_site_bits(self, site_bits: str) -> str:
        if len(site_bits) != self.num_qubits:
            raise FrameStateError(f'expected {self.num_qubits} site bits, got {len(site_bits)}')
        return ''.join((site_bits[self._logical_to_site[q]] for q in range(self.num_qubits)))

    def site_bits_from_logical_bits(self, logical_bits: str) -> str:
        if len(logical_bits) != self.num_qubits:
            raise FrameStateError(f'expected {self.num_qubits} logical bits, got {len(logical_bits)}')
        return ''.join((logical_bits[self._site_to_logical[s]] for s in range(self.num_qubits)))
