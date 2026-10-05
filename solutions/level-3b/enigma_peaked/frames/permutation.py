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
from typing import Iterable, Iterator

class PermutationError(ValueError):
    pass

class Permutation:
    __slots__ = ('_mapping',)

    def __init__(self, mapping: Iterable[int]) -> None:
        values = tuple(mapping)
        if not values:
            raise PermutationError('permutation must not be empty')
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise PermutationError(f'permutation entries must be integers, got {value!r}')
        size = len(values)
        if any((value < 0 or value >= size for value in values)):
            raise PermutationError(f'permutation entries must lie in [0, {size - 1}]')
        if len(set(values)) != size:
            raise PermutationError('permutation entries must be unique')
        self._mapping = values

    @classmethod
    def identity(cls, size: int) -> Permutation:
        if size <= 0:
            raise PermutationError('size must be positive')
        return cls(range(size))

    @property
    def mapping(self) -> tuple[int, ...]:
        return self._mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __getitem__(self, index: int) -> int:
        return self._mapping[index]

    def __iter__(self) -> Iterator[int]:
        return iter(self._mapping)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permutation):
            return NotImplemented
        return self._mapping == other._mapping

    def __hash__(self) -> int:
        return hash(self._mapping)

    def __repr__(self) -> str:
        return f'Permutation({list(self._mapping)})'

    def inverse(self) -> Permutation:
        inverse = [0] * len(self._mapping)
        for source, destination in enumerate(self._mapping):
            inverse[destination] = source
        return Permutation(inverse)

    def compose(self, other: Permutation) -> Permutation:
        if len(other) != len(self):
            raise PermutationError('cannot compose permutations of different sizes')
        return Permutation((self._mapping[value] for value in other))

    def is_identity(self) -> bool:
        return all((value == index for index, value in enumerate(self._mapping)))

    def is_involution(self) -> bool:
        return all((self._mapping[value] == index for index, value in enumerate(self._mapping)))

    def fixed_points(self) -> list[int]:
        return [index for index, value in enumerate(self._mapping) if value == index]

    def cycles(self) -> list[list[int]]:
        seen = [False] * len(self._mapping)
        cycles: list[list[int]] = []
        for start in range(len(self._mapping)):
            if seen[start]:
                continue
            cycle = [start]
            seen[start] = True
            current = self._mapping[start]
            while current != start:
                cycle.append(current)
                seen[current] = True
                current = self._mapping[current]
            if len(cycle) > 1:
                cycles.append(cycle)
        return cycles

def permutation_to_transpositions(permutation: Permutation) -> list[tuple[int, int]]:
    swaps: list[tuple[int, int]] = []
    for cycle in permutation.cycles():
        for index in range(1, len(cycle)):
            swaps.append((cycle[0], cycle[index]))
    return swaps
