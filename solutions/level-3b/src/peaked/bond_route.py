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

class BondProfile:

    def __init__(self):
        self.bonds = None

    def record(self, mpo):
        n = len(mpo.sites) if hasattr(mpo, 'sites') else mpo.L
        self.bonds = [int(mpo.bond_size(i, i + 1)) for i in range(n - 1)]

def near_front_cost(layers, bonds, horizon):
    cost, seen, total = (0, 0, 0)
    for layer in layers:
        for ins in layer.data:
            if len(ins.qubits) != 2 or ins.operation.name in ('measure', 'barrier'):
                continue
            total += 1
            if seen < horizon:
                a, b = (layer.find_bit(q).index for q in ins.qubits)
                cut = min(a, b)
                if 0 <= cut < len(bonds):
                    cost += int(bonds[cut])
                seen += 1
    return (cost, total)

def make_bond_rewire(base_rewire, profile, candidates, horizon, event=None):

    def rewire_layers(layers, perm, seed=None, sabre_trials: int=200):
        bonds = profile.bonds
        if bonds is None or candidates <= 1:
            return base_rewire(layers, perm, seed=seed, sabre_trials=sabre_trials)
        base = 0 if seed is None else int(seed)
        best = None
        for k in range(candidates):
            routed = base_rewire(layers, perm, seed=base * 1000 + k, sabre_trials=sabre_trials)
            cost, total = near_front_cost(routed, bonds, horizon)
            key = (cost, total, k)
            if best is None or key < best[0]:
                best = (key, routed)
        if event is not None:
            event(best[0])
        return best[1]
    return rewire_layers
