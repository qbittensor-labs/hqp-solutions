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

# fold Pauli-Z byproducts of undo010 into the NEXT u gate on the wire (Z then u(t,p,l) = u(t,p,l+pi) up to phase);
# a trailing Z with no later u on the wire is dropped (diagonal, no effect on measurement probabilities)
import re, sys, math
from pathlib import Path
src, dst = sys.argv[1], sys.argv[2]
lines = Path(src).read_text().splitlines()
def ang(x):
    return float(eval(x.replace('pi', 'math.pi'), {'__builtins__': {}}, {'math': math}))
pending = {}   # wire -> phase to add to lambda of the next u
out = []
nz = ndrop = 0
for ln in lines:
    m = re.match(r"\s*z\s+q\[(\d+)\];", ln)
    if m:
        w = int(m.group(1)); pending[w] = pending.get(w, 0.0) + math.pi; nz += 1; continue
    m = re.match(r"\s*u\(([^,]+),([^,]+),([^)]+)\)\s+q\[(\d+)\];", ln)
    if m and int(m.group(4)) in pending:
        t, p, l = (ang(m.group(i)) for i in (1, 2, 3)); w = int(m.group(4))
        l = l + pending.pop(w)
        out.append(f"u({t!r},{p!r},{l!r}) q[{w}];"); continue
    m = re.match(r"\s*cz\s+q\[(\d+)\],\s*q\[(\d+)\];", ln)
    if m and (int(m.group(1)) in pending or int(m.group(2)) in pending):
        pass   # Z commutes with CZ: keep pending
    out.append(ln)
ndrop = len(pending)
Path(dst).write_text("\n".join(out) + "\n")
print(f"folded {nz} z gates, {ndrop} trailing dropped; lines {len(lines)} -> {len(out)}")
