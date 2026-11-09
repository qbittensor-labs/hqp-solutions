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

# scratchpad wrapper: run level3/scripts/undo_inserted_pairs.py with the register size read from the source
import re, sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))         # level3/scripts
import undo_inserted_pairs as U
src = sys.argv[sys.argv.index("--source") + 1]
nq = int(re.search(r"qreg\s+\w+\[(\d+)\]", Path(src).read_text()).group(1))
orig = U.C.circuit_of
U.C.circuit_of = lambda res, n: orig(res, nq)
U.main()
