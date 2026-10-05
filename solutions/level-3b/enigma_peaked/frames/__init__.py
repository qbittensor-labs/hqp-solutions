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

from .permutation import Permutation, PermutationError, permutation_to_transpositions
from .state import FrameState, FrameStateError
from .schedule import FrameEvent, FrameSchedule, FrameScheduleError, ModuleFrame
from .structural import StructuralConfig, evaluate_eligibility, verify_module
from .transform import FrameTransformError, materialize_frame_circuit, program_to_circuit, virtual_frame_program
__all__ = ['FrameEvent', 'FrameSchedule', 'FrameScheduleError', 'FrameState', 'FrameStateError', 'FrameTransformError', 'ModuleFrame', 'Permutation', 'PermutationError', 'StructuralConfig', 'evaluate_eligibility', 'materialize_frame_circuit', 'permutation_to_transpositions', 'program_to_circuit', 'verify_module', 'virtual_frame_program']
