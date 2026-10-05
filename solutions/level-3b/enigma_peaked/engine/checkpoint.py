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
import io
import os
import pickle
from pathlib import Path
from typing import Any
import numpy as np
CHECKPOINT_SCHEMA = 'enigma-peaked-checkpoint-v1'
_ENGINE_CRITICAL_KEYS = ('mode', 'max_bond', 'cutoff', 'final_cutoff', 'unswap_threshold', 'max_its', 'beam_size', 'early_stopping_gates', 'center_ratio', 'sabre_trials', 'post_sabre_trials', 'compress_method', 'apply_cutoff', 'probe_cutoff', 'align_weight', 'align_protect', 'hows', 'equal', 'balanced_absorption', 'absorb_swaps_as_perm', 'gate_mpo_mode', 'preserve_raw_gates', 'equalize_norms', 'absorb_window', 'initial_layout', 'instruction_order', 'absorb_regions', 'absorb_param_schedule', 'unswap_hysteresis', 'unswap_cycle_cap')
_ENGINE_CRITICAL_DEFAULTS = {'post_sabre_trials': None, 'balanced_absorption': False, 'absorb_swaps_as_perm': False, 'gate_mpo_mode': 'quimb_graph', 'preserve_raw_gates': False, 'equalize_norms': False, 'absorb_window': None, 'initial_layout': None, 'instruction_order': None, 'absorb_regions': None, 'absorb_param_schedule': None, 'unswap_hysteresis': True, 'unswap_cycle_cap': 0}

class CheckpointError(RuntimeError):
    pass

def _circuits_to_qpy(circuits) -> bytes:
    from qiskit import qpy
    buffer = io.BytesIO()
    qpy.dump(list(circuits), buffer)
    return buffer.getvalue()

def _circuits_from_qpy(payload: bytes):
    from qiskit import qpy
    return list(qpy.load(io.BytesIO(payload)))

def mpo_to_portable(mpo) -> dict[str, Any]:
    arrays = []
    metadata = []
    for tensor in mpo:
        data = tensor.data
        if hasattr(data, 'cpu'):
            data = data.cpu()
        arrays.append(np.asarray(data))
        metadata.append({'inds': list(tensor.inds), 'tags': sorted(tensor.tags)})
    return {'L': int(mpo.L), 'site_tag_id': mpo.site_tag_id, 'upper_ind_id': mpo.upper_ind_id, 'lower_ind_id': mpo.lower_ind_id, 'exponent': float(mpo.exponent), 'arrays': arrays, 'tensors': metadata}

def mpo_from_portable(payload: dict[str, Any], to_backend=None):
    from quimb.tensor import MatrixProductOperator, Tensor, TensorNetwork
    tensors = []
    for array, meta in zip(payload['arrays'], payload['tensors']):
        data = to_backend(array) if to_backend is not None else array
        tensors.append(Tensor(data=data, inds=tuple(meta['inds']), tags=set(meta['tags'])))
    tn = TensorNetwork(tensors)
    mpo = tn.view_as_(MatrixProductOperator, cyclic=False, L=payload['L'], site_tag_id=payload['site_tag_id'], upper_ind_id=payload['upper_ind_id'], lower_ind_id=payload['lower_ind_id'])
    mpo.ensure_bonds_exist()
    mpo.exponent = float(payload.get('exponent', 0.0))
    return mpo

def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    payload = dict(state)
    payload['schema'] = CHECKPOINT_SCHEMA
    for key in ('layers_left', 'layers_right', 'init_meas', 'final_meas'):
        payload[key] = _circuits_to_qpy(payload[key])
    payload['mpo'] = mpo_to_portable(payload['mpo'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    with temp_path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)

def load_checkpoint(path: str | Path, *, expected_input_sha256: str | None=None, expected_config: dict[str, Any] | None=None, to_backend=None) -> dict[str, Any]:
    path = Path(path)
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    if payload.get('schema') != CHECKPOINT_SCHEMA:
        raise CheckpointError(f"unsupported checkpoint schema {payload.get('schema')!r}; expected {CHECKPOINT_SCHEMA}")
    if expected_input_sha256 is not None and payload['input_qasm_sha256'] != expected_input_sha256:
        raise CheckpointError('checkpoint input hash does not match the requested input circuit')
    if expected_config is not None:
        saved = payload['engine_config']
        for key in _ENGINE_CRITICAL_KEYS:
            default = _ENGINE_CRITICAL_DEFAULTS.get(key)
            saved_value = saved.get(key, default)
            expected_value = expected_config.get(key, default)
            if saved_value != expected_value:
                raise CheckpointError(f'checkpoint engine config mismatch on {key!r}: {saved_value!r} != {expected_value!r}')
    for key in ('layers_left', 'layers_right', 'init_meas', 'final_meas'):
        payload[key] = _circuits_from_qpy(payload[key])
    payload['mpo'] = mpo_from_portable(payload['mpo'], to_backend=to_backend)
    return payload
