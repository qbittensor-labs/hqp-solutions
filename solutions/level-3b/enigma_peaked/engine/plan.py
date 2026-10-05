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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from ..config import ConfigError, load_plan
_KNOWN_TOP_KEYS = {'schema_version', 'plan_id', 'instance_id', 'target_access', 'purpose', 'qasm_path', 'qasm_sha256', 'engine', 'execution', 'decoder', 'verifier'}
_KNOWN_ENGINE_KEYS = {'name', 'mode', 'max_bond', 'cutoff', 'final_cutoff', 'unswap_threshold', 'max_its', 'beam_size', 'topk', 'ordering_seeds', 'early_stopping_gates', 'center_ratio', 'sabre_trials', 'post_sabre_trials', 'compress_method', 'apply_cutoff', 'probe_cutoff', 'align_weight', 'align_protect', 'hows', 'equal', 'frame_report', 'checkpoint_every_seconds', 'checkpoint_every_layers', 'implementation_status', 'balanced_absorption', 'absorb_swaps_as_perm', 'gate_mpo_mode', 'preserve_raw_gates', 'equalize_norms', 'absorb_window', 'factor_export', 'initial_layout', 'instruction_order', 'absorb_regions', 'staged_transpilation', 'staged_region_windows', 'staged_exact_module_frames', 'absorb_param_schedule', 'unswap_hysteresis', 'unswap_cycle_cap'}
_KNOWN_EXECUTION_KEYS = {'enabled', 'wall_seconds', 'dtype', 'device', 'backend'}
_DTYPES = {'complex64', 'complex128'}
_MODES = {'baseline', 'module_frame'}
_INSTANCES = {'s1', 's2', 'p9-control', 'local'}

@dataclass(frozen=True)
class GeneratorPlan:
    plan_id: str
    instance_id: str
    qasm_path: str | None
    qasm_sha256: str | None
    engine: dict[str, Any]
    execution: dict[str, Any]
    decoder: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    base_dir: Path | None = None

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute() and self.base_dir is not None:
            return self.base_dir / path
        return path

    @property
    def mode(self) -> str:
        return self.engine['mode']

    @property
    def enabled(self) -> bool:
        return bool(self.execution.get('enabled', True))

def _require(table: dict[str, Any], key: str, types, label: str):
    if key not in table:
        raise ConfigError(f'{label}: missing required key {key!r}')
    value = table[key]
    types_tuple = types if isinstance(types, tuple) else (types,)
    if isinstance(value, bool) and bool not in types_tuple:
        raise ConfigError(f'{label}.{key}: unexpected type bool')
    if not isinstance(value, types_tuple):
        raise ConfigError(f'{label}.{key}: unexpected type {type(value).__name__}')
    return value

def _reject_unknown(table: dict[str, Any], known: set[str], label: str) -> None:
    unknown = sorted(set(table) - known)
    if unknown:
        raise ConfigError(f"{label}: unknown high-impact keys: {', '.join(unknown)}")

def load_generator_plan(path: str | Path) -> GeneratorPlan:
    plan = load_plan(path)
    if plan['target_access'] != 'blind':
        raise ConfigError('generator plans must be blind; evaluation is post-hoc only')
    _reject_unknown(plan, _KNOWN_TOP_KEYS, 'plan')
    instance_id = plan['instance_id']
    if instance_id not in _INSTANCES:
        raise ConfigError(f'unknown instance_id {instance_id!r}')
    if instance_id == 'local':
        if not plan.get('qasm_path') or not plan.get('qasm_sha256'):
            raise ConfigError('local plans require qasm_path and qasm_sha256')
    engine = plan['engine']
    if not isinstance(engine, dict):
        raise ConfigError('engine must be a table')
    _reject_unknown(engine, _KNOWN_ENGINE_KEYS, 'engine')
    if _require(engine, 'name', str, 'engine') != 'mpo-unswap':
        raise ConfigError("engine.name must be 'mpo-unswap'")
    mode = _require(engine, 'mode', str, 'engine')
    if mode not in _MODES:
        raise ConfigError(f'engine.mode must be one of {sorted(_MODES)}')
    if mode == 'module_frame' and (not engine.get('frame_report')):
        raise ConfigError("engine.mode='module_frame' requires engine.frame_report")
    _require(engine, 'max_bond', int, 'engine')
    _require(engine, 'cutoff', (int, float), 'engine')
    _require(engine, 'final_cutoff', (int, float), 'engine')
    _require(engine, 'unswap_threshold', (int, float), 'engine')
    _require(engine, 'max_its', int, 'engine')
    _require(engine, 'beam_size', int, 'engine')
    seeds = _require(engine, 'ordering_seeds', list, 'engine')
    if not seeds or not all((isinstance(s, int) and (not isinstance(s, bool)) for s in seeds)):
        raise ConfigError('engine.ordering_seeds must be a nonempty list of integers')
    if len(set(seeds)) != len(seeds):
        raise ConfigError('engine.ordering_seeds must be unique')
    for key, lower in (('max_bond', 1), ('max_its', 1), ('beam_size', 1)):
        if engine[key] < lower:
            raise ConfigError(f'engine.{key} must be >= {lower}')
    for key in ('cutoff', 'final_cutoff', 'unswap_threshold'):
        if engine[key] < 0:
            raise ConfigError(f'engine.{key} must be nonnegative')
    for key in ('balanced_absorption', 'absorb_swaps_as_perm', 'preserve_raw_gates', 'equalize_norms', 'factor_export', 'unswap_hysteresis', 'staged_transpilation'):
        if key in engine and (not isinstance(engine[key], bool)):
            raise ConfigError(f'engine.{key} must be a boolean')
    if engine.get('gate_mpo_mode', 'quimb_graph') not in {'quimb_graph', 'strict_chain'}:
        raise ConfigError("engine.gate_mpo_mode must be 'quimb_graph' or 'strict_chain'")
    if engine.get('compress_method', 'zipup') not in {'zipup', 'dm', 'fit', 'legacy_apply'}:
        raise ConfigError("engine.compress_method must be 'zipup', 'dm', 'fit', or 'legacy_apply'")
    if 'unswap_cycle_cap' in engine:
        cap = engine['unswap_cycle_cap']
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
            raise ConfigError('engine.unswap_cycle_cap must be a nonnegative integer (0 = disabled)')
    if 'absorb_window' in engine:
        window = engine['absorb_window']
        if not isinstance(window, list) or len(window) != 2 or any((isinstance(v, bool) or not isinstance(v, int) for v in window)):
            raise ConfigError('engine.absorb_window must be a two-integer list [start, end]')
        if window[0] < 0 or window[1] <= window[0]:
            raise ConfigError('engine.absorb_window must satisfy 0 <= start < end')
    if 'initial_layout' in engine:
        layout = engine['initial_layout']
        if not isinstance(layout, list) or not layout or any((isinstance(v, bool) or not isinstance(v, int) for v in layout)) or (sorted(layout) != list(range(len(layout)))):
            raise ConfigError('engine.initial_layout must be a permutation of 0..n-1 as an integer list')
    if 'instruction_order' in engine:
        order = engine['instruction_order']
        if not isinstance(order, list) or not order or any((isinstance(v, bool) or not isinstance(v, int) for v in order)) or (sorted(order) != list(range(len(order)))):
            raise ConfigError('engine.instruction_order must be a permutation of 0..n-1 as an integer list (topological legality is enforced at run time)')
    if 'absorb_param_schedule' in engine:
        sched = engine['absorb_param_schedule']
        if not isinstance(sched, list) or not sched:
            raise ConfigError('engine.absorb_param_schedule must be a nonempty list of tables')
        previous_end = None
        for index, phase in enumerate(sched):
            if not isinstance(phase, dict) or set(phase) - {'window', 'max_bond', 'cutoff', 'unswap_threshold'}:
                raise ConfigError(f"engine.absorb_param_schedule[{index}] must be a table with keys 'window' and optional 'max_bond'/'cutoff'/'unswap_threshold'")
            window = phase.get('window')
            if not isinstance(window, list) or len(window) != 2 or any((isinstance(v, bool) or not isinstance(v, int) for v in window)) or (window[0] < 0) or (window[1] <= window[0]):
                raise ConfigError(f'engine.absorb_param_schedule[{index}].window must be [start, end] integers with 0 <= start < end')
            if previous_end is not None and window[0] != previous_end:
                raise ConfigError('engine.absorb_param_schedule windows must be contiguous and ascending')
            previous_end = window[1]
            mb = phase.get('max_bond')
            if mb is not None and (isinstance(mb, bool) or not isinstance(mb, int) or mb < 1):
                raise ConfigError(f'engine.absorb_param_schedule[{index}].max_bond must be a positive integer')
            co = phase.get('cutoff')
            if co is not None and (isinstance(co, bool) or not isinstance(co, (int, float)) or co < 0):
                raise ConfigError(f'engine.absorb_param_schedule[{index}].cutoff must be a nonnegative number')
            threshold = phase.get('unswap_threshold')
            if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold < 0):
                raise ConfigError(f'engine.absorb_param_schedule[{index}].unswap_threshold must be a nonnegative number')
    if 'absorb_regions' in engine:
        regions = engine['absorb_regions']
        if not isinstance(regions, list) or not regions:
            raise ConfigError('engine.absorb_regions must be a nonempty list of tables')
        previous_end = None
        for index, region in enumerate(regions):
            if not isinstance(region, dict) or set(region) - {'window', 'layout'}:
                raise ConfigError(f"engine.absorb_regions[{index}] must be a table with keys 'window' and optional 'layout'")
            window = region.get('window')
            if not isinstance(window, list) or len(window) != 2 or any((isinstance(v, bool) or not isinstance(v, int) for v in window)) or (window[0] < 0) or (window[1] <= window[0]):
                raise ConfigError(f'engine.absorb_regions[{index}].window must be [start, end] integers with 0 <= start < end')
            if previous_end is not None and window[0] != previous_end:
                raise ConfigError('engine.absorb_regions windows must be contiguous and ascending')
            previous_end = window[1]
            layout = region.get('layout')
            if layout is not None and (not isinstance(layout, list) or not layout or any((isinstance(v, bool) or not isinstance(v, int) for v in layout)) or (sorted(layout) != list(range(len(layout))))):
                raise ConfigError(f'engine.absorb_regions[{index}].layout must be a permutation of 0..n-1')
        if not engine.get('absorb_swaps_as_perm'):
            raise ConfigError('engine.absorb_regions requires absorb_swaps_as_perm = true: the inter-region seam layers are exact only when routing stays virtual')
        if 'absorb_window' in engine or 'initial_layout' in engine:
            raise ConfigError('engine.absorb_regions replaces absorb_window/initial_layout; do not combine them')
    if engine.get('staged_transpilation'):
        if 'absorb_regions' in engine:
            raise ConfigError('engine.staged_transpilation and absorb_regions are alternative chunking modes; do not combine them')
        if engine.get('early_stopping_gates', 30) != 0:
            raise ConfigError('engine.staged_transpilation requires early_stopping_gates = 0 so both delayed outer chunks are always reconnected')
        windows = engine.get('staged_region_windows')
        if windows is not None:
            if not isinstance(windows, list) or len(windows) < 3 or len(windows) % 2 == 0:
                raise ConfigError('engine.staged_region_windows must be an odd-length list of at least three contiguous [start, end] windows')
            previous_end = None
            for index, window in enumerate(windows):
                if not isinstance(window, list) or len(window) != 2 or any((isinstance(v, bool) or not isinstance(v, int) for v in window)) or (window[0] < 0) or (window[1] <= window[0]):
                    raise ConfigError(f'engine.staged_region_windows[{index}] must be [start, end] with 0 <= start < end')
                if previous_end is not None and window[0] != previous_end:
                    raise ConfigError('engine.staged_region_windows must be contiguous and ascending')
                previous_end = window[1]
    if engine.get('staged_exact_module_frames'):
        if mode != 'module_frame':
            raise ConfigError("engine.staged_exact_module_frames requires mode='module_frame'")
        if not engine.get('staged_transpilation'):
            raise ConfigError('engine.staged_exact_module_frames requires staged_transpilation=true')
        if not engine.get('absorb_swaps_as_perm'):
            raise ConfigError('engine.staged_exact_module_frames requires absorb_swaps_as_perm=true')
    if engine.get('factor_export') and engine.get('early_stopping_gates', 30) != 0:
        raise ConfigError('engine.factor_export requires early_stopping_gates = 0: a factor must ingest its complete window, never stop early')
    execution = plan['execution']
    if not isinstance(execution, dict):
        raise ConfigError('execution must be a table')
    _reject_unknown(execution, _KNOWN_EXECUTION_KEYS, 'execution')
    wall = _require(execution, 'wall_seconds', (int, float), 'execution')
    if wall <= 0:
        raise ConfigError('execution.wall_seconds must be positive')
    dtype = _require(execution, 'dtype', str, 'execution')
    if dtype not in _DTYPES:
        raise ConfigError(f'execution.dtype must be one of {sorted(_DTYPES)}')
    device = _require(execution, 'device', str, 'execution')
    backend = execution.get('backend', 'numpy' if device == 'cpu' else 'torch')
    if backend not in {'numpy', 'torch'}:
        raise ConfigError("execution.backend must be 'numpy' or 'torch'")
    if backend == 'numpy' and device != 'cpu':
        raise ConfigError("the numpy backend requires device='cpu'")
    return GeneratorPlan(plan_id=plan['plan_id'], instance_id=instance_id, qasm_path=plan.get('qasm_path'), qasm_sha256=plan.get('qasm_sha256'), engine=dict(engine), execution={**execution, 'backend': backend}, decoder=dict(plan.get('decoder', {})), raw=plan, base_dir=Path(path).resolve().parent)
