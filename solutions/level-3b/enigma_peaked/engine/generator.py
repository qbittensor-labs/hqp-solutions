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

"""Blind MPO-unswap generator orchestration.

Ports the audited ``mpo_compress_unswap`` / ``mpo_to_mps`` absorption loop with
the corrections required by the migration plan:

- baseline and module-frame modes execute the same engine path (the validated
  frame schedule is materialized into the circuit before splitting; baseline
  uses the identity schedule);
- left/right MPO boundaries carry independent, persistent frame states that
  survive every reroute and checkpoint;
- counters distinguish absorbed work gates, routing SWAPs, unswap iterations,
  ordering restarts, and censored termination;
- checkpoints are atomic and complete; resume validates compatibility;
- no challenge output packaging and no target or probe injection of any kind;
- every run writes an ignored run directory with a manifest, effective config,
  an event ledger, candidates, and checkpoints.

Candidate weights are relative engine weights within the surviving truncated
state — they are not physical probabilities, and the telemetry surrogate is
never a fidelity.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np
from qiskit import QuantumCircuit

from .. import __version__
from ..frames.schedule import FrameSchedule, FrameScheduleError, ModuleFrame
from ..frames.permutation import Permutation, permutation_to_transpositions
from ..frames.state import FrameState
from ..frames.structural import StructuralConfig, evaluate_eligibility
from ..frames.transform import materialize_frame_circuit
from . import telemetry
from .backend import Backend, install_torch_linalg_patch, make_backend
from .checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from .extraction import beam_search
from .layers import elem_counts, get_tn_info, iter_layers, merge_instructions, merge_layers
from .mpo import (
    apply_circuit,
    apply_qiskit_circuit_strict_chain,
    mpo_from_circuit,
    mps_zero_state,
    quimb_circuit_from_qiskit,
    strict_chain_circuit_mpo,
)
from .plan import GeneratorPlan, load_generator_plan
from .unswap import (
    UnswapConfig,
    measurement_permutation,
    rewire_layers,
    unswap,
)


class GeneratorError(RuntimeError):
    """Raised when a generator run cannot proceed safely."""


def remap_work_layer(layer, perm):
    """Port of p9solver.perm_absorb.remap_work_layer.

    Build a circuit of this layer's WORK gates only, remapped onto MPO sites
    via ``perm[wire]`` (routing SWAPs are folded into ``perm`` instead of
    materialized, so they cost zero bond). Returns ``(work_circuit_or_None,
    new_perm)`` where ``new_perm`` applies this layer's SWAP pairs.
    """
    from qiskit import QuantumCircuit

    n = layer.num_qubits
    work = QuantumCircuit(n)
    any_work = False
    swaps: list[tuple[int, int]] = []
    for ci in layer.data:
        qs = [layer.find_bit(q).index for q in ci.qubits]
        name = ci.operation.name
        if name == "swap":
            swaps.append((qs[0], qs[1]))
        elif name in ("barrier", "measure", "delay"):
            continue
        else:
            work.append(ci.operation, [perm[q] for q in qs])
            any_work = True
    new_perm = list(perm)
    for i, j in swaps:
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return (work if any_work else None), new_perm


def compose_route_with_site_permutation(route, site_to_old):
    """Update ``route[wire] -> site`` after a physical site permutation.

    ``unswap`` records its cycle permutation by swapping *entries* of an
    identity list.  That list maps new sites to old sites; a running virtual
    routing map needs the inverse (old site to new site).  Composing the
    inverse into every route value keeps future work gates in the same fixed
    MPO frame without rerouting or materializing the routing SWAPs.
    """

    if sorted(route) != list(range(len(route))):
        raise GeneratorError(f"route map is not a permutation: {route}")
    if sorted(site_to_old) != list(range(len(route))):
        raise GeneratorError(f"site permutation is invalid: {site_to_old}")
    old_to_new = np.argsort(np.asarray(site_to_old, dtype=int)).tolist()
    return [old_to_new[site] for site in route]


ENGINE_DEFAULTS: dict[str, Any] = {
    "topk": 32,
    "early_stopping_gates": 30,
    "center_ratio": 0.5,
    "sabre_trials": 200,
    "post_sabre_trials": None,  # post-unswap reroute effort; None = same as sabre_trials
    "compress_method": "zipup",
    "apply_cutoff": None,
    "probe_cutoff": None,
    "align_weight": 0.0,
    "align_protect": None,
    "hows": ["both", "left", "right"],
    "equal": False,
    "checkpoint_every_seconds": 0.0,
    "checkpoint_every_layers": 0,
    "balanced_absorption": False,
    "absorb_swaps_as_perm": False,
    "gate_mpo_mode": "quimb_graph",
    "preserve_raw_gates": False,
    "equalize_norms": False,
    "absorb_window": None,
    "factor_export": False,
    "initial_layout": None,
    "instruction_order": None,
    "absorb_regions": None,
    # Route only the center-adjacent half of each side initially. The two
    # outer chunks remain raw until both inner chunks have been absorbed, then
    # they are routed under the accumulated boundary permutations.
    "staged_transpilation": False,
    # Optional contiguous stage windows for staged_transpilation. The region
    # containing center_ratio is routed first; adjacent pairs are activated
    # only after the current pair drains (e.g. gap -> M0/M1 -> prefix/suffix).
    "staged_region_windows": None,
    # For module_frame plans, rebuild five exact raw-layer chunks and apply
    # sigma only inside each report-validated module interval.
    "staged_exact_module_frames": False,
    # Per-region absorption parameters for a SINGLE continuous run: a list of
    # {window=[a,b), max_bond=..., cutoff=...} tables in consolidated units,
    # contiguous and ascending. Each absorbing front switches to a phase's
    # parameters when its cumulative absorbed work enters that phase's span,
    # so e.g. obfuscated modules get high bond / tight cutoff while gaps stay
    # cheap — no stitching, one operator, one routing. None = uniform (legacy).
    "absorb_param_schedule": None,
    "unswap_hysteresis": True,
    # Livelock hard cap: force one absorb after this many CONSECUTIVE unswap
    # cycles that made no absorption progress, even if each cycle applied swaps
    # (the zero-swap force_absorb guard does not fire when unswap is perpetually
    # "effective", which livelocks the S2 module centers). 0 = disabled (default,
    # preserves frozen-solver cadence). A small value (e.g. 6) bounds the thrash.
    "unswap_cycle_cap": 0,
}


@dataclass
class OrderingCounters:
    seed: int
    work_ops_total: int = 0
    work_ops_absorbed_left: int = 0
    work_ops_absorbed_right: int = 0
    work_ops_absorbed_total: int = 0
    routing_swaps_absorbed: int = 0
    layers_absorbed: int = 0
    unswap_cycles: int = 0
    unswap_iterations: int = 0
    unswap_swaps_applied: int = 0
    unswap_probe_batches: int = 0
    ordering_restarts: int = 0
    termination: str = "in_progress"
    censored: bool = False


@dataclass
class AbsorptionState:
    mpo: Any
    layers_left: list
    layers_right: list
    init_meas: list
    final_meas: list
    ii_left: int
    ii_right: int
    frame_left: FrameState
    frame_right: FrameState
    counters: OrderingCounters
    remaining_seeds: list[int] = field(default_factory=list)
    perm_left_route: list[int] = field(default_factory=list)
    perm_right_route: list[int] = field(default_factory=list)
    force_absorb: bool = False
    unswap_rearm: float = 0.0
    pending_outer_left: QuantumCircuit | None = None
    pending_outer_right: QuantumCircuit | None = None
    inner_end_left: list[int] | None = None
    inner_end_right: list[int] | None = None
    staged_outer_activated: bool = False
    pending_stage_left: list[QuantumCircuit] = field(default_factory=list)
    pending_stage_right: list[QuantumCircuit] = field(default_factory=list)
    staged_activation_index: int = 0


def _git_sha(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, check=True
        )
        return result.stdout.decode().strip()
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    versions = {"enigma-peaked": __version__}
    for name in ("qiskit", "quimb", "cotengra", "numpy", "scipy", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def effective_engine_config(plan: GeneratorPlan) -> dict[str, Any]:
    config = {**ENGINE_DEFAULTS, **plan.engine}
    config.pop("implementation_status", None)
    return config


def _count_work_ops_layers(layers) -> int:
    """Total WORK ops across a list of layer circuits (starvation guard)."""
    return sum(_count_work_ops(layer)[0] for layer in layers)


def _count_work_ops(circuit: QuantumCircuit) -> tuple[int, int]:
    """(work gates, swap gates); barriers/measures/delays are neither."""

    work = 0
    swaps = 0
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in {"barrier", "measure", "delay"}:
            continue
        if name == "swap":
            swaps += 1
        else:
            work += 1
    return work, swaps


def resolve_input_qasm(
    plan: GeneratorPlan, project_root: Path, qasm_override: str | Path | None = None
) -> tuple[Path, str]:
    """Locate and hash-verify the input circuit for the plan."""

    if plan.instance_id == "local":
        path = Path(qasm_override) if qasm_override else plan.resolve_path(plan.qasm_path)
        expected = plan.qasm_sha256
    else:
        manifest = json.loads(
            (project_root / "instances" / "manifest.json").read_text(encoding="utf-8")
        )
        entry = next(item for item in manifest["instances"] if item["id"] == plan.instance_id)
        path = Path(qasm_override or project_root / entry["expected_local_path"])
        expected = entry["qasm_sha256"]
    if not path.is_file():
        raise GeneratorError(f"input circuit not found: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise GeneratorError(f"input circuit hash {digest} does not match expected {expected}")
    return path, digest


def load_frame_schedule_for_plan(
    report: dict[str, Any], expected_sha256: str, num_qubits: int
) -> FrameSchedule:
    """Validate a frame report and rebuild its schedule, failing closed."""

    if report.get("qasm_sha256") != expected_sha256:
        raise GeneratorError("frame report qasm_sha256 does not match the input circuit")
    if report.get("num_qubits") != num_qubits:
        raise GeneratorError("frame report qubit count does not match the circuit")
    payload = report.get("schedule")
    if not payload:
        raise GeneratorError("frame report contains no schedule (no eligible modules)")
    try:
        modules = [ModuleFrame.from_dict(entry) for entry in report.get("modules", ())]
        schedule = FrameSchedule.from_dict(payload)
    except (FrameScheduleError, KeyError, ValueError) as exc:
        raise GeneratorError(f"invalid frame report: {exc}") from exc
    if schedule.qasm_sha256 != expected_sha256 or schedule.num_qubits != num_qubits:
        raise GeneratorError("frame schedule identity does not match the input circuit")
    eligible: dict[int, ModuleFrame] = {}
    for module in modules:
        if not module.eligible:
            continue
        if module.qasm_sha256 != expected_sha256:
            raise GeneratorError(f"module {module.module_id} hash does not match the input circuit")
        confirmed, reasons = evaluate_eligibility(module.evidence, StructuralConfig())
        if not confirmed:
            raise GeneratorError(
                f"module {module.module_id} fails the confidence gate: {'; '.join(reasons)}"
            )
        eligible[module.module_id] = module
    for event in schedule.events:
        module = eligible.get(event.module_id)
        if module is None:
            raise GeneratorError(
                f"schedule references module {event.module_id} which is not eligible"
            )
        if event.permutation != module.permutation:
            raise GeneratorError(
                f"schedule permutation for module {event.module_id} does not match "
                "the validated module"
            )
    return schedule


def prepare_circuit(
    qasm_path: Path,
    schedule: FrameSchedule,
    *,
    preserve_raw_gates: bool = False,
) -> QuantumCircuit:
    """Load, strip final measurements, apply the exact frame schedule, and
    consolidate 2q blocks. Baseline and module-frame modes share this path;
    baseline simply uses the identity schedule."""

    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks

    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    circuit.remove_final_measurements(inplace=True)
    framed = materialize_frame_circuit(circuit, schedule)
    if preserve_raw_gates:
        return framed
    return PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(framed)


def prepare_exact_staged_frame_circuit(
    qasm_path: Path,
    schedule: FrameSchedule,
    report: dict[str, Any],
    config: dict[str, Any],
) -> QuantumCircuit:
    """Build prefix/M0/gap/M1/suffix with exact interior sigma frames.

    The structural report is in raw ASAP-layer coordinates, whereas the normal
    staged plan is in globally consolidated instruction coordinates.  Rather
    than materializing seams globally (which invalidates that coordinate), this
    path partitions the raw dependency layers first, applies each sigma only
    over its validated module interval, consolidates the five regions
    independently, and translates every staged coordinate to the resulting
    exact circuit.
    """

    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks

    from ..structure.instruction_windows import asap_layers

    raw = QuantumCircuit.from_qasm_file(str(qasm_path))
    raw.remove_final_measurements(inplace=True)
    layers = asap_layers(raw)
    depth = max(layers, default=-1) + 1
    modules = sorted(
        (
            ModuleFrame.from_dict(entry)
            for entry in report.get("modules", ())
            if entry.get("eligible")
        ),
        key=lambda module: module.enter_layer,
    )
    if len(modules) != 2:
        raise GeneratorError(
            "staged_exact_module_frames requires exactly two eligible modules"
        )
    module_by_id = {module.module_id: module for module in modules}
    enter_events = [event for event in schedule.events if event.kind == "enter"]
    if {event.module_id for event in enter_events} != set(module_by_id):
        raise GeneratorError("exact staged frame schedule does not match report modules")

    starts = []
    for module in modules:
        boundaries = module.evidence.get("boundaries", {})
        try:
            starts.append(int(boundaries["early_layer_min"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise GeneratorError(
                f"module {module.module_id} lacks an early-layer boundary"
            ) from exc
    region_bounds = [
        (0, starts[0], None),
        (starts[0], modules[0].exit_layer, modules[0]),
        (modules[0].exit_layer, starts[1], None),
        (starts[1], modules[1].exit_layer, modules[1]),
        (modules[1].exit_layer, depth, None),
    ]
    if any(not (0 <= start < stop <= depth) for start, stop, _ in region_bounds):
        raise GeneratorError(f"invalid exact staged raw-layer regions: {region_bounds}")

    ordered = sorted(enumerate(raw.data), key=lambda item: (layers[item[0]], item[0]))
    prepared_regions: list[QuantumCircuit] = []
    assigned = 0

    def consolidate(region: QuantumCircuit) -> QuantumCircuit:
        return PassManager(
            [Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]
        ).run(region)

    for start, stop, module in region_bounds:
        before = QuantumCircuit(raw.num_qubits)
        inside = QuantumCircuit(raw.num_qubits)

        def append_seam(target: QuantumCircuit, permutation: Permutation) -> None:
            for site_a, site_b in permutation_to_transpositions(permutation):
                target.swap(site_a, site_b)

        for raw_index, instruction in ordered:
            layer = layers[raw_index]
            if not (start <= layer < stop):
                continue
            assigned += 1
            qubits = [raw.find_bit(q).index for q in instruction.qubits]
            if module is not None and layer >= module.enter_layer:
                qubits = [module.permutation[q] for q in qubits]
                inside.append(instruction.operation, qubits)
            else:
                before.append(instruction.operation, qubits)
        if module is None:
            prepared_regions.append(consolidate(before))
            continue
        # Consolidate independently on each side of the seam. Otherwise the
        # block pass folds logical sigma SWAPs into generic unitary blocks and
        # the virtual-permutation absorber can no longer recognize them.
        region = QuantumCircuit(raw.num_qubits)
        region.compose(consolidate(before), inplace=True)
        append_seam(region, module.permutation)
        region.compose(consolidate(inside), inplace=True)
        append_seam(region, module.permutation.inverse())
        prepared_regions.append(region)
    if assigned != len(raw.data):
        raise GeneratorError(
            f"exact staged partition assigned {assigned}/{len(raw.data)} raw operations"
        )

    circuit = QuantumCircuit(raw.num_qubits)
    windows = []
    cursor = 0
    for region in prepared_regions:
        circuit.compose(region, inplace=True)
        windows.append([cursor, cursor + len(region.data)])
        cursor += len(region.data)

    old_windows = config.get("staged_region_windows")
    old_center = int(config["center_ratio"])
    if old_windows is None or len(old_windows) != 5:
        raise GeneratorError(
            "staged_exact_module_frames requires five staged_region_windows"
        )
    old_gap_start, old_gap_stop = map(int, old_windows[2])
    center_fraction = (old_center - old_gap_start) / (old_gap_stop - old_gap_start)
    new_gap_start, new_gap_stop = windows[2]
    new_center = new_gap_start + round(
        center_fraction * (new_gap_stop - new_gap_start)
    )
    new_center = min(max(new_center, new_gap_start + 1), new_gap_stop - 1)

    config["instruction_order"] = None
    config["staged_region_windows"] = windows
    config["absorb_window"] = [0, len(circuit.data)]
    config["center_ratio"] = new_center
    schedule_entries = config.get("absorb_param_schedule")
    if schedule_entries is not None:
        if len(schedule_entries) != 5:
            raise GeneratorError(
                "exact staged frames require a five-entry absorb_param_schedule"
            )
        config["absorb_param_schedule"] = [
            {**entry, "window": window}
            for entry, window in zip(schedule_entries, windows)
        ]
    return circuit


def resolve_absorb_window(config: dict[str, Any], num_instructions: int) -> tuple[int, int, int]:
    """Return ``(start, center, end)`` instruction bounds for absorption.

    Without ``absorb_window`` the whole circuit is absorbed (legacy behavior).
    With a window only ``data[start:end]`` is routed and absorbed, so material
    outside the window never enters SABRE routing or the MPO; a float
    ``center_ratio`` is then relative to the window.
    """

    regions = config.get("absorb_regions")
    window = config.get("absorb_window")
    if regions is not None:
        start, end = int(regions[0]["window"][0]), int(regions[-1]["window"][1])
    elif window is None:
        start, end = 0, num_instructions
    else:
        start, end = int(window[0]), int(window[1])
    if not (0 <= start < end <= num_instructions):
        raise GeneratorError(
            f"absorb window [{start}, {end}) outside circuit with "
            f"{num_instructions} instructions"
        )
    center_ratio = config["center_ratio"]
    if isinstance(center_ratio, float):
        center = start + int((end - start) * center_ratio)
    else:
        center = int(center_ratio)
    if not (start <= center <= end):
        raise GeneratorError(
            f"absorption center {center} outside window [{start}, {end}]"
        )
    return start, center, end


def relabel_circuit(circuit: QuantumCircuit, layout: list[int]) -> QuantumCircuit:
    """Exact change of tensor layout: qubit ``q`` becomes site ``layout[q]``.

    A relabel preserves every operation (no gates are added or removed); it
    only changes which 1-D site each logical qubit occupies, so gate spans —
    and therefore MPO bond growth — depend on it.
    """

    if sorted(layout) != list(range(circuit.num_qubits)):
        raise GeneratorError(
            f"initial_layout is not a permutation of 0..{circuit.num_qubits - 1}: {layout}"
        )
    relabeled = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        relabeled.append(
            instruction.operation,
            [layout[circuit.find_bit(q).index] for q in instruction.qubits],
        )
    return relabeled


def reorder_instructions(circuit: QuantumCircuit, order: list[int]) -> QuantumCircuit:
    """Reorder instructions along a different topological order of the DAG.

    Exact by construction: the reorder is rejected unless it is a permutation
    that preserves the relative order of instructions on every qubit, which is
    exactly the condition for the reordered circuit to equal the original
    unitary (dependencies are per-qubit chains).
    """

    n = len(circuit.data)
    if sorted(order) != list(range(n)):
        raise GeneratorError(
            f"instruction_order is not a permutation of 0..{n - 1} "
            f"(length {len(order)})"
        )
    position = [0] * n
    for new_index, old_index in enumerate(order):
        position[old_index] = new_index
    last_position: dict[int, int] = {}
    for old_index, instruction in enumerate(circuit.data):
        for qubit in instruction.qubits:
            qubit_index = circuit.find_bit(qubit).index
            if qubit_index in last_position and position[old_index] < last_position[qubit_index]:
                raise GeneratorError(
                    "instruction_order is not a topological order: it reverses "
                    f"two operations on qubit {qubit_index}"
                )
            last_position[qubit_index] = max(
                last_position.get(qubit_index, -1), position[old_index]
            )
    reordered = QuantumCircuit(circuit.num_qubits)
    for old_index in order:
        instruction = circuit.data[old_index]
        reordered.append(
            instruction.operation,
            [circuit.find_bit(q).index for q in instruction.qubits],
        )
    return reordered


def _routed_segment(
    circuit: QuantumCircuit,
    start: int,
    stop: int,
    layout: list[int] | None,
    *,
    invert: bool,
    seed: int,
    sabre_trials: int,
) -> tuple[list, list[int], list[int]]:
    """Route ``data[start:stop]`` in its own layout as an isolated segment.

    Returns ``(layers, start_site_of, end_site_of)`` where the site lists map
    logical qubit q to its site at the segment's absorption-stream entry and
    exit. Routing sees only this segment's gates, so one region's connectivity
    never influences another region's transpilation.
    """

    num_qubits = circuit.num_qubits
    segment = merge_instructions(circuit, start, stop)
    if layout is None:
        layout = list(range(num_qubits))
    else:
        layout = [int(v) for v in layout]
        segment = relabel_circuit(segment, layout)
    if invert:
        segment = segment.inverse()
    segment.measure_all()
    routed = rewire_layers(
        list(iter_layers(segment)),
        list(range(num_qubits)),
        seed=seed,
        sabre_trials=sabre_trials,
    )
    layers, measurement = routed[:-2], routed[-2:]
    wire_to_site = measurement_permutation(measurement)
    end_site_of = [wire_to_site[layout[q]] for q in range(num_qubits)]
    return layers, list(layout), end_site_of


def _seam_swap_layer(
    previous_end_site_of: list[int], next_start_site_of: list[int]
) -> QuantumCircuit | None:
    """SWAP-only layer moving every qubit between two segment placements.

    Under permutation absorption these swaps are folded into the virtual route
    maps and cost zero bond; the layer exists so the concatenated layer stream
    remains one exactly-equivalent routed circuit.
    """

    num_qubits = len(previous_end_site_of)
    site_map = [0] * num_qubits
    for q in range(num_qubits):
        site_map[previous_end_site_of[q]] = next_start_site_of[q]
    if site_map == list(range(num_qubits)):
        return None
    layer = QuantumCircuit(num_qubits)
    seen = [False] * num_qubits
    for site in range(num_qubits):
        if seen[site] or site_map[site] == site:
            seen[site] = True
            continue
        cycle = []
        cursor = site
        while not seen[cursor]:
            seen[cursor] = True
            cycle.append(cursor)
            cursor = site_map[cursor]
        for index in range(len(cycle) - 1, 0, -1):
            layer.swap(cycle[index - 1], cycle[index])
    return layer


def _synthetic_measurement_layers(site_of: list[int]) -> list[QuantumCircuit]:
    """Barrier + measurement layers recording ``logical q -> site_of[q]``."""

    num_qubits = len(site_of)
    barrier = QuantumCircuit(num_qubits, num_qubits)
    barrier.barrier()
    measure = QuantumCircuit(num_qubits, num_qubits)
    for q in range(num_qubits):
        measure.measure(site_of[q], q)
    return [barrier, measure]


def _route_staged_chunk(
    chunk: QuantumCircuit,
    *,
    seed: int,
    sabre_trials: int,
    perm: list[int] | None = None,
    layout: list[int] | None = None,
) -> tuple[list, list, list[int], list[int]]:
    """Route one delayed-transpilation chunk under its current wire frame."""

    if layout is None:
        layout = list(range(chunk.num_qubits))
    else:
        layout = [int(value) for value in layout]
        if sorted(layout) != list(range(chunk.num_qubits)):
            raise GeneratorError("staged chunk layout is not a permutation")
    routed_input = (
        chunk.copy()
        if layout == list(range(chunk.num_qubits))
        else relabel_circuit(chunk, layout)
    )
    routed_input.measure_all()
    if perm is None:
        perm = list(range(chunk.num_qubits))
    routed = rewire_layers(
        list(iter_layers(routed_input)),
        perm,
        seed=seed,
        sabre_trials=sabre_trials,
    )
    routed_measurement = routed[-2:]
    wire_to_site = measurement_permutation(routed_measurement)
    end_site_of = [wire_to_site[layout[q]] for q in range(chunk.num_qubits)]
    # Downstream extraction is expressed in original logical-qubit labels.
    # SABRE's measurement layer still labels the relabeled wires, so replace
    # it with an equivalent logical measurement map at the routed endpoint.
    measurement = _synthetic_measurement_layers(end_site_of)
    return routed[:-2], measurement, list(layout), end_site_of


def _transport_boundary_frame(
    old_base: list[int],
    current: FrameState,
    new_base: list[int],
) -> FrameState:
    """Move a routed endpoint through site permutations accrued in stage one.

    ``old_base`` and ``new_base`` are the logical-to-site endpoints produced by
    independently routing the inner and outer chunks. ``current`` is
    ``old_base`` after any MPO unswaps. Recovering that physical site
    permutation and applying it to ``new_base`` preserves every relabeling
    across the lazy-transpilation seam.
    """

    size = len(old_base)
    if len(new_base) != size or current.num_qubits != size:
        raise GeneratorError("staged boundary maps have inconsistent sizes")
    site_transform = [0] * size
    for logical in range(size):
        site_transform[old_base[logical]] = current.logical_to_site[logical]
    if sorted(site_transform) != list(range(size)):
        raise GeneratorError("staged boundary site transform is not a permutation")
    return FrameState([site_transform[new_base[q]] for q in range(size)])


def _build_region_side(
    circuit: QuantumCircuit,
    segments: list[tuple[int, int, list[int] | None]],
    *,
    invert: bool,
    seed: int,
    sabre_trials: int,
) -> tuple[list, list[int]]:
    """Concatenate independently routed segments with virtual seam layers."""

    layers_all: list = []
    previous_end: list[int] | None = None
    for start, stop, layout in segments:
        if start >= stop:
            continue
        layers, start_site_of, end_site_of = _routed_segment(
            circuit,
            start,
            stop,
            layout,
            invert=invert,
            seed=seed,
            sabre_trials=sabre_trials,
        )
        if previous_end is not None:
            seam = _seam_swap_layer(previous_end, start_site_of)
            if seam is not None:
                layers_all.append(seam)
        layers_all.extend(layers)
        previous_end = end_site_of
    if previous_end is None:
        previous_end = list(range(circuit.num_qubits))
    return layers_all, previous_end


def _initial_state(
    circuit: QuantumCircuit,
    config: dict[str, Any],
    seed: int,
    to_backend,
    counters: OrderingCounters,
    schedule: FrameSchedule | None = None,
) -> AbsorptionState:
    num_qubits = circuit.num_qubits
    order = config.get("instruction_order")
    if order is not None:
        circuit = reorder_instructions(circuit, [int(v) for v in order])
    start, center, end = resolve_absorb_window(config, len(circuit.data))

    regions = config.get("absorb_regions")
    if regions is not None:
        bounds = [(int(r["window"][0]), int(r["window"][1])) for r in regions]
        layouts = [r.get("layout") for r in regions]
        seed_index = next(
            (i for i, (a, b) in enumerate(bounds) if a <= center < b), len(bounds) - 1
        )
        seed_a, seed_b = bounds[seed_index]
        right_segments = [(center, seed_b, layouts[seed_index])] + [
            (a, b, lay)
            for (a, b), lay in zip(bounds[seed_index + 1 :], layouts[seed_index + 1 :])
        ]
        left_segments = [(seed_a, center, layouts[seed_index])] + [
            (a, b, lay)
            for (a, b), lay in zip(
                reversed(bounds[:seed_index]), reversed(layouts[:seed_index])
            )
        ]
        layers_left, left_end = _build_region_side(
            circuit,
            left_segments,
            invert=True,
            seed=seed,
            sabre_trials=config["sabre_trials"],
        )
        init_meas = _synthetic_measurement_layers(left_end)
        layers_right, right_end = _build_region_side(
            circuit,
            right_segments,
            invert=False,
            seed=seed,
            sabre_trials=config["sabre_trials"],
        )
        final_meas = _synthetic_measurement_layers(right_end)
        counters.work_ops_total, _ = _count_work_ops(merge_instructions(circuit, start, end))
        mpo = mpo_from_circuit(
            quimb_circuit_from_qiskit(QuantumCircuit(num_qubits), to_backend=to_backend)
        )
        return AbsorptionState(
            mpo=mpo,
            layers_left=layers_left,
            layers_right=layers_right,
            init_meas=init_meas,
            final_meas=final_meas,
            ii_left=0,
            ii_right=0,
            frame_left=FrameState(measurement_permutation(init_meas)),
            frame_right=FrameState(measurement_permutation(final_meas)),
            counters=counters,
            perm_left_route=list(range(num_qubits)),
            perm_right_route=list(range(num_qubits)),
        )

    if config.get("staged_transpilation"):
        staged_windows = config.get("staged_region_windows")
        if staged_windows is not None:
            windows = [(int(a), int(b)) for a, b in staged_windows]
            if windows[0][0] != start or windows[-1][1] != end:
                raise GeneratorError(
                    "staged_region_windows must cover the complete absorption window"
                )
            seed_index = next(
                (i for i, (a, b) in enumerate(windows) if a <= center < b), None
            )
            if seed_index is None or seed_index == 0 or seed_index == len(windows) - 1:
                raise GeneratorError(
                    "staged_region_windows requires center inside a non-outer stage"
                )
            if not (seed_index * 2 + 1 == len(windows)):
                raise GeneratorError(
                    "staged_region_windows must have symmetric stage count around center"
                )

            staged_layouts: dict[int, list[int]] = {}
            if schedule is not None and schedule.events:
                enters = sorted(
                    (event for event in schedule.events if event.kind == "enter"),
                    key=lambda event: event.layer,
                )
                module_stages = [
                    index
                    for index in range(1, len(windows) - 1)
                    if index != seed_index
                ]
                if len(enters) != len(module_stages):
                    raise GeneratorError(
                        "staged module-frame schedule/module-window count mismatch: "
                        f"{len(enters)} enter events for {len(module_stages)} module stages"
                    )
                staged_layouts = {
                    stage: list(event.permutation)
                    for stage, event in zip(module_stages, enters)
                }
                if not config.get("absorb_swaps_as_perm"):
                    raise GeneratorError(
                        "staged module-frame mode requires absorb_swaps_as_perm=true"
                    )

            def region_chunk(
                stage: int, a: int, b: int, *, invert: bool
            ) -> QuantumCircuit:
                result = merge_instructions(circuit, a, b)
                result = result.inverse() if invert else result
                base_layout = config.get("initial_layout")
                if base_layout is None:
                    base_layout = list(range(num_qubits))
                else:
                    base_layout = [int(value) for value in base_layout]
                sigma = staged_layouts.get(stage)
                # A validated sigma is a routing prior, not a gate rewrite:
                # place logical q where sigma[q] would sit in the base layout.
                # The stage seam and endpoint maps make this arbitrary layout
                # choice exactly equivalent to the unchanged input circuit.
                stage_layout = (
                    base_layout
                    if sigma is None
                    else [base_layout[sigma[q]] for q in range(num_qubits)]
                )
                result.metadata = {
                    **(result.metadata or {}),
                    "_staged_start_layout": stage_layout,
                    "_staged_sigma_layout": sigma is not None,
                }
                return result

            seed_a, seed_b = windows[seed_index]
            inner_left = region_chunk(seed_index, seed_a, center, invert=True)
            inner_right = region_chunk(seed_index, center, seed_b, invert=False)
            pending_left = [
                region_chunk(stage, a, b, invert=True)
                for stage, (a, b) in reversed(list(enumerate(windows[:seed_index])))
            ]
            pending_right = [
                region_chunk(stage, a, b, invert=False)
                for stage, (a, b) in enumerate(
                    windows[seed_index + 1 :], start=seed_index + 1
                )
            ]
            layers_left, init_meas, _, inner_end_left = _route_staged_chunk(
                inner_left,
                seed=seed,
                sabre_trials=config["sabre_trials"],
                layout=inner_left.metadata["_staged_start_layout"],
            )
            layers_right, final_meas, _, inner_end_right = _route_staged_chunk(
                inner_right,
                seed=seed,
                sabre_trials=config["sabre_trials"],
                layout=inner_right.metadata["_staged_start_layout"],
            )
            counters.work_ops_total, _ = _count_work_ops(
                merge_instructions(circuit, start, end)
            )
            mpo = mpo_from_circuit(
                quimb_circuit_from_qiskit(
                    QuantumCircuit(num_qubits), to_backend=to_backend
                )
            )
            return AbsorptionState(
                mpo=mpo,
                layers_left=layers_left,
                layers_right=layers_right,
                init_meas=init_meas,
                final_meas=final_meas,
                ii_left=0,
                ii_right=0,
                frame_left=FrameState(inner_end_left),
                frame_right=FrameState(inner_end_right),
                counters=counters,
                perm_left_route=list(range(num_qubits)),
                perm_right_route=list(range(num_qubits)),
                pending_outer_left=pending_left.pop(0),
                pending_outer_right=pending_right.pop(0),
                inner_end_left=inner_end_left,
                inner_end_right=inner_end_right,
                pending_stage_left=pending_left,
                pending_stage_right=pending_right,
            )
        if center - start < 2 or end - center < 2:
            raise GeneratorError(
                "staged_transpilation needs at least two prepared instructions "
                "on each side of the absorption center"
            )
        left_split = start + (center - start) // 2
        right_split = center + (end - center) // 2
        layout = config.get("initial_layout")

        def chunk(a: int, b: int, *, invert: bool) -> QuantumCircuit:
            result = merge_instructions(circuit, a, b)
            result = result.inverse() if invert else result
            result.metadata = {
                **(result.metadata or {}),
                "_staged_start_layout": (
                    list(range(num_qubits))
                    if layout is None
                    else [int(value) for value in layout]
                ),
            }
            return result

        pending_left = chunk(start, left_split, invert=True)
        inner_left = chunk(left_split, center, invert=True)
        inner_right = chunk(center, right_split, invert=False)
        pending_right = chunk(right_split, end, invert=False)

        layers_left, init_meas, _, inner_end_left = _route_staged_chunk(
            inner_left,
            seed=seed,
            sabre_trials=config["sabre_trials"],
            layout=inner_left.metadata["_staged_start_layout"],
        )
        layers_right, final_meas, _, inner_end_right = _route_staged_chunk(
            inner_right,
            seed=seed,
            sabre_trials=config["sabre_trials"],
            layout=inner_right.metadata["_staged_start_layout"],
        )
        counters.work_ops_total, _ = _count_work_ops(
            merge_instructions(circuit, start, end)
        )
        mpo = mpo_from_circuit(
            quimb_circuit_from_qiskit(
                QuantumCircuit(num_qubits), to_backend=to_backend
            )
        )
        return AbsorptionState(
            mpo=mpo,
            layers_left=layers_left,
            layers_right=layers_right,
            init_meas=init_meas,
            final_meas=final_meas,
            ii_left=0,
            ii_right=0,
            frame_left=FrameState(inner_end_left),
            frame_right=FrameState(inner_end_right),
            counters=counters,
            perm_left_route=list(range(num_qubits)),
            perm_right_route=list(range(num_qubits)),
            pending_outer_left=pending_left,
            pending_outer_right=pending_right,
            inner_end_left=inner_end_left,
            inner_end_right=inner_end_right,
        )

    circuit_left = merge_instructions(circuit, start, center)
    circuit_right = merge_instructions(circuit, center, end)
    layout = config.get("initial_layout")
    if layout is not None:
        layout = [int(v) for v in layout]
        circuit_left = relabel_circuit(circuit_left, layout)
        circuit_right = relabel_circuit(circuit_right, layout)
    circuit_left = circuit_left.inverse()
    circuit_left.measure_all()
    circuit_right.measure_all()

    layers_left = rewire_layers(
        list(iter_layers(circuit_left)),
        list(range(num_qubits)),
        seed=seed,
        sabre_trials=config["sabre_trials"],
    )
    init_meas = layers_left[-2:]
    layers_left = layers_left[:-2]
    layers_right = rewire_layers(
        list(iter_layers(circuit_right)),
        list(range(num_qubits)),
        seed=seed,
        sabre_trials=config["sabre_trials"],
    )
    final_meas = layers_right[-2:]
    layers_right = layers_right[:-2]

    counters.work_ops_total, _ = _count_work_ops(merge_instructions(circuit, start, end))
    mpo = mpo_from_circuit(
        quimb_circuit_from_qiskit(QuantumCircuit(num_qubits), to_backend=to_backend)
    )
    return AbsorptionState(
        mpo=mpo,
        layers_left=layers_left,
        layers_right=layers_right,
        init_meas=init_meas,
        final_meas=final_meas,
        ii_left=0,
        ii_right=0,
        frame_left=FrameState(measurement_permutation(init_meas)),
        frame_right=FrameState(measurement_permutation(final_meas)),
        counters=counters,
        perm_left_route=list(range(num_qubits)),
        perm_right_route=list(range(num_qubits)),
    )


def _derive_phase_params(
    circuit: QuantumCircuit, config: dict[str, Any]
) -> dict[str, list[tuple[int, dict[str, Any]]]] | None:
    """Convert ``absorb_param_schedule`` windows into per-side phase lists.

    Each side's list is ordered from the absorption center outward; entries are
    ``(cumulative_work_ops, params)`` — the side switches to ``params`` while
    its ``work_ops_absorbed_<side>`` counter is below the cumulative bound.
    Work-op counts come from the same ``_count_work_ops`` used by the loop's
    counters, so phase boundaries land on the schedule's window edges (routing
    adds only swaps, which are not work ops).
    """

    sched = config.get("absorb_param_schedule")
    if not sched:
        return None
    start, center, end = resolve_absorb_window(config, len(circuit.data))
    phases = []
    for entry in sched:
        a, b = int(entry["window"][0]), int(entry["window"][1])
        phases.append((a, b, {
            "max_bond": int(entry.get("max_bond", config["max_bond"])),
            "cutoff": float(entry.get("cutoff", config["cutoff"])),
            "unswap_threshold": float(
                entry.get("unswap_threshold", config["unswap_threshold"])
            ),
        }))
    sides: dict[str, list[tuple[int, dict[str, Any]]]] = {"left": [], "right": []}
    cum = 0
    for a, b, params in sorted(phases, key=lambda p: -p[0]):  # center -> start
        lo, hi = max(a, start), min(b, center)
        if lo >= hi:
            continue
        work, _ = _count_work_ops(merge_instructions(circuit, lo, hi))
        cum += work
        sides["left"].append((cum, params))
    cum = 0
    for a, b, params in sorted(phases, key=lambda p: p[0]):  # center -> end
        lo, hi = max(a, center), min(b, end)
        if lo >= hi:
            continue
        work, _ = _count_work_ops(merge_instructions(circuit, lo, hi))
        cum += work
        sides["right"].append((cum, params))
    return sides


def absorb(
    state: AbsorptionState,
    config: dict[str, Any],
    seed: int,
    to_backend,
    deadline: float | None,
    event: Callable[[dict[str, Any]], None],
    checkpoint_fn: Callable[[AbsorptionState, str | None], None] | None = None,
    param_phases: dict[str, list[tuple[int, dict[str, Any]]]] | None = None,
) -> AbsorptionState:
    """Run the absorption loop (both boundaries, unswap cycles, rewiring)."""

    def q2c(circuit: QuantumCircuit):
        return quimb_circuit_from_qiskit(circuit, to_backend=to_backend)

    def active_params(side: str) -> dict[str, Any] | None:
        if param_phases is None:
            return None
        lst = param_phases["left" if side == "left" else "right"]
        if not lst:
            return None
        work = (
            state.counters.work_ops_absorbed_left
            if side == "left"
            else state.counters.work_ops_absorbed_right
        )
        for bound, params in lst:
            if work < bound:
                return params
        return lst[-1][1]

    def apply_layer(mpo, circuit: QuantumCircuit, side: str):
        params = active_params(side)
        max_bond = params["max_bond"] if params else config["max_bond"]
        cutoff = params["cutoff"] if params else config["cutoff"]
        if config["gate_mpo_mode"] == "strict_chain":
            return apply_qiskit_circuit_strict_chain(
                mpo,
                circuit,
                side=side,
                max_bond=max_bond,
                cutoff=cutoff,
                compress_method=config["compress_method"],
                equalize_norms=config["equalize_norms"],
                to_backend=to_backend,
            )
        return apply_circuit(
            mpo,
            q2c(circuit),
            side=side,
            max_bond=max_bond,
            cutoff=cutoff,
            compress_method=config["compress_method"],
            equalize_norms=config["equalize_norms"],
        )

    counters = state.counters

    def make_unswap_config(max_bond: int, cutoff: float) -> UnswapConfig:
        return UnswapConfig(
            max_bond=max_bond,
            cutoff=cutoff,
            apply_cutoff=config["apply_cutoff"],
            probe_cutoff=config["probe_cutoff"],
            max_its=config["max_its"],
            hows=tuple(config["hows"]),
            equal=config["equal"],
            align_weight=config["align_weight"],
            align_protect=config["align_protect"],
            sabre_trials=config["sabre_trials"],
            compress_method=config["compress_method"],
            equalize_norms=config["equalize_norms"],
        )

    unswap_config = make_unswap_config(config["max_bond"], config["cutoff"])
    last_phase_key: tuple | None = None
    checkpoint_every_layers = int(config["checkpoint_every_layers"])
    checkpoint_every_seconds = float(config["checkpoint_every_seconds"])
    last_checkpoint_time = time.time()
    layers_since_checkpoint = 0
    # Livelock guard (legacy defect: when an unswap cycle applies no swaps and
    # both sides still exceed the threshold, the loop spins until the wall
    # deadline with no progress). After an ineffective unswap, force-absorb the
    # smaller side once so the run always advances.
    # Additional hard cap: force absorb after this many consecutive unswap cycles
    # with no absorption progress (breaks the perpetually-effective-unswap livelock
    # that the zero-swap force_absorb guard cannot catch). 0 = disabled.
    consec_unswap = 0
    cycle_cap = int(config["unswap_cycle_cap"])
    drain_phase = False
    # Starvation guard (third livelock class, found 2026-07-17 on the GPU
    # parity canary): when unswap fires between every absorb, each cycle's
    # rewiring RESETS ii and regenerates the routing prelude, so a work gate
    # sitting behind >=2 routing layers is never reached -- the loop absorbs
    # the same zero-work layer forever (observed: 3,294 identical cycles at
    # 120/122 work ops; also the tail of the first identity-center run).
    # After STARVE_CAP consecutive zero-work absorbs while work remains, find
    # the side that still contains work and DRAIN it (absorb straight through,
    # unswap suspended) until a work op lands.
    STARVE_CAP = 24
    consec_zero_work = 0
    starve_side: str | None = None
    starve_drain = 0

    def activate_staged_outer_chunks() -> None:
        """Transpile and reconnect the two raw outer chunks exactly once."""

        if state.pending_outer_left is None or state.pending_outer_right is None:
            raise GeneratorError("staged transpilation checkpoint has only one pending side")
        if state.inner_end_left is None or state.inner_end_right is None:
            raise GeneratorError("staged transpilation checkpoint lacks inner endpoint maps")

        as_perm = bool(config.get("absorb_swaps_as_perm"))
        left_layout = (state.pending_outer_left.metadata or {}).get(
            "_staged_start_layout"
        )
        right_layout = (state.pending_outer_right.metadata or {}).get(
            "_staged_start_layout"
        )
        if as_perm:
            left_layers, left_meas, left_start, left_end = _route_staged_chunk(
                state.pending_outer_left,
                seed=seed,
                sabre_trials=config["post_sabre_trials"] or config["sabre_trials"],
                layout=left_layout,
            )
            right_layers, right_meas, right_start, right_end = _route_staged_chunk(
                state.pending_outer_right,
                seed=seed,
                sabre_trials=config["post_sabre_trials"] or config["sabre_trials"],
                layout=right_layout,
            )
            left_seam = _seam_swap_layer(state.inner_end_left, left_start)
            right_seam = _seam_swap_layer(state.inner_end_right, right_start)
        else:
            # rewire_layers composes with argsort(perm). Passing the inverse of
            # logical->site therefore places outer logical wire q at exactly
            # the site reached by the completed inner stream.
            left_perm = np.argsort(state.frame_left.logical_to_site).tolist()
            right_perm = np.argsort(state.frame_right.logical_to_site).tolist()
            left_layers, left_meas, _, left_end = _route_staged_chunk(
                state.pending_outer_left,
                seed=seed,
                sabre_trials=config["post_sabre_trials"] or config["sabre_trials"],
                perm=left_perm,
            )
            right_layers, right_meas, _, right_end = _route_staged_chunk(
                state.pending_outer_right,
                seed=seed,
                sabre_trials=config["post_sabre_trials"] or config["sabre_trials"],
                perm=right_perm,
            )
            left_seam = None
            right_seam = None

        state.layers_left = ([left_seam] if left_seam is not None else []) + left_layers
        state.layers_right = ([right_seam] if right_seam is not None else []) + right_layers
        state.ii_left = 0
        state.ii_right = 0
        state.init_meas = left_meas
        state.final_meas = right_meas
        if as_perm:
            state.frame_left = _transport_boundary_frame(
                state.inner_end_left, state.frame_left, left_end
            )
            state.frame_right = _transport_boundary_frame(
                state.inner_end_right, state.frame_right, right_end
            )
        else:
            state.frame_left = FrameState(left_end)
            state.frame_right = FrameState(right_end)
        # Preserve the newly routed endpoint maps for the next deferred pair.
        # If more stages remain, leave precisely that next pair pending; the
        # main loop will activate it only after this pair drains.
        state.inner_end_left = left_end
        state.inner_end_right = right_end
        state.pending_outer_left = (
            state.pending_stage_left.pop(0) if state.pending_stage_left else None
        )
        state.pending_outer_right = (
            state.pending_stage_right.pop(0) if state.pending_stage_right else None
        )
        if (state.pending_outer_left is None) != (state.pending_outer_right is None):
            raise GeneratorError("staged routing has unmatched pending side")
        state.staged_outer_activated = True
        state.staged_activation_index += 1
        event(
            {
                "event": "staged_outer_activated",
                "stage_index": state.staged_activation_index,
                "stages_remaining": (
                    len(state.pending_stage_left)
                    + int(state.pending_outer_left is not None)
                ),
                "work_ops_absorbed_total": state.counters.work_ops_absorbed_total,
                "left_layers": len(state.layers_left),
                "right_layers": len(state.layers_right),
                "left_seam_swaps": (
                    0 if left_seam is None else _count_work_ops(left_seam)[1]
                ),
                "right_seam_swaps": (
                    0 if right_seam is None else _count_work_ops(right_seam)[1]
                ),
                "frame_left": list(state.frame_left.logical_to_site),
                "frame_right": list(state.frame_right.logical_to_site),
            }
        )

    while (
        state.ii_left < len(state.layers_left)
        or state.ii_right < len(state.layers_right)
        or state.pending_outer_left is not None
        or state.pending_outer_right is not None
    ):
        if (
            state.ii_left >= len(state.layers_left)
            and state.ii_right >= len(state.layers_right)
            and (
                state.pending_outer_left is not None
                or state.pending_outer_right is not None
            )
        ):
            # This is the durable boundary between routing stages.  Preserve
            # the absorbed MPO, both accumulated frames, and the still-raw
            # next chunks *before* transpiling them, so a difficult M0/M1
            # phase can always restart from the completed gap without replay.
            if checkpoint_fn is not None:
                checkpoint_fn(
                    state,
                    f"stage-{state.staged_activation_index}-complete",
                )
                event(
                    {
                        "event": "staged_boundary_checkpoint_saved",
                        "stage_index": state.staged_activation_index,
                        "work_ops_absorbed_total": state.counters.work_ops_absorbed_total,
                    }
                )
            activate_staged_outer_chunks()
            # Persist the reconnected state as well: M0/M1 parameter sweeps
            # must restart after the seams have been routed, not replay the
            # completed gap or spend time re-establishing its boundary frame.
            if checkpoint_fn is not None:
                checkpoint_fn(
                    state,
                    f"stage-{state.staged_activation_index}-activated",
                )
                event(
                    {
                        "event": "staged_activation_checkpoint_saved",
                        "stage_index": state.staged_activation_index,
                        "work_ops_absorbed_total": state.counters.work_ops_absorbed_total,
                    }
                )

        # Tail drain: once every WORK op is absorbed, the remaining layers are
        # zero-work (routing/measurement). Unswap cycles between them cost
        # ~bond^3 each and cannot improve the factor's content — switch to
        # absorb/unswap alternation (cap 1) so the tail drains ~6x faster.
        # (Measured: hours of tail at bond 4096 with cap 6.)
        if (not drain_phase and cycle_cap != 1
                and counters.work_ops_total > 0
                and counters.work_ops_absorbed_total >= counters.work_ops_total):
            drain_phase = True
            cycle_cap = 1
            event({"event": "tail_drain_phase", "cycle_cap": 1})
        if checkpoint_fn is not None and (
            (checkpoint_every_layers and layers_since_checkpoint >= checkpoint_every_layers)
            or (
                checkpoint_every_seconds
                and time.time() - last_checkpoint_time >= checkpoint_every_seconds
            )
        ):
            checkpoint_fn(state)
            last_checkpoint_time = time.time()
            layers_since_checkpoint = 0
        if deadline is not None and time.time() > deadline:
            counters.termination = "deadline_censored"
            counters.censored = True
            event(
                {
                    "event": "absorption_deadline",
                    "work_ops_remaining": counters.work_ops_total
                    - counters.work_ops_absorbed_total,
                }
            )
            return state

        as_perm = config.get("absorb_swaps_as_perm")
        if as_perm:
            # Eager free-layer drain (2026-07-17): a swaps-only layer under
            # virtual routing is a pure route-map update -- the MPO is
            # untouched, so absorbing it is always safe and greedy would pick
            # it anyway (its "cost" equals the current MPO). Without this, the
            # loop first PROBES the other side -- a full apply at that side's
            # bond, then discarded -- per free layer (measured: ~700 s wasted
            # per free layer at bond 512 CPU; ~100 free layers queued).
            drained = 0
            for side_layers, side_ii, side_route in (
                ("layers_left", "ii_left", "perm_left_route"),
                ("layers_right", "ii_right", "perm_right_route"),
            ):
                layers = getattr(state, side_layers)
                while getattr(state, side_ii) < len(layers):
                    layer = layers[getattr(state, side_ii)]
                    work, new_route = remap_work_layer(layer, getattr(state, side_route))
                    if work is not None:
                        break
                    setattr(state, side_route, new_route)
                    setattr(state, side_ii, getattr(state, side_ii) + 1)
                    counters.layers_absorbed += 1
                    counters.routing_swaps_absorbed += _count_work_ops(layer)[1]
                    layers_since_checkpoint += 1
                    drained += 1
            if drained:
                event({"event": "free_layers_drained", "count": drained})
                continue
        if state.ii_left < len(state.layers_left):
            if as_perm:
                work_left, _ = remap_work_layer(
                    state.layers_left[state.ii_left], state.perm_left_route
                )
                circ_left = None if work_left is None else work_left.inverse()
            else:
                circ_left = state.layers_left[state.ii_left].inverse()
            with telemetry.capture() as cap_left:
                mpo_left = (
                    state.mpo
                    if circ_left is None
                    else apply_layer(state.mpo, circ_left, side="right")
                )
            counts_left = elem_counts(mpo_left)
        else:
            mpo_left, cap_left, counts_left = None, None, float("inf")

        if state.ii_right < len(state.layers_right):
            if as_perm:
                work_right, _ = remap_work_layer(
                    state.layers_right[state.ii_right], state.perm_right_route
                )
                circ_right = work_right
            else:
                circ_right = state.layers_right[state.ii_right]
            with telemetry.capture() as cap_right:
                mpo_right = (
                    state.mpo
                    if circ_right is None
                    else apply_layer(state.mpo, circ_right, side="left")
                )
            counts_right = elem_counts(mpo_right)
        else:
            mpo_right, cap_right, counts_right = None, None, float("inf")

        work_remaining = (
            counters.work_ops_total > 0
            and counters.work_ops_absorbed_total < counters.work_ops_total
        )
        if (starve_drain == 0 and work_remaining
                and consec_zero_work >= STARVE_CAP):
            left_work = _count_work_ops_layers(state.layers_left[state.ii_left:])
            right_work = _count_work_ops_layers(state.layers_right[state.ii_right:])
            if left_work > 0 or right_work > 0:
                starve_side = "left" if left_work >= right_work else "right"
                pending = (state.layers_left[state.ii_left:] if starve_side == "left"
                           else state.layers_right[state.ii_right:])
                starve_drain = len(pending) + 4
                event({
                    "event": "starvation_guard_drain",
                    "side": starve_side,
                    "left_work_remaining": left_work,
                    "right_work_remaining": right_work,
                    "drain_budget": starve_drain,
                })

        if config.get("balanced_absorption"):
            # Symmetric mode: keep the two fronts synchronized in absorbed
            # 2q-gate count so mirror-paired gates (g on one side, its sigma-
            # partner on the other) complete together and can cancel, instead
            # of the greedy cheapest-side choice that pulls the fronts apart.
            left_avail = state.ii_left < len(state.layers_left)
            right_avail = state.ii_right < len(state.layers_right)
            if left_avail and right_avail:
                do_left = (
                    counters.work_ops_absorbed_left <= counters.work_ops_absorbed_right
                )
            else:
                do_left = left_avail
        else:
            do_left = counts_left < counts_right
        if starve_drain > 0 and starve_side is not None:
            # drain override: head straight for the starving side's work gate
            if starve_side == "left" and state.ii_left < len(state.layers_left):
                do_left = True
            elif starve_side == "right" and state.ii_right < len(state.layers_right):
                do_left = False
        chosen_counts = counts_left if do_left else counts_right
        chosen_phase = active_params("left" if do_left else "right")
        unswap_threshold = (
            chosen_phase["unswap_threshold"]
            if chosen_phase is not None
            else config["unswap_threshold"]
        )
        # Ineffective-cycle hysteresis: when a cycle applies no swaps, the MPO
        # is already at its natural size for the current material, so re-probing
        # every layer only burns wall time (measured: one ~10 s zero-swap cycle
        # per absorbed layer on S2). After such a cycle, unswap stays disarmed
        # until the tensor grows 1.5x past the size that defeated it; effective
        # cycles reset the trigger to the configured threshold, preserving the
        # P9 fire-on-inflation cadence.
        # ``unswap_hysteresis = false`` restores the frozen-solver cadence: the
        # trigger is the raw threshold on every layer, with no disarm window.
        if (
            # Tail drain: all work is absorbed, so remaining layers carry only
            # routing swaps (free map compositions under virtual routing).
            # Unswap between them cannot change the exported content, and at
            # high bond each cycle costs ~10 min — absorb straight through.
            drain_phase
            or starve_drain > 0
            or chosen_counts < unswap_threshold
            or chosen_counts < state.unswap_rearm
            or state.force_absorb
            or (cycle_cap > 0 and consec_unswap >= cycle_cap)
        ):
            if state.force_absorb and chosen_counts >= unswap_threshold:
                event({"event": "forced_absorb_after_ineffective_unswap"})
            if cycle_cap > 0 and consec_unswap >= cycle_cap:
                event({"event": "forced_absorb_after_cycle_cap", "cycles": consec_unswap})
                # Reset so unswap RESUMES after this single forced absorb. Draining
                # many layers in a row without unswapping (leaving consec_unswap
                # high) lets the bond grow to the cap and truncate hard (measured:
                # minret 41% at bond 1024). One forced absorb per cap window breaks
                # the livelock while unswap keeps the bond compressed between them.
                consec_unswap = 0
            state.force_absorb = False
            if do_left:
                state.mpo = mpo_left
                telemetry.commit(cap_left)
                layer = state.layers_left[state.ii_left]
                if as_perm:
                    _, state.perm_left_route = remap_work_layer(
                        layer, state.perm_left_route
                    )
                state.ii_left += 1
                side = "left"
            else:
                state.mpo = mpo_right
                telemetry.commit(cap_right)
                layer = state.layers_right[state.ii_right]
                if as_perm:
                    _, state.perm_right_route = remap_work_layer(
                        layer, state.perm_right_route
                    )
                state.ii_right += 1
                side = "right"
            work, swaps = _count_work_ops(layer)
            counters.work_ops_absorbed_total += work
            counters.routing_swaps_absorbed += swaps
            counters.layers_absorbed += 1
            # Reset the livelock counter only on genuine WORK-gate progress. The
            # module-center thrash interleaves zero-work swap-layer absorbs with
            # unswaps, so resetting on every absorb never lets the counter reach
            # the cap; keying it to work>0 makes the cap fire on "no work progress
            # despite unswapping" and drain the swap layers to the next work gate.
            if work > 0:
                consec_unswap = 0
                consec_zero_work = 0
                if starve_drain > 0:
                    event({"event": "starvation_guard_resolved", "side": side})
                starve_drain = 0
                starve_side = None
            else:
                if counters.work_ops_absorbed_total < counters.work_ops_total:
                    consec_zero_work += 1
                if starve_drain > 0:
                    starve_drain -= 1
                    if starve_drain == 0:
                        # budget exhausted without reaching work: stand down and
                        # let the normal loop (and a later re-trip) take over
                        event({"event": "starvation_guard_exhausted"})
                        starve_side = None
            layers_since_checkpoint += 1
            if side == "left":
                counters.work_ops_absorbed_left += work
            else:
                counters.work_ops_absorbed_right += work
            event(
                {
                    "event": "layer_absorbed",
                    "side": side,
                    "work_ops": work,
                    "routing_swaps": swaps,
                    "absorbed_total": counters.work_ops_absorbed_total,
                    "work_ops_total": counters.work_ops_total,
                    "retained_local_frobenius_log10": telemetry.retained_local_frobenius_log10(),
                    **get_tn_info(state.mpo),
                }
            )
        else:
            counters.unswap_cycles += 1
            consec_unswap += 1
            if param_phases is not None:
                # The operator spans both fronts' material: unswap at the
                # STRONGER of the two active phases so it never truncates the
                # high-fidelity region down to the cheap region's budget.
                active = [p for p in (active_params("left"), active_params("right")) if p]
                if active:
                    mb = max(p["max_bond"] for p in active)
                    co = min(p["cutoff"] for p in active)
                    key = (mb, co)
                    if key != last_phase_key:
                        unswap_config = make_unswap_config(mb, co)
                        last_phase_key = key
                        event({"event": "param_phase_unswap",
                               "max_bond": mb, "cutoff": co})
            state.mpo, cycle = unswap(
                state.mpo,
                unswap_config,
                frame_left=state.frame_left,
                frame_right=state.frame_right,
                to_backend=to_backend,
                deadline=deadline,
                event_cb=event,
            )
            counters.unswap_iterations += cycle.iterations
            counters.unswap_swaps_applied += cycle.swaps_applied
            counters.unswap_probe_batches += cycle.probe_batches

            # In permutation-absorption mode routing SWAPs have never entered
            # the MPO: ``perm_*_route`` is the open-boundary map.  Re-running
            # SABRE here discards that map and corrupts the represented
            # operator.  Instead compose the physical unswap permutation into
            # the virtual maps and continue through the existing routed layers.
            # This is exact and also avoids the swap-only rewire thrash that the
            # virtual-routing representation was designed to remove.
            if as_perm:
                before_left = list(state.perm_left_route)
                before_right = list(state.perm_right_route)
                state.perm_left_route = compose_route_with_site_permutation(
                    state.perm_left_route, cycle.perm_left
                )
                state.perm_right_route = compose_route_with_site_permutation(
                    state.perm_right_route, cycle.perm_right
                )
                event(
                    {
                        "event": "virtual_route_update",
                        "left_before": before_left,
                        "left_after": list(state.perm_left_route),
                        "right_before": before_right,
                        "right_after": list(state.perm_right_route),
                    }
                )
            # In the materialized-routing path, rewire remaining layers under
            # the cycle permutations (even after a deadline hit, so the
            # checkpointed state stays self-consistent). Routing returns the
            # explicit boundary updates via the measurement layers.
            elif state.ii_left < len(state.layers_left):
                rewired = rewire_layers(
                    state.layers_left[state.ii_left :] + state.init_meas,
                    cycle.perm_left,
                    seed=seed,
                    sabre_trials=config["post_sabre_trials"] or config["sabre_trials"],
                )
                state.init_meas = rewired[-2:]
                state.layers_left = rewired[:-2]
                previous = state.frame_left.logical_to_site
                state.frame_left = FrameState(measurement_permutation(state.init_meas))
                event(
                    {
                        "event": "routing_update",
                        "boundary": "left",
                        "logical_to_site_before": list(previous),
                        "logical_to_site_after": list(state.frame_left.logical_to_site),
                    }
                )
            elif not as_perm:
                # Side fully absorbed, but the unswap cycle may still have
                # pulled swaps out through this boundary; the measurement
                # placement and frame MUST follow or extraction contracts the
                # meas layers in a stale frame (caught by the boundary-frame
                # guard: frames differ by exactly the dropped permutation).
                state.layers_left = []
                if list(cycle.perm_left) != list(range(len(cycle.perm_left))):
                    rewired = rewire_layers(
                        state.init_meas,
                        cycle.perm_left,
                        seed=seed,
                        sabre_trials=config["post_sabre_trials"] or config["sabre_trials"],
                    )
                    state.init_meas = rewired[-2:]
                    state.frame_left = FrameState(measurement_permutation(state.init_meas))
            if not as_perm and state.ii_right < len(state.layers_right):
                rewired = rewire_layers(
                    state.layers_right[state.ii_right :] + state.final_meas,
                    cycle.perm_right,
                    seed=seed,
                    sabre_trials=config["post_sabre_trials"] or config["sabre_trials"],
                )
                state.final_meas = rewired[-2:]
                state.layers_right = rewired[:-2]
                previous = state.frame_right.logical_to_site
                state.frame_right = FrameState(measurement_permutation(state.final_meas))
                event(
                    {
                        "event": "routing_update",
                        "boundary": "right",
                        "logical_to_site_before": list(previous),
                        "logical_to_site_after": list(state.frame_right.logical_to_site),
                    }
                )
            elif not as_perm:
                # Mirror of the left-boundary case above.
                state.layers_right = []
                if list(cycle.perm_right) != list(range(len(cycle.perm_right))):
                    rewired = rewire_layers(
                        state.final_meas,
                        cycle.perm_right,
                        seed=seed,
                        sabre_trials=config["post_sabre_trials"] or config["sabre_trials"],
                    )
                    state.final_meas = rewired[-2:]
                    state.frame_right = FrameState(measurement_permutation(state.final_meas))
            if not as_perm:
                state.ii_left = 0
                state.ii_right = 0
                counters.ordering_restarts += 1
            state.force_absorb = cycle.swaps_applied == 0
            if cycle.swaps_applied == 0 and config["unswap_hysteresis"]:
                rearm = max(state.unswap_rearm, chosen_counts * 1.5)
                if rearm != state.unswap_rearm:
                    state.unswap_rearm = rearm
                    event({"event": "unswap_rearm_raised", "rearm_elems": rearm})
            else:
                state.unswap_rearm = 0.0
            if cycle.deadline_hit:
                counters.termination = "deadline_censored"
                counters.censored = True
                return state
            remaining = counters.work_ops_total - counters.work_ops_absorbed_total
            if remaining <= config["early_stopping_gates"]:
                counters.termination = "early_stopped"
                event({"event": "early_stop", "work_ops_remaining": remaining})
                break

    if counters.termination == "in_progress":
        counters.termination = "completed"
    return state


def _maxabs(tn) -> float:
    best = 0.0
    for tensor in tn.tensors:
        data = tensor.data
        array = np.asarray(data.cpu()) if hasattr(data, "cpu") else np.asarray(data)
        if array.size:
            value = float(np.abs(array).max())
            if np.isfinite(value) and value > best:
                best = value
    return best


def _renorm(mps, tag: str, event: Callable[[dict[str, Any]], None]):
    """Underflow-safe renormalization (audited d3 fix): norm() squares
    amplitudes and can underflow to exactly zero on deeply compressed states;
    fall back to a max-abs rescale (argmax is scale invariant)."""

    try:
        norm = float(abs(mps.norm()))
    except Exception as exc:
        event({"event": "renorm_norm_failed", "tag": tag, "error": repr(exc)})
        norm = float("nan")
    maxabs = _maxabs(mps)
    event({"event": "renorm", "tag": tag, "norm": norm, "maxabs": maxabs})
    if np.isfinite(norm) and norm > 0:
        try:
            mps.normalize()
            return mps
        except Exception as exc:
            event({"event": "renorm_normalize_failed", "tag": tag, "error": repr(exc)})
    if np.isfinite(maxabs) and maxabs > 0:
        tensor = mps.tensors[0]
        tensor.modify(data=tensor.data / maxabs)
    return mps


def extract_mps(
    mpo_core,
    layers_left,
    layers_right,
    *,
    max_bond: int,
    cutoff: float,
    to_backend,
    event: Callable[[dict[str, Any]], None],
    as_perm: bool = False,
    perm_left_route: list[int] | None = None,
    perm_right_route: list[int] | None = None,
    gate_mpo_mode: str = "quimb_graph",
    compress_method: str = "zipup",
    equalize_norms: bool = False,
):
    """Contract the absorbed MPO with the leftover layers into the final MPS.

    ``layers_left`` must exclude the left measurement layers; ``layers_right``
    includes the trailing measurement layers, which carry the site mapping.
    Returns ``(mps, logical_to_site)``. With ``as_perm`` the running routing
    permutations are continued through the leftover layers and the final
    ``logical_to_site`` is composed with the right-route perm (the left perm
    folds into the symmetric |0...0> input and needs no reconciliation).
    """

    def q2c(circuit: QuantumCircuit):
        return quimb_circuit_from_qiskit(circuit, to_backend=to_backend)

    def apply_to_mps(operator, mps):
        """Apply and compress through the configured bounded 1D path."""

        from quimb.tensor import tensor_network_1d_compress

        product = operator.apply(mps, compress=False, contract=True)
        mps_method = "zipup" if compress_method == "legacy_apply" else compress_method
        return tensor_network_1d_compress(
            product,
            max_bond=max_bond,
            cutoff=cutoff,
            method=mps_method,
            optimize="auto-hq",
            equalize_norms=equalize_norms,
            inplace=True,
        )

    pl = list(perm_left_route) if (as_perm and perm_left_route is not None) else None
    pr = list(perm_right_route) if (as_perm and perm_right_route is not None) else None
    pl_core = list(pl) if pl is not None else None
    # Permutation absorption can place leftover work on arbitrary site pairs.
    # Converting those layers through quimb's direct nonlocal graph leaves
    # high-rank site tensors that are not a valid MPS after application.  A
    # strict-chain gate MPO is exact and keeps extraction one-dimensional.
    strict_leftovers = as_perm or gate_mpo_mode == "strict_chain"
    num_qubits = len(mpo_core.sites)
    final_mps = mps_zero_state(num_qubits, to_backend=to_backend)
    if as_perm and pl is not None:
        # ``pl`` describes the open boundary at the absorbed core.  Extraction
        # traverses the remaining left circuit in reverse, starting at the
        # far/input end, so first advance a throwaway copy through the remaining
        # routed layers to obtain that end map.  Starting the reverse traversal
        # from the core map is wrong whenever any routing SWAP remains (and was
        # enough to change the argmax even with no physical unswap at all).
        for layer in layers_left:
            _, pl = remap_work_layer(layer, pl)
    left_layers = list(iter_layers(merge_layers(layers_left).inverse())) if layers_left else []
    for index, layer in enumerate(left_layers):
        if as_perm:
            work, pl = remap_work_layer(layer, pl)
            if work is None:
                continue
            layer_circ = work
        else:
            layer_circ = layer
        layer_mpo = (
            strict_chain_circuit_mpo(
                layer_circ,
                max_bond=max_bond,
                cutoff=cutoff,
                compress_method=compress_method,
                equalize_norms=equalize_norms,
                to_backend=to_backend,
            )
            if strict_leftovers
            else mpo_from_circuit(q2c(layer_circ))
        )
        final_mps = apply_to_mps(layer_mpo, final_mps)
        final_mps = _renorm(final_mps, f"left-{index}", event)
    if as_perm and pl_core is not None and pl != pl_core:
        raise GeneratorError(
            f"left virtual route did not unwind during extraction: {pl} != {pl_core}"
        )

    # Per-site rescale of the absorbed MPO (audited d3 fix): after thousands of
    # compressions its entries can sit near 1e-160 and underflow the norm.
    before = _maxabs(mpo_core)
    for tensor in mpo_core.tensors:
        data = tensor.data
        array = np.asarray(data.cpu()) if hasattr(data, "cpu") else np.asarray(data)
        value = float(np.abs(array).max()) if array.size else 0.0
        if np.isfinite(value) and value > 0:
            tensor.modify(data=data / value)
    event({"event": "mpo_rescale", "maxabs_before": before, "maxabs_after": _maxabs(mpo_core)})

    final_mps = apply_to_mps(mpo_core, final_mps)
    final_mps = _renorm(final_mps, "core", event)

    final_meas = []
    for index, layer in enumerate(layers_right):
        ops = dict(layer.count_ops())
        if "barrier" in ops or "measure" in ops:
            final_meas.append(layer)
        else:
            if as_perm:
                work, pr = remap_work_layer(layer, pr)
                if work is None:
                    continue
                layer_circ = work
            else:
                layer_circ = layer
            layer_mpo = (
                strict_chain_circuit_mpo(
                    layer_circ,
                    max_bond=max_bond,
                    cutoff=cutoff,
                    compress_method=compress_method,
                    equalize_norms=equalize_norms,
                    to_backend=to_backend,
                )
                if strict_leftovers
                else mpo_from_circuit(q2c(layer_circ))
            )
            final_mps = apply_to_mps(layer_mpo, final_mps)
            final_mps = _renorm(final_mps, f"right-{index}", event)
    logical_to_site = measurement_permutation(final_meas)
    if as_perm and pr is not None:
        # The MPS is in logical frame; the routed measurement expects position q
        # at MPO site pr[logical_to_site[q]].
        logical_to_site = [pr[s] for s in logical_to_site]
    return final_mps, logical_to_site


class RunDirectory:
    """Ignored on-disk record of one generator run."""

    def __init__(self, root: Path, plan_id: str, stamp: str) -> None:
        base = root / plan_id / stamp
        path = base
        suffix = 0
        while path.exists():
            suffix += 1
            path = base.with_name(f"{base.name}-{suffix}")
        self.path = path
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "checkpoints").mkdir()
        self._events = (self.path / "events.jsonl").open("a", encoding="utf-8")
        self._log = (self.path / "generator.log").open("a", encoding="utf-8")
        self._start = time.time()

    def event(self, payload: dict[str, Any]) -> None:
        record = {"elapsed_seconds": round(time.time() - self._start, 3), **payload}
        self._events.write(json.dumps(record, sort_keys=True) + "\n")
        self._events.flush()

    def log(self, message: str) -> None:
        self._log.write(message + "\n")
        self._log.flush()

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        (self.path / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def checkpoint_path(self, seed: int) -> Path:
        return self.path / "checkpoints" / f"seed{seed}.ckpt"

    def close(self) -> None:
        self._events.close()
        self._log.close()


def _candidate_rows(candidates, frame: FrameState, beam_size: int) -> list[dict[str, Any]]:
    rows = []
    for rank, (site_bits, weight) in enumerate(candidates, start=1):
        rows.append(
            {
                "rank": rank,
                "site_bits": site_bits,
                "logical_bits": frame.logical_bits_from_site_bits(site_bits),
                "engine_weight": weight,
            }
        )
    return rows


def _setup_backend(plan: GeneratorPlan) -> Backend:
    execution = plan.execution
    backend = make_backend(execution["backend"], execution["dtype"], execution["device"])
    if backend.name == "torch":
        install_torch_linalg_patch(backend)
    return backend


def _checkpoint_state_payload(
    state: AbsorptionState,
    *,
    plan: GeneratorPlan,
    config: dict[str, Any],
    input_path: Path,
    input_sha256: str,
    schedule: FrameSchedule,
    seed: int,
    remaining_seeds: list[int],
    git_sha: str | None,
) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "engine_config": config,
        "execution_config": plan.execution,
        "input_qasm_path": str(input_path),
        "input_qasm_sha256": input_sha256,
        "git_sha": git_sha,
        "package_versions": _package_versions(),
        "engine_mode": plan.mode,
        "frame_schedule": schedule.to_dict(),
        "seed": seed,
        "remaining_seeds": remaining_seeds,
        "ii_left": state.ii_left,
        "ii_right": state.ii_right,
        "counters": asdict(state.counters),
        "telemetry": telemetry.REAL.snapshot(),
        "frame_left": list(state.frame_left.logical_to_site),
        "frame_right": list(state.frame_right.logical_to_site),
        "perm_left_route": list(state.perm_left_route),
        "perm_right_route": list(state.perm_right_route),
        "force_absorb": state.force_absorb,
        "unswap_rearm": state.unswap_rearm,
        "pending_outer_left": state.pending_outer_left,
        "pending_outer_right": state.pending_outer_right,
        "inner_end_left": state.inner_end_left,
        "inner_end_right": state.inner_end_right,
        "staged_outer_activated": state.staged_outer_activated,
        "pending_stage_left": state.pending_stage_left,
        "pending_stage_right": state.pending_stage_right,
        "staged_activation_index": state.staged_activation_index,
        "layers_left": state.layers_left,
        "layers_right": state.layers_right,
        "init_meas": state.init_meas,
        "final_meas": state.final_meas,
        "mpo": state.mpo,
    }


def _run_one_ordering(
    *,
    plan: GeneratorPlan,
    config: dict[str, Any],
    circuit: QuantumCircuit | None,
    seed: int,
    remaining_seeds: list[int],
    backend: Backend,
    run_dir: RunDirectory,
    input_path: Path,
    input_sha256: str,
    schedule: FrameSchedule,
    git_sha: str | None,
    deadline: float | None,
    resumed_state: AbsorptionState | None = None,
) -> dict[str, Any]:
    to_backend = backend.to_backend()

    def event(payload: dict[str, Any]) -> None:
        run_dir.event({"seed": seed, **payload})

    telemetry.install()
    param_phases = None
    if resumed_state is None:
        telemetry.reset()
        counters = OrderingCounters(seed=seed)
        state = _initial_state(
            circuit, config, seed, to_backend, counters, schedule=schedule
        )
        start, center, end = resolve_absorb_window(config, len(circuit.data))
        param_phases = _derive_phase_params(circuit, config)
        event(
            {
                "event": "ordering_start",
                "work_ops_total": counters.work_ops_total,
                "absorb_window": [start, end],
                "absorb_center": center,
                "staged_transpilation": bool(config.get("staged_transpilation")),
                "initial_left_layers": len(state.layers_left),
                "initial_right_layers": len(state.layers_right),
                "outer_chunks_pending": state.pending_outer_left is not None,
                "frame_left": list(state.frame_left.logical_to_site),
                "frame_right": list(state.frame_right.logical_to_site),
                **(
                    {"param_phases": {
                        side: [[bound, p["max_bond"], p["cutoff"]]
                               for bound, p in lst]
                        for side, lst in param_phases.items()}}
                    if param_phases else {}
                ),
            }
        )
    else:
        state = resumed_state
        # The prepared circuit is available on resume, so rebuild the same
        # work-count phase boundaries instead of discarding a costly staged
        # checkpoint merely because M0/M1 use stronger parameters.
        if config.get("absorb_param_schedule"):
            if circuit is None:
                raise GeneratorError(
                    "absorb_param_schedule resume requires the prepared circuit"
                )
            param_phases = _derive_phase_params(circuit, config)
        event({"event": "ordering_resumed", "ii_left": state.ii_left, "ii_right": state.ii_right})

    def checkpoint_fn(current: AbsorptionState, label: str | None = None) -> None:
        payload = _checkpoint_state_payload(
            current,
            plan=plan,
            config=config,
            input_path=input_path,
            input_sha256=input_sha256,
            schedule=schedule,
            seed=seed,
            remaining_seeds=remaining_seeds,
            git_sha=git_sha,
        )
        checkpoint_path = (
            run_dir.checkpoint_path(seed)
            if label is None
            else run_dir.path / "checkpoints" / f"seed{seed}-{label}.ckpt"
        )
        save_checkpoint(checkpoint_path, payload)
        event(
            {
                "event": "checkpoint_saved",
                "checkpoint": str(checkpoint_path),
                "label": label,
                "ii_left": current.ii_left,
                "ii_right": current.ii_right,
            }
        )

    state = absorb(
        state,
        config,
        seed=seed,
        to_backend=to_backend,
        deadline=deadline,
        event=event,
        checkpoint_fn=checkpoint_fn,
        param_phases=param_phases,
    )
    checkpoint_fn(state)

    if config.get("factor_export"):
        # A factor build ends at the absorbed operator: the checkpoint above is
        # the exported artifact (MPO + boundary frames + virtual route maps).
        # Candidate extraction is meaningless for a partial-circuit factor and
        # is skipped; the manifest records completeness explicitly.
        window_complete = (
            state.ii_left >= len(state.layers_left)
            and state.ii_right >= len(state.layers_right)
            and state.pending_outer_left is None
            and state.pending_outer_right is None
        )
        manifest = {
            "plan_id": plan.plan_id,
            "seed": seed,
            "engine_mode": plan.mode,
            "input_qasm_sha256": input_sha256,
            "absorb_window": config.get("absorb_window"),
            "center_ratio": config["center_ratio"],
            "initial_layout": config.get("initial_layout"),
            "absorb_regions": config.get("absorb_regions"),
            "staged_transpilation": bool(config.get("staged_transpilation")),
            "reordered": config.get("instruction_order") is not None,
            "window_complete": window_complete,
            "layers_left_remaining": len(state.layers_left) - state.ii_left,
            "layers_right_remaining": len(state.layers_right) - state.ii_right,
            "counters": asdict(state.counters),
            "frame_left": list(state.frame_left.logical_to_site),
            "frame_right": list(state.frame_right.logical_to_site),
            "perm_left_route": list(state.perm_left_route),
            "perm_right_route": list(state.perm_right_route),
            "absorb_swaps_as_perm": bool(config.get("absorb_swaps_as_perm")),
            "bond_profile": [
                int(state.mpo.bond_size(i, i + 1)) for i in range(len(state.mpo.sites) - 1)
            ],
            **get_tn_info(state.mpo),
            "telemetry": {
                **telemetry.REAL.snapshot(),
                "retained_local_frobenius_log10": telemetry.retained_local_frobenius_log10(),
                "note": "cumulative local truncation surrogate; not physical fidelity",
            },
            "checkpoint": str(run_dir.checkpoint_path(seed)),
            "contract": (
                "dense(mpo) == P(frame_right) @ V @ P(frame_left).T in the quimb "
                "site-0-most-significant convention, where P(pi) routes logical "
                "qubit q to site pi[q], U is data[start:end] of the prepared "
                "circuit after any instruction_order reorder, V == U without "
                "initial_layout and V == P(initial_layout) @ U @ P(initial_layout).T "
                "with it. With absorb_swaps_as_perm (no-unswap regime) routing never "
                "enters the MPO: dense(mpo) == V exactly, and perm_left_route/"
                "perm_right_route carry the boundary maps. Verified in "
                "tests/engine/test_generator.py::test_factor_export_window_is_exact."
            ),
        }
        run_dir.write_json(f"factor-seed{seed}.json", manifest)
        event(
            {
                "event": "factor_exported",
                "window_complete": window_complete,
                "final_bond": int(state.mpo.max_bond()),
                "termination": state.counters.termination,
                "censored": state.counters.censored,
            }
        )
        return {
            "seed": seed,
            "engine_mode": plan.mode,
            "final_bond": int(state.mpo.max_bond()),
            "logical_to_site": [],
            "frame_left": list(state.frame_left.logical_to_site),
            "counters": asdict(state.counters),
            "telemetry": manifest["telemetry"],
            "beam_size": config["beam_size"],
            "topk": 0,
            "rank_censoring": "factor export: no candidate extraction performed",
            "candidates": [],
            "factor": {
                "window_complete": window_complete,
                "manifest": f"factor-seed{seed}.json",
            },
        }

    if state.counters.censored:
        # A wall deadline bounds absorption, not an unbounded contraction of
        # every leftover layer.  Extracting here can greatly exceed the budget;
        # for virtual nonlocal routing it was also previously unqualified.
        # Preserve the resumable state and return no candidates rather than
        # silently turning a censored diagnostic into a purported result.
        result = {
            "seed": seed,
            "engine_mode": plan.mode,
            "final_bond": int(state.mpo.max_bond()),
            "logical_to_site": [],
            "frame_left": list(state.frame_left.logical_to_site),
            "counters": asdict(state.counters),
            "telemetry": {
                **telemetry.REAL.snapshot(),
                "retained_local_frobenius_log10": telemetry.retained_local_frobenius_log10(),
                "note": "cumulative local truncation surrogate; not physical fidelity",
            },
            "beam_size": config["beam_size"],
            "topk": 0,
            "rank_censoring": "no rank bound: candidate extraction skipped after deadline",
            "candidates": [],
        }
        event(
            {
                "event": "ordering_censored_without_extraction",
                "final_bond": result["final_bond"],
                "termination": state.counters.termination,
            }
        )
        return result

    leftovers_left = (
        state.layers_left[state.ii_left :] if state.ii_left < len(state.layers_left) else []
    )
    leftovers_right = (
        state.layers_right[state.ii_right :] if state.ii_right < len(state.layers_right) else []
    )
    as_perm = config.get("absorb_swaps_as_perm")
    mps, logical_to_site = extract_mps(
        state.mpo,
        leftovers_left,
        leftovers_right + state.final_meas,
        max_bond=config["max_bond"],
        cutoff=config["final_cutoff"],
        to_backend=to_backend,
        event=event,
        as_perm=as_perm,
        perm_left_route=state.perm_left_route,
        perm_right_route=state.perm_right_route,
        gate_mpo_mode=config["gate_mpo_mode"],
        compress_method=config["compress_method"],
        equalize_norms=config["equalize_norms"],
    )
    final_frame = FrameState(logical_to_site)
    # The boundary-frame guard is a non-perm-path invariant; with as_perm the
    # deliberate composition of the right-route perm makes them differ.
    if not as_perm and list(state.frame_right.logical_to_site) != list(logical_to_site):
        raise GeneratorError(
            "boundary frame state disagrees with the measurement mapping at "
            f"extraction: {list(state.frame_right.logical_to_site)} != {logical_to_site}"
        )
    candidates = beam_search(mps, beam=config["beam_size"], k=max(8, config["topk"]))
    rows = _candidate_rows(candidates, final_frame, config["beam_size"])
    result = {
        "seed": seed,
        "engine_mode": plan.mode,
        "final_bond": int(mps.max_bond()),
        "logical_to_site": list(logical_to_site),
        "frame_left": list(state.frame_left.logical_to_site),
        "counters": asdict(state.counters),
        "telemetry": {
            **telemetry.REAL.snapshot(),
            "retained_local_frobenius_log10": telemetry.retained_local_frobenius_log10(),
            "note": "cumulative local truncation surrogate; not physical fidelity",
        },
        "beam_size": config["beam_size"],
        "topk": len(rows),
        "rank_censoring": (
            f"any bitstring absent from these {len(rows)} rows has rank "
            f">= {len(rows) + 1} (right-censored bound, not an observation)"
        ),
        "candidates": rows,
    }
    event(
        {
            "event": "ordering_extracted",
            "final_bond": result["final_bond"],
            "termination": state.counters.termination,
            "censored": state.counters.censored,
        }
    )
    return result


def _finalize_run(
    run_dir: RunDirectory, plan: GeneratorPlan, input_sha256: str, orderings: list[dict]
) -> dict[str, Any]:
    summary = {
        "plan_id": plan.plan_id,
        "instance_id": plan.instance_id,
        "engine_mode": plan.mode,
        "input_qasm_sha256": input_sha256,
        "orderings": orderings,
        "note": (
            "Blind generation output. Engine weights are relative weights within "
            "the surviving truncated state, not physical probabilities."
        ),
    }
    run_dir.write_json("candidates.json", summary)
    return summary


def run_generator(
    plan_path: str | Path,
    *,
    qasm_override: str | Path | None = None,
    out_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute a validated blind generator plan; returns the run summary."""

    from ..evidence import find_project_root

    plan = load_generator_plan(plan_path)
    if not plan.enabled:
        raise GeneratorError(
            f"plan {plan.plan_id} is preregistered but disabled "
            "(execution.enabled = false); refusing to run"
        )
    root = Path(project_root) if project_root else find_project_root()
    input_path, input_sha256 = resolve_input_qasm(plan, root, qasm_override)

    report: dict[str, Any] | None = None
    if plan.mode == "module_frame":
        report_path = Path(plan.engine["frame_report"])
        if not report_path.is_absolute():
            report_path = Path(plan_path).resolve().parent / report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        probe_circuit = QuantumCircuit.from_qasm_file(str(input_path))
        schedule = load_frame_schedule_for_plan(report, input_sha256, probe_circuit.num_qubits)
    else:
        probe_circuit = QuantumCircuit.from_qasm_file(str(input_path))
        schedule = FrameSchedule.identity(probe_circuit.num_qubits, input_sha256)

    config = effective_engine_config(plan)
    if config.get("staged_exact_module_frames"):
        if report is None or plan.mode != "module_frame":
            raise GeneratorError(
                "staged_exact_module_frames requires engine.mode='module_frame'"
            )
        circuit = prepare_exact_staged_frame_circuit(
            input_path, schedule, report, config
        )
        state_schedule = FrameSchedule.identity(
            probe_circuit.num_qubits, input_sha256
        )
    else:
        # Ordinary staged routing slices in the original prepared-circuit
        # coordinate and uses module sigmas only as optional routing priors.
        preparation_schedule = (
            FrameSchedule.identity(probe_circuit.num_qubits, input_sha256)
            if plan.mode == "module_frame" and config.get("staged_transpilation")
            else schedule
        )
        circuit = prepare_circuit(
            input_path,
            preparation_schedule,
            preserve_raw_gates=config["preserve_raw_gates"],
        )
        state_schedule = schedule
    backend = _setup_backend(plan)
    git_sha = _git_sha(root)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = RunDirectory(Path(out_root) if out_root else root / "runs", plan.plan_id, stamp)
    plan_bytes = Path(plan_path).read_bytes()
    run_dir.write_json(
        "run_manifest.json",
        {
            "plan_id": plan.plan_id,
            "plan_path": str(plan_path),
            "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "input_qasm_path": str(input_path),
            "input_qasm_sha256": input_sha256,
            "engine_mode": plan.mode,
            "backend": backend.describe(),
            "git_sha": git_sha,
            "package_versions": _package_versions(),
            "started_utc": stamp,
        },
    )
    run_dir.write_json("effective_config.json", {"engine": config, "execution": plan.execution})

    deadline = time.time() + float(plan.execution["wall_seconds"])
    seeds = list(plan.engine["ordering_seeds"])
    orderings = []
    try:
        for index, seed in enumerate(seeds):
            run_dir.log(f"ordering seed={seed} start")
            result = _run_one_ordering(
                plan=plan,
                config=config,
                circuit=circuit,
                seed=seed,
                remaining_seeds=seeds[index + 1 :],
                backend=backend,
                run_dir=run_dir,
                input_path=input_path,
                input_sha256=input_sha256,
                schedule=state_schedule,
                git_sha=git_sha,
                deadline=deadline,
            )
            orderings.append(result)
            run_dir.log(
                f"ordering seed={seed} done termination={result['counters']['termination']}"
            )
            if time.time() > deadline:
                run_dir.event({"event": "run_deadline", "completed_orderings": len(orderings)})
                break
        summary = _finalize_run(run_dir, plan, input_sha256, orderings)
        summary["run_dir"] = str(run_dir.path)
        return summary
    finally:
        run_dir.close()


def resume_generator(
    checkpoint_path: str | Path,
    *,
    plan_path: str | Path,
    qasm_override: str | Path | None = None,
    out_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resume an interrupted ordering from its checkpoint, then continue.

    The plan is revalidated and the checkpoint is rejected unless its input
    hash and engine-critical configuration match.
    """

    from ..evidence import find_project_root

    plan = load_generator_plan(plan_path)
    root = Path(project_root) if project_root else find_project_root()
    input_path, input_sha256 = resolve_input_qasm(plan, root, qasm_override)
    config = effective_engine_config(plan)
    circuit: QuantumCircuit | None = None
    if config.get("staged_exact_module_frames"):
        report_path = Path(plan.engine["frame_report"])
        if not report_path.is_absolute():
            report_path = Path(plan_path).resolve().parent / report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        probe_circuit = QuantumCircuit.from_qasm_file(str(input_path))
        validated_schedule = load_frame_schedule_for_plan(
            report, input_sha256, probe_circuit.num_qubits
        )
        circuit = prepare_exact_staged_frame_circuit(
            input_path, validated_schedule, report, config
        )
    backend = _setup_backend(plan)
    payload = load_checkpoint(
        checkpoint_path,
        expected_input_sha256=input_sha256,
        expected_config=config,
        to_backend=backend.to_backend(),
    )
    if payload["plan_id"] != plan.plan_id:
        raise CheckpointError(
            f"checkpoint belongs to plan {payload['plan_id']!r}, not {plan.plan_id!r}"
        )
    schedule = FrameSchedule.from_dict(payload["frame_schedule"])
    seed = payload["seed"]
    remaining_seeds = list(payload["remaining_seeds"])

    telemetry.install()
    telemetry.reset(
        payload["telemetry"]["log_retained_ln"],
        payload["telemetry"]["truncation_count"],
        payload["telemetry"].get(
            "svd_fallbacks", payload["telemetry"].get("svd_nonfinite_fallbacks", 0)
        ),
    )
    counters = OrderingCounters(**payload["counters"])
    counters.termination = "in_progress"
    counters.censored = False
    if config.get("absorb_swaps_as_perm") and (
        "perm_left_route" not in payload or "perm_right_route" not in payload
    ):
        raise CheckpointError(
            "permutation-absorption checkpoint lacks virtual route maps; refusing unsafe resume"
        )
    num_qubits = len(payload["frame_left"])

    def checkpoint_route(name: str) -> list[int]:
        route = list(payload.get(name, range(num_qubits)))
        if len(route) != num_qubits or sorted(route) != list(range(num_qubits)):
            raise CheckpointError(f"checkpoint contains invalid {name}: {route}")
        return route

    force_absorb = payload.get("force_absorb", False)
    if not isinstance(force_absorb, bool):
        raise CheckpointError(
            f"checkpoint contains invalid force_absorb: {force_absorb!r}"
        )
    unswap_rearm = payload.get("unswap_rearm", 0.0)
    if isinstance(unswap_rearm, bool) or not isinstance(unswap_rearm, (int, float)) or unswap_rearm < 0:
        raise CheckpointError(
            f"checkpoint contains invalid unswap_rearm: {unswap_rearm!r}"
        )

    state = AbsorptionState(
        mpo=payload["mpo"],
        layers_left=payload["layers_left"],
        layers_right=payload["layers_right"],
        init_meas=payload["init_meas"],
        final_meas=payload["final_meas"],
        ii_left=payload["ii_left"],
        ii_right=payload["ii_right"],
        frame_left=FrameState(payload["frame_left"]),
        frame_right=FrameState(payload["frame_right"]),
        counters=counters,
        perm_left_route=checkpoint_route("perm_left_route"),
        perm_right_route=checkpoint_route("perm_right_route"),
        force_absorb=force_absorb,
        unswap_rearm=float(unswap_rearm),
        pending_outer_left=payload.get("pending_outer_left"),
        pending_outer_right=payload.get("pending_outer_right"),
        inner_end_left=payload.get("inner_end_left"),
        inner_end_right=payload.get("inner_end_right"),
        staged_outer_activated=bool(payload.get("staged_outer_activated", False)),
        pending_stage_left=payload.get("pending_stage_left", []),
        pending_stage_right=payload.get("pending_stage_right", []),
        staged_activation_index=int(payload.get("staged_activation_index", 0)),
    )
    if config.get("staged_transpilation"):
        pending = (
            state.pending_outer_left is not None,
            state.pending_outer_right is not None,
        )
        if pending[0] != pending[1]:
            raise CheckpointError(
                "staged-transpilation checkpoint contains only one pending outer chunk"
            )
        if any(pending) and (
            state.inner_end_left is None or state.inner_end_right is None
        ):
            raise CheckpointError(
                "staged-transpilation checkpoint lacks inner endpoint maps"
            )
    git_sha = _git_sha(root)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-resume"
    run_dir = RunDirectory(Path(out_root) if out_root else root / "runs", plan.plan_id, stamp)
    run_dir.write_json(
        "run_manifest.json",
        {
            "plan_id": plan.plan_id,
            "resumed_from": str(checkpoint_path),
            "input_qasm_path": str(input_path),
            "input_qasm_sha256": input_sha256,
            "engine_mode": plan.mode,
            "backend": backend.describe(),
            "git_sha": git_sha,
            "package_versions": _package_versions(),
            "started_utc": stamp,
        },
    )
    run_dir.write_json("effective_config.json", {"engine": config, "execution": plan.execution})

    deadline = time.time() + float(plan.execution["wall_seconds"])
    if circuit is None:
        preparation_schedule = (
            FrameSchedule.identity(schedule.num_qubits, input_sha256)
            if plan.mode == "module_frame" and config.get("staged_transpilation")
            else schedule
        )
        circuit = prepare_circuit(
            input_path,
            preparation_schedule,
            preserve_raw_gates=config["preserve_raw_gates"],
        )
    orderings = []
    try:
        result = _run_one_ordering(
            plan=plan,
            config=config,
            # Reuse the freshly prepared circuit to reconstruct any
            # work-indexed phase schedule. The absorbed state itself comes
            # solely from the checkpoint, so this does not replay the gap or
            # any other completed region.
            circuit=circuit,
            seed=seed,
            remaining_seeds=remaining_seeds,
            backend=backend,
            run_dir=run_dir,
            input_path=input_path,
            input_sha256=input_sha256,
            schedule=schedule,
            git_sha=git_sha,
            deadline=deadline,
            resumed_state=state,
        )
        orderings.append(result)
        for index, next_seed in enumerate(remaining_seeds):
            if time.time() > deadline:
                break
            orderings.append(
                _run_one_ordering(
                    plan=plan,
                    config=config,
                    circuit=circuit,
                    seed=next_seed,
                    remaining_seeds=remaining_seeds[index + 1 :],
                    backend=backend,
                    run_dir=run_dir,
                    input_path=input_path,
                    input_sha256=input_sha256,
                    schedule=schedule,
                    git_sha=git_sha,
                    deadline=deadline,
                )
            )
        summary = _finalize_run(run_dir, plan, input_sha256, orderings)
        summary["run_dir"] = str(run_dir.path)
        return summary
    finally:
        run_dir.close()
