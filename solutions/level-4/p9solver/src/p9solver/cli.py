"""Command-line entry point for the laptop P9 solver."""

import argparse
import contextlib
from collections import Counter
import csv
import importlib.util
import io
import json
import logging
import os
import pickle
import platform
import re
import sys
import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
import qiskit
import quimb
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks

from p9solver.pipeline import mpo_compress_unswap, mpo_to_mps
from p9solver.retention import TruncationFid10Tracker
from p9solver.vendor_info import build_effective_config


DEFAULT_EXPECTED_P9 = (
    "01101110111001100000100000001010011100101101010111110111"
)



def _make_backend(spec):
    """Build the to_backend callable (same contract as the GPU engine)."""
    if spec == "numpy":
        return None
    from . import torch_hardening  # noqa: F401 -- global SVD/QR hardening
    import torch
    dtype = torch.complex64 if spec == "torch-c64" else torch.complex128
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("[backend] torch %s on %s", dtype, dev)
    def to_backend(x):
        return torch.as_tensor(x, dtype=dtype, device=dev)
    return to_backend

def parse_center(value):
    if any(ch in value for ch in ".eE"):
        return float(value)
    return int(value)


def parse_log_level(value):
    level_name = value.upper()
    if level_name not in logging._nameToLevel:
        raise argparse.ArgumentTypeError(f"invalid log level: {value}")
    return level_name


def write_rows_csv(rows, path):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        tmp_path.write_text("")
        tmp_path.replace(path)
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def write_json(path, data):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(data, handle, indent=2, default=str)
    tmp_path.replace(path)


def write_factor_bundle(path, mpo, layers_left, layers_right, factor_interface=None):
    """Atomically save the composition-grade core MPO plus residual layers.

    This mirrors the proven GPU ``mpo_dump.pkl`` artifact: an unfinished tail
    is represented exactly as left/right layer lists, rather than silently
    discarded or forced through an unrelated truncation path.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(
            {
                "mpo": mpo,
                "layers_left": layers_left,
                "layers_right": layers_right,
                "virtual_tail_frames": getattr(mpo, "_p9_tail_virtual_frames", None),
                "factor_interface": factor_interface,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp_path.replace(path)


def build_factor_interface(path, qasm_path, circuit, mpo, layers_left, layers_right):
    """Load and validate the logical-leg contract for a saved MPO factor."""
    interface = json.loads(Path(path).read_text())
    qubit_count = circuit.num_qubits
    logical_to_site = interface.get("original_logical_to_factor_site")
    if not isinstance(logical_to_site, list) or sorted(logical_to_site) != list(range(qubit_count)):
        raise ValueError(
            "factor interface must provide a permutation original_logical_to_factor_site "
            f"of 0..{qubit_count - 1}"
        )
    site_to_logical = [None] * qubit_count
    for logical, site in enumerate(logical_to_site):
        site_to_logical[site] = logical
    interface.update(
        {
            "schema_version": "p9solver.factor-interface.v1",
            "qubit_count": qubit_count,
            "qasm_filename": qasm_path.name,
            "qasm_sha256": hashlib.sha256(qasm_path.read_bytes()).hexdigest(),
            "factor_site_to_original_logical": site_to_logical,
            "mpo_site_order": list(range(qubit_count)),
            "input_frame": list(range(qubit_count)),
            "virtual_tail_frame_convention": (
                "frame[output_factor_site] = input_factor_site; apply each saved "
                "left/right tail frame at its named boundary, never as a generic gate MPO"
            ),
            "virtual_tail_frames": getattr(mpo, "_p9_tail_virtual_frames", None),
            "residual_layers": {
                "left": [dict(layer.count_ops()) for layer in layers_left],
                "right": [dict(layer.count_ops()) for layer in layers_right],
                "composition_action": "strip barriers and measurements; no residual unitary work gates",
            },
        }
    )
    return interface


def sanitize_local_metadata(value):
    if isinstance(value, dict):
        return {key: sanitize_local_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_local_metadata(item) for item in value]
    if not isinstance(value, str):
        return value

    text = value
    cwd = str(Path.cwd())
    home = str(Path.home())
    if cwd and cwd in text:
        text = text.replace(cwd, "<repo>")
    if home and home in text:
        text = text.replace(home, "~")
    text = re.sub(r"/private/var/folders/[^\s'\",)]+", "/private/var/folders/<redacted>", text)
    text = re.sub(r"node='[^']+'", "node='<redacted>'", text)
    return text


def load_diagnostic_module(name):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "diagnostics" / f"{name}.py"
    if not module_path.exists():
        module_path = Path.cwd() / "diagnostics" / f"{name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"diagnostic module not found: diagnostics/{name}.py")
    spec = importlib.util.spec_from_file_location(f"p9solver_diagnostics_{name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load diagnostics/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def summarize_rows(rows):
    stage_rows = dict(Counter(row.get("stage", "unknown") for row in rows))
    consumed = numeric_values(rows, "u_consumed_total")
    termination = next(
        (row for row in reversed(rows) if row.get("stage") == "termination"),
        None,
    )
    summary = {
        "stage_rows": stage_rows,
        "peak_max_bond": max(numeric_values(rows, "max_bond"), default=None),
        "peak_total_elems": max(numeric_values(rows, "total_elems"), default=None),
        "peak_total_shapes": max(numeric_values(rows, "total_shapes"), default=None),
        "last_work_consumed": consumed[-1] if consumed else None,
    }
    if termination is not None:
        summary["termination_reason"] = termination.get("termination_reason")
        summary["termination_detail"] = termination.get("termination_detail")
        summary["termination_unswap_cycle"] = termination.get("unswap_cycle")
    for row in reversed(rows):
        if row.get("stage") == "timing_summary":
            for key, value in row.items():
                if key.endswith("_time_s") or key == "accounted_time_s":
                    summary[key] = value
            break
    return summary


def last_value(rows, key):
    for row in reversed(rows):
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def collect_environment():
    def capture_output(fn):
        buf = io.StringIO()
        try:
            with (
                contextlib.redirect_stdout(buf),
                contextlib.redirect_stderr(buf),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore")
                fn()
        except Exception as exc:
            return f"<failed: {type(exc).__name__}: {exc}>"
        return buf.getvalue()

    env = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "qiskit": qiskit.__version__,
        "quimb": quimb.__version__,
        "numpy": np.__version__,
        "numpy_config": capture_output(np.__config__.show),
        "thread_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
            if os.environ.get(key) is not None
        },
    }
    if hasattr(np, "show_runtime"):
        env["numpy_runtime"] = capture_output(np.show_runtime)
    try:
        import numpy._core._multiarray_umath as numpy_core

        env["numpy_multiarray_umath"] = getattr(numpy_core, "__file__", None)
    except Exception as exc:
        env["numpy_multiarray_umath"] = f"<failed: {type(exc).__name__}: {exc}>"
    try:
        import scipy

        env["scipy"] = scipy.__version__
        env["scipy_config"] = capture_output(scipy.show_config)
        try:
            import scipy.linalg._flapack as scipy_flapack

            env["scipy_flapack"] = getattr(scipy_flapack, "__file__", None)
        except Exception as exc:
            env["scipy_flapack"] = f"<failed: {type(exc).__name__}: {exc}>"
    except Exception as exc:
        env["scipy"] = f"<failed: {type(exc).__name__}: {exc}>"
    return sanitize_local_metadata(env)


def summarize_truncation(rows, max_bond_limit):
    if max_bond_limit is None:
        return {}

    max_bond_limit = float(max_bond_limit)
    rows_at_max = [
        row for row in rows
        if row.get("max_bond") not in (None, "")
        and float(row.get("max_bond")) >= max_bond_limit
    ]
    probe_left_hits = sum(1 for row in rows if row.get("probe_left_hit_max_bond"))
    probe_right_hits = sum(1 for row in rows if row.get("probe_right_hit_max_bond"))
    selected_hits = sum(1 for row in rows if row.get("selected_hit_max_bond"))
    probe_left_threshold_blocks = sum(
        1 for row in rows if row.get("probe_left_over_unswap_threshold")
    )
    probe_right_threshold_blocks = sum(
        1 for row in rows if row.get("probe_right_over_unswap_threshold")
    )
    rows_at_max_by_stage = dict(Counter(row.get("stage", "unknown") for row in rows_at_max))
    first_at_max = rows_at_max[0] if rows_at_max else None
    return {
        "truncation_diagnostics": {
            "max_bond_limit": max_bond_limit,
            "rows_at_max_bond": len(rows_at_max),
            "rows_at_max_bond_by_stage": rows_at_max_by_stage,
            "absorb_probe_left_hits_max_bond": probe_left_hits,
            "absorb_probe_right_hits_max_bond": probe_right_hits,
            "selected_absorbs_hit_max_bond": selected_hits,
            "absorb_probe_left_over_unswap_threshold": probe_left_threshold_blocks,
            "absorb_probe_right_over_unswap_threshold": probe_right_threshold_blocks,
            "first_stage_at_max_bond": first_at_max.get("stage") if first_at_max else None,
            "first_unswap_cycle_at_max_bond": first_at_max.get("unswap_cycle") if first_at_max else None,
            "first_time_at_max_bond_s": first_at_max.get("time") if first_at_max else None,
        }
    }


def make_summary(
    *,
    qasm_path,
    tag,
    circuit,
    initial_ops,
    args,
    environment,
    stats,
    compress_time_s,
    run_status,
    final_fields=None,
):
    summary = {
        "qasm": str(qasm_path),
        "tag": tag,
        "num_qubits": circuit.num_qubits,
        "initial_ops": initial_ops,
        "consolidated_ops": dict(circuit.count_ops()),
        "compress_time_s": compress_time_s,
        "stats_rows": len(stats),
        "run_status": run_status,
        "environment": environment,
        "parameters": vars(args),
        "effective_config": build_effective_config(args),
    }
    if final_fields:
        summary.update(final_fields)
    summary.update(summarize_rows(stats))
    summary.update(summarize_truncation(stats, args.max_bond))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the CPU P9 peaked-circuit solver with the validated laptop "
            "configuration."
        )
    )
    parser.add_argument("--qasm", required=True, help="Input OpenQASM circuit.")
    parser.add_argument("--outdir", default="runs", help="Directory for outputs.")
    parser.add_argument("--tag", default=None, help="Run name under --outdir.")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument(
        "--track-fid10",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record retained-spectrum fid10 from the solver's existing SVDs.",
    )
    parser.add_argument(
        "--fid10-svd-driver",
        choices=("native", "gesvd"),
        default="native",
        help=(
            "SVD kernel used by the in-place fid10 tracker. Native preserves "
            "Quimb's NumPy/Accelerate production path; gesvd is the classical "
            "SciPy LAPACK variant."
        ),
    )
    parser.add_argument(
        "--expected-bitstring",
        default=DEFAULT_EXPECTED_P9,
        help="Expected P9 peak bitstring. Use empty string to disable comparison.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--numpy-seed",
        type=int,
        default=None,
        help="Seed for stochastic unswap tie-breaks; defaults to --seed.",
    )
    parser.add_argument("--max-bond", type=int, default=512)
    parser.add_argument("--cutoff", type=float, default=0.0006)
    parser.add_argument(
        "--early-stopping-gates",
        type=int,
        default=100,
        help=(
            "[peakpilot] Stop the absorb/unswap loop once at most this many work "
            "gates remain; the rest are applied during MPS extraction. Upstream "
            "CLI hard-coded 0 (never stop early); pass 0 for that legacy path."
        ),
    )
    parser.add_argument(
        "--unswap-apply-cutoff",
        type=float,
        default=None,
        help="[peakpilot port of fable fork] Two-tier unswap: apply kept swaps to the "
        "surviving lineage at this cutoff while probes keep the probe/working cutoff. "
        "Unset -> single-tier (upstream behaviour).",
    )
    parser.add_argument(
        "--swap-cutoff",
        type=float,
        default=None,
        help="[peakpilot port of fable fork] Cutoff applied ONLY to SWAP-only routing "
        "layers during absorb (work layers keep --cutoff). Unset -> --cutoff everywhere.",
    )
    parser.add_argument(
        "--absorb-swaps-as-perm",
        action="store_true",
        help="[peakpilot port of fable fork, DIAGNOSTIC] Absorb routing SWAPs as "
        "permutation updates instead of gates; work gates land on perm-mapped sites. "
        "Recorded as P9-negative (entanglement_blowup at gate 286); leftover work "
        "layers at extraction are not validated.",
    )
    parser.add_argument("--unswap-threshold", type=float, default=500000.0)
    parser.add_argument("--center-ratio", type=parse_center, default=0.5)
    parser.add_argument(
        "--staged-transpilation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Route only the two center-adjacent quarters before activating raw outer chunks.",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "torch-c64", "torch-c128"],
        default="numpy",
        help=(
            "Array backend. torch-c64/torch-c128 move every tensor op onto "
            "torch (CUDA when available -- the launch log line reports the "
            "device; abort if it says cpu when you expected cuda). c64 is the "
            "GPU winner's dtype: ~2-4x faster, fid10 telemetry unaffected."
        ),
    )
    parser.add_argument(
        "--cutoff-schedule",
        default=None,
        help=(
            "Comma list of frac:cutoff pairs, e.g. '0:6e-4,0.10:1e-4,0.60:6e-4'. "
            "The entry with the largest frac <= consumed work fraction sets the "
            "SVD cutoff -- tighten only while sigma obfuscation pairs are open. "
            "Same contract as the GPU winner's HQP_US_CUTOFF_SCHEDULE."
        ),
    )
    parser.add_argument(
        "--forced-drain-by-cost",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When a forced drain has live work on both sides, pick the cheaper "
            "side by probed element count instead of the nearer one. The drain "
            "bypasses the normal cost guards, so choosing by proximity can absorb "
            "an arbitrarily expensive gate when a cheap alternative exists."
        ),
    )
    parser.add_argument(
        "--staged-activate-per-side",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --staged-transpilation, release each outer chunk as soon as that "
            "side's inner chunk is exhausted instead of waiting for both. Prevents "
            "the smaller chunk's front from starving at distance=inf, which turns "
            "every no-progress unswap into a forced drain on the one live side."
        ),
    )
    parser.add_argument("--max-its", type=int, default=20)
    parser.add_argument("--sabre-trials", type=int, default=90)
    parser.add_argument("--post-sabre-trials", type=int, default=50)
    parser.add_argument(
        "--absorb-score",
        choices=("total_elems", "max_bond", "bond_l2", "hot_elems"),
        default="total_elems",
        help="Score used to choose whether to absorb the next left or right layer.",
    )
    parser.add_argument(
        "--route-candidates",
        type=int,
        default=1,
        help="Number of post-unswap Sabre reroutes to generate and score.",
    )
    parser.add_argument(
        "--route-score",
        choices=(
            "none",
            "static",
            "bond_profile",
            "bond_profile_swaps",
            "lookahead_total",
            "lookahead_peak",
            "lookahead_hot",
        ),
        default="none",
        help=(
            "How to choose among post-unswap reroute candidates. "
            "bond_profile scores upcoming routed gates against the current MPO bond sizes."
        ),
    )
    parser.add_argument(
        "--route-score-lookahead",
        type=int,
        default=8,
        help="Number of non-empty routed layers scored for post-unswap route selection.",
    )
    parser.add_argument(
        "--parallel-route-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate independent post-unswap route candidates concurrently when "
            "--route-candidates is greater than 1. Diagnostic option; default is off."
        ),
    )
    parser.add_argument(
        "--route-candidate-workers",
        type=int,
        default=None,
        help="Worker count for --parallel-route-candidates. Defaults to route-candidates.",
    )
    parser.add_argument(
        "--route-seed-stride",
        type=int,
        default=1009,
        help="Seed stride used to generate post-unswap reroute candidates.",
    )
    parser.add_argument(
        "--unswap-select-mode",
        choices=("bond", "bond_aligned", "bond_aligned_budget", "bond_aligned_tiebreak", "bond_route_proxy", "layer", "pair_lookahead"),
        default="bond",
        help=(
            "Swap-selection rule during unswapping. 'bond' is the verified "
            "default; 'bond_aligned', 'bond_aligned_budget', "
            "'bond_aligned_tiebreak', and "
            "'bond_route_proxy' are experimental permutation-aware variants."
        ),
    )
    parser.add_argument(
        "--unswap-alignment-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty, in bond-gain units, for unswap choices that increase the "
            "distance between the left and right boundary permutations."
        ),
    )
    parser.add_argument(
        "--unswap-alignment-target",
        default=None,
        help=(
            "Frame to pull BOTH boundary permutations toward, as a JSON list, or "
            "'identity'. Without this, --unswap-alignment-weight only pulls an "
            "object's own two frames together, which does nothing for a seam: two "
            "independently routed modules still end up random relative to each "
            "other (measured 619/664 inversions against 566 for a random 48-perm). "
            "Building every module against the SAME target makes their edges line "
            "up by construction, and the joining SWAP block costs 2^crossings, so "
            "each crossing removed halves it. A large weight approximates the hard "
            "constraint 'legs keep their original qubit labels'."
        ),
    )
    parser.add_argument(
        "--unswap-alignment-metric",
        choices=["l1", "crossings"],
        default="l1",
        help=(
            "How frame distance is scored. 'l1' is total displacement (the "
            "original). 'crossings' is the max wires crossing any cut, which is "
            "what a join actually costs: the SWAP block is 2^crossings. "
            "Minimising l1 does not minimise crossings and measurably made seams "
            "worse (k sum 52 -> 62 at weight 8) while improving retention."
        ),
    )
    parser.add_argument(
        "--unswap-alignment-protect-gain",
        type=float,
        default=None,
        help=(
            "In bond_aligned mode, always keep a swap whose local bond-size "
            "gain is at least this value, even if it worsens boundary alignment."
        ),
    )
    parser.add_argument(
        "--unswap-alignment-max-replacements",
        type=int,
        default=None,
        help=(
            "In bond_aligned_budget mode, cap how many baseline swaps may be "
            "replaced by alignment-friendlier alternatives in each parity pass."
        ),
    )
    parser.add_argument(
        "--unswap-alignment-tie-loss",
        type=float,
        default=1.0,
        help=(
            "In bond_aligned_tiebreak mode, maximum local bond-gain loss "
            "allowed when replacing a baseline swap for better alignment."
        ),
    )
    parser.add_argument(
        "--unswap-route-proxy-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty, in bond-gain units, for unswap choices that increase a "
            "cheap frontier-span estimate of the next reroute burden."
        ),
    )
    parser.add_argument(
        "--unswap-route-proxy-lookahead",
        type=int,
        default=8,
        help="Number of non-empty two-qubit frontier layers used by bond_route_proxy.",
    )
    parser.add_argument(
        "--unswap-route-proxy-include-swaps",
        action="store_true",
        help="Include routed SWAP layers in the frontier-span proxy.",
    )
    parser.add_argument(
        "--unswap-route-proxy-allow-nonbond",
        action="store_true",
        help=(
            "Let bond_route_proxy select swaps that are not immediately "
            "bond-improving when the frontier-span gain is large enough."
        ),
    )
    parser.add_argument(
        "--unswap-route-proxy-policy",
        choices=("veto", "augment", "hybrid"),
        default="veto",
        help=(
            "How bond_route_proxy modifies the bond selector: 'veto' may reject "
            "bond-improving swaps, 'augment' keeps them and adds cheap "
            "route-helpful swaps, and 'hybrid' does both."
        ),
    )
    parser.add_argument(
        "--unswap-route-proxy-min-benefit",
        type=float,
        default=1.0,
        help="Minimum frontier-span improvement required for augmenting swaps.",
    )
    parser.add_argument(
        "--unswap-route-proxy-max-bond-loss",
        type=float,
        default=0.0,
        help="Maximum local bond-size loss allowed for augmenting swaps.",
    )
    parser.add_argument(
        "--unswap-route-proxy-protect-gain",
        type=float,
        default=None,
        help=(
            "In veto/hybrid mode, always keep bond-improving swaps with at "
            "least this local bond-size gain."
        ),
    )
    parser.add_argument(
        "--unswap-route-proxy-max-cycles",
        type=int,
        default=None,
        help=(
            "Use bond_route_proxy for only this many unswap cycles, then fall "
            "back to the verified bond selector."
        ),
    )
    parser.add_argument(
        "--unswap-pair-lookahead-limit",
        type=int,
        default=8,
        help="Number of hot candidate pairs tested by pair_lookahead unswap mode.",
    )
    parser.add_argument(
        "--swap-gate-representation",
        choices=("cx", "current", "block"),
        default="current",
        help=(
            "How routed SWAP gates are converted before MPO absorption. "
            "'cx' decomposes routed and unswap-probe SWAPs to CX gates; "
            "'current' keeps routed SWAPs raw but decomposes unswap probes; "
            "'block' keeps both raw."
        ),
    )
    parser.add_argument(
        "--max-unswap-cycles",
        type=int,
        default=None,
        help="Debug option: stop early before final sampling.",
    )
    parser.add_argument(
        "--max-work-gates",
        type=int,
        default=None,
        help="Debug option: stop compression after at least this many work gates are consumed.",
    )
    parser.add_argument(
        "--abort-after-no-progress-unswap-cycles",
        type=int,
        default=2,
        help=(
            "Stop cleanly after this many consecutive unswap cycles consume "
            "zero work gates. If this trips at cutoff 0.0006, rerun the same "
            "command with --cutoff 0.001 first. Use a "
            "negative value to disable this fail-fast guardrail."
        ),
    )
    parser.add_argument(
        "--force-absorb-tail-gates",
        type=int,
        default=0,
        help=(
            "When this many or fewer work gates remain, bypass further "
            "unswapping and directly absorb the residual layers into the MPO."
        ),
    )
    parser.add_argument(
        "--virtualize-tail-swaps",
        action="store_true",
        help=(
            "For direct tail absorption, store routed SWAPs as explicit "
            "permutation frames instead of multiplying them into the MPO."
        ),
    )
    parser.add_argument(
        "--parallel-absorb-probes",
        action="store_true",
        help="Probe left and right absorption candidates concurrently.",
    )
    parser.add_argument(
        "--parallel-rewire",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run independent left/right post-unswap Sabre reroutes concurrently. "
            "Default is off; use --no-parallel-rewire for the public benchmark."
        ),
    )
    parser.add_argument(
        "--adaptive-parallel-rewire",
        action="store_true",
        help=(
            "Diagnostic mode: run an initial same-input sequential-vs-parallel "
            "rewire probe, then keep parallel rewire only if it beats the "
            "configured speedup threshold."
        ),
    )
    parser.add_argument(
        "--adaptive-parallel-rewire-min-speedup",
        type=float,
        default=1.15,
        help="Required probe speedup before adaptive parallel rewire stays enabled.",
    )
    parser.add_argument(
        "--unswap-probe-max-bond",
        type=int,
        default=None,
        help="Use this lower max bond only while probing candidate unswap layers.",
    )
    parser.add_argument(
        "--unswap-probe-cutoff",
        type=float,
        default=None,
        help="Use this cutoff only while probing candidate unswap layers.",
    )
    parser.add_argument(
        "--reuse-full-swap-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the full candidate swap probe when every probed swap is selected.",
    )
    parser.add_argument(
        "--unswap-trigger-max-bond",
        type=int,
        default=None,
        help="Trigger unswapping if the chosen absorption candidate exceeds this max bond.",
    )
    parser.add_argument(
        "--unswap-hot-bonds",
        type=int,
        default=0,
        help="Limit unswap probes to candidate pairs touching the top-k hottest current bonds.",
    )
    parser.add_argument(
        "--unswap-hot-radius",
        type=int,
        default=0,
        help="Include this radius around each hot bond when --unswap-hot-bonds is used.",
    )
    parser.add_argument(
        "--unswap-adaptive-stop-max-bond",
        type=int,
        default=None,
        help="Enable adaptive unswap stop once max bond is at or below this value.",
    )
    parser.add_argument(
        "--unswap-adaptive-stop-min-rel-improvement",
        type=float,
        default=None,
        help="Stop an unswap cycle when relative element reduction falls below this value.",
    )
    parser.add_argument(
        "--unswap-adaptive-stop-min-iteration",
        type=int,
        default=0,
        help="Minimum unswap iterations before adaptive stopping can fire.",
    )
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Compress only. Full submissions should leave sampling enabled.",
    )
    parser.add_argument(
        "--save-mpo",
        action="store_true",
        help=(
            "Serialize mpo_dump.pkl containing the final MPO and any residual "
            "left/right layers, matching the GPU factor artifact."
        ),
    )
    parser.add_argument(
        "--seed-mpo",
        default=None,
        help="Seed the absorption with a saved factor MPO instead of the identity "
             "(pipeline already supports mpo_core; this exposes it). The bundle must "
             "be a STRICT CHAIN — factors built with --virtualize-tail-swaps carry "
             "dim-2 long-range bonds that make every downstream sweep cost "
             "2^(bonds crossing a cut).",
    )
    parser.add_argument(
        "--seed-absorb-mpo",
        default=None,
        help="Comma-separated bundles absorbed into the seeded core BEFORE the loop, "
             "e.g. 'left=m0.pkl,right=m1.pkl'. Each join is followed by the normal "
             "unswap cycles, so the combined object gets compressed rather than "
             "merely truncated.",
    )
    parser.add_argument(
        "--factor-interface-json",
        type=Path,
        help=(
            "JSON contract mapping original logical qubits to factor MPO sites. "
            "When used with --save-mpo, embeds and writes a composition manifest."
        ),
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write diagnostic PNGs by default: live plot.png during compression "
            "and samples.png after sampling. Use --no-plots to disable."
        ),
    )
    parser.add_argument(
        "--plot-interval-s",
        type=float,
        default=30.0,
        help="Minimum seconds between live plot.png refreshes during compression.",
    )
    parser.add_argument(
        "--console-log-level",
        type=parse_log_level,
        default="WARNING",
        help=(
            "Minimum level for terminal logging. run.log still records INFO. "
            "Use INFO to restore verbose terminal logs."
        ),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    # The unswap equal-bond tie-breaker uses NumPy's RNG.  Seed it alongside
    # SABRE so --seed identifies the complete compression trajectory.
    np.random.seed(args.seed if args.numpy_seed is None else args.numpy_seed)
    qasm_path = Path(args.qasm)
    tag = args.tag or qasm_path.stem
    outdir = Path(args.outdir) / tag
    outdir.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging._nameToLevel[args.console_log_level])
    file_handler = logging.FileHandler(outdir / "run.log", mode="w")
    file_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=min(logging.INFO, logging._nameToLevel[args.console_log_level]),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[console_handler, file_handler],
        force=True,
    )

    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    initial_ops = dict(circuit.count_ops())
    logging.info("loaded %s: qubits=%s ops=%s", qasm_path, circuit.num_qubits, initial_ops)

    pass_manager = PassManager([
        Collect2qBlocks(),
        ConsolidateBlocks(force_consolidate=True),
    ])
    circuit = pass_manager.run(circuit)
    logging.info("after two-qubit block consolidation: ops=%s", dict(circuit.count_ops()))
    environment = collect_environment()
    # [peakpilot] effective-config block, printed before any compute.
    print("=== p9solve effective config ===\n"
          + json.dumps(build_effective_config(args), indent=2, default=str)
          + "\n=== end effective config ===", file=sys.stderr, flush=True)
    logging.info(
        "environment: python=%s qiskit=%s quimb=%s numpy=%s scipy=%s machine=%s",
        sys.version.split()[0],
        qiskit.__version__,
        quimb.__version__,
        np.__version__,
        environment.get("scipy"),
        environment.get("machine"),
    )

    stats_live = []
    fid10_tracker = (
        TruncationFid10Tracker(args.fid10_svd_driver).install()
        if args.track_fid10
        else None
    )
    compression_started = None
    plot_state = {
        "last_live_refresh": 0.0,
        "run_module": None,
        "run_module_loaded": False,
        "samples_module": None,
        "samples_module_loaded": False,
        "warned_run": False,
        "warned_samples": False,
    }

    def write_partial_summary(status):
        elapsed = (
            time.perf_counter() - compression_started
            if compression_started is not None
            else None
        )
        summary = make_summary(
            qasm_path=qasm_path,
            tag=tag,
            circuit=circuit,
            initial_ops=initial_ops,
            args=args,
            environment=environment,
            stats=stats_live,
            compress_time_s=elapsed,
            run_status=status,
        )
        summary["partial"] = True
        write_rows_csv(stats_live, outdir / "stats.csv")
        write_json(outdir / "stats.json", stats_live)
        write_json(outdir / "summary.json", summary)
        return summary

    def maybe_render_live_plot(force=False):
        if not args.plots:
            return
        now = time.perf_counter()
        interval = max(0.0, args.plot_interval_s)
        if (
            not force
            and plot_state["last_live_refresh"]
            and now - plot_state["last_live_refresh"] < interval
        ):
            return
        stats_csv = outdir / "stats.csv"
        if not stats_csv.exists():
            return
        plot_state["last_live_refresh"] = now
        if not plot_state["run_module_loaded"]:
            try:
                plot_state["run_module"] = load_diagnostic_module("plot_run")
            except Exception as exc:
                plot_state["warned_run"] = True
                logging.warning("live plot.png generation disabled: %s", exc)
            finally:
                plot_state["run_module_loaded"] = True
        module = plot_state["run_module"]
        if module is None:
            return
        try:
            module.render(stats_csv, outdir / "summary.json", outdir / "plot.png")
        except Exception as exc:
            if not plot_state["warned_run"]:
                logging.warning("live plot.png refresh failed: %s", exc)
                plot_state["warned_run"] = True

    def render_samples_plot():
        if not args.plots:
            return
        if not (outdir / "summary.json").exists():
            return
        if not plot_state["samples_module_loaded"]:
            try:
                plot_state["samples_module"] = load_diagnostic_module("plot_samples")
            except Exception as exc:
                plot_state["warned_samples"] = True
                logging.warning("samples.png generation disabled: %s", exc)
            finally:
                plot_state["samples_module_loaded"] = True
        module = plot_state["samples_module"]
        if module is None:
            return
        try:
            module.render(outdir, outdir / "samples.png")
        except SystemExit as exc:
            if not plot_state["warned_samples"]:
                logging.info("samples.png not written: %s", exc)
                plot_state["warned_samples"] = True
        except Exception as exc:
            if not plot_state["warned_samples"]:
                logging.warning("samples.png generation failed: %s", exc)
                plot_state["warned_samples"] = True

    def handle_live_stats(row):
        row["fid10"] = (
            fid10_tracker.log10_retained if fid10_tracker is not None else None
        )
        row["fid10_retained_fraction"] = (
            fid10_tracker.retained_fraction if fid10_tracker is not None else None
        )
        row["fid10_svd_events"] = fid10_tracker.events if fid10_tracker is not None else 0
        row["fid10_svd_fallbacks"] = fid10_tracker.fallbacks if fid10_tracker is not None else 0
        stats_live.append(row)
        if row.get("stage") == "termination":
            write_partial_summary("terminated")
            maybe_render_live_plot(force=True)
            return
        now = time.perf_counter()
        interval = max(0.0, args.plot_interval_s)
        plot_due = (
            not plot_state["last_live_refresh"]
            or now - plot_state["last_live_refresh"] >= interval
        )
        should_checkpoint = plot_due
        if row.get("stage") == "cycle_progress":
            elapsed_s = float(row.get("time", 0.0))
            gates_consumed = int(row.get("gates_consumed", row.get("u_consumed_total", 0)))
            total_gates = row.get("total_work_gates")
            total_gates = int(total_gates) if total_gates not in (None, "") else 0
            percent = (100.0 * gates_consumed / total_gates) if total_gates else 0.0
            print(
                f"[Cycle {row.get('unswap_cycle')}] "
                f"{gates_consumed}/{total_gates} gates ({percent:.1f} %) "
                f"after {elapsed_s:.0f} sec",
                flush=True,
            )
            should_checkpoint = True
        if should_checkpoint:
            write_partial_summary("running")
        if plot_due:
            maybe_render_live_plot(force=True)

    compression_started = time.perf_counter()
    seeded_core = None
    if args.seed_mpo:
        import pickle as _pk
        _b = _pk.load(open(args.seed_mpo, "rb"))
        seeded_core = _b["mpo"] if isinstance(_b, dict) and "mpo" in _b else _b
        seeded_core.fuse_multibonds_()
        # Seeds are shipped as numpy (c64 on disk); under a torch backend every
        # OTHER tensor in the run is a torch Tensor, and quimb's contraction
        # refuses the mix ("tensordot(): argument 'other' must be Tensor, not
        # numpy.ndarray") at the first probe. Move the seed onto the backend
        # before anything touches it.
        if args.backend != "numpy":
            seeded_core.apply_to_arrays(_make_backend(args.backend))
        _lr = sum(
            1
            for ix in seeded_core.ind_map
            if len({i for i in range(seeded_core.L) if ix in seeded_core[i].inds}) == 2
            and max(i for i in range(seeded_core.L) if ix in seeded_core[i].inds)
            - min(i for i in range(seeded_core.L) if ix in seeded_core[i].inds) > 1
        )
        logging.info(
            "[seed] core from %s: bond=%d long_range=%d",
            args.seed_mpo, seeded_core.max_bond(), _lr,
        )
        if _lr:
            logging.warning(
                "[seed] core has %d long-range bonds; sweeps will cost 2^crossings. "
                "Rebuild the factor without --virtualize-tail-swaps.", _lr
            )
        if args.seed_absorb_mpo:
            for _spec in args.seed_absorb_mpo.split(","):
                _side, _, _path = _spec.partition("=")
                _o = _pk.load(open(_path.strip(), "rb"))
                _o = _o["mpo"] if isinstance(_o, dict) and "mpo" in _o else _o
                _o.fuse_multibonds_()
                if args.backend != "numpy":
                    _o.apply_to_arrays(_make_backend(args.backend))
                logging.info("[seed] absorbing %s factor %s (bond=%d)",
                             _side.strip(), _path.strip(), _o.max_bond())
                seeded_core = _o.apply(
                    seeded_core, compress=True,
                    max_bond=args.max_bond, cutoff=args.cutoff,
                )
                logging.info("[seed] core after %s join: bond=%d",
                             _side.strip(), seeded_core.max_bond())

    seeded_frames = None
    if args.seed_mpo:
        def _bar(layers, n):
            for qc in layers or ():
                for inst in qc.data:
                    if inst.operation.name == "barrier" and len(inst.qubits) == n:
                        return [qc.find_bit(q).index for q in inst.qubits]
            return None
        _n = seeded_core.L
        _L, _R = _bar(_b.get("layers_left"), _n), _bar(_b.get("layers_right"), _n)
        _vtf = _b.get("virtual_tail_frames") if isinstance(_b, dict) else None
        if _vtf:
            # A --virtualize-tail-swaps factor obeys a different contract:
            # upper = identity, lower = lof . L, where lof is the recorded
            # left_output_frame. Handing the strict (R, L) reading to the
            # rewire misaligns it exactly as an unframed core would be.
            _lof = _vtf.get("left_output_frame") or list(range(_n))
            if _L is not None:
                _L = [_lof[x] for x in _L]
            _R = list(range(_n))
            logging.info("[seed] vperm core: using identity output frame and "
                         "lof-composed input frame")
        seeded_frames = {"L": _L, "R": _R}
        logging.info("[seed] core frames L=%s R=%s",
                     seeded_frames["L"], seeded_frames["R"])

    _alignment_target = None
    if args.unswap_alignment_target:
        _spec = args.unswap_alignment_target.strip()
        if _spec.lower() == "identity":
            _alignment_target = list(range(circuit.num_qubits))
        elif _spec.startswith("edges:"):
            # edges:<left.pkl>,<right.pkl> -- take the left target from the
            # module that precedes this one (its R frame) and the right target
            # from the module that follows (its L frame). A middle module built
            # this way meets both neighbours at a low-crossing seam, without
            # constraining those neighbours at all.
            import pickle as _p2
            _lp, _, _rp = _spec[len("edges:"):].partition(",")
            def _bar2(layers, n):
                for qc in layers or ():
                    for inst in qc.data:
                        if inst.operation.name == "barrier" and len(inst.qubits) == n:
                            return [qc.find_bit(q).index for q in inst.qubits]
                return None
            _nq = circuit.num_qubits
            _tl = _tr = None
            if _lp.strip():
                _d = _p2.load(open(_lp.strip(), "rb"))
                _tl = _bar2(_d.get("layers_right"), _nq)   # predecessor's OUTPUT
            if _rp.strip():
                _d = _p2.load(open(_rp.strip(), "rb"))
                _tr = _bar2(_d.get("layers_left"), _nq)    # successor's INPUT
            _alignment_target = {"L": _tl, "R": _tr}
        else:
            _obj = json.loads(_spec)
            _alignment_target = _obj if isinstance(_obj, dict) else list(_obj)
        logging.info("[align] target=%s weight=%s mode=%s",
                     _alignment_target, args.unswap_alignment_weight,
                     args.unswap_select_mode)

    mpo, layers_left, layers_right, stats = mpo_compress_unswap(
        circuit,
        mpo_core=seeded_core,
        mpo_core_frames=seeded_frames,
        staged_transpilation=args.staged_transpilation,
        staged_activate_per_side=args.staged_activate_per_side,
        forced_drain_by_cost=args.forced_drain_by_cost,
        cutoff_schedule=(
            [(float(f), float(c)) for f, c in
             (item.split(":") for item in args.cutoff_schedule.split(","))]
            if args.cutoff_schedule else None
        ),
        max_bond=args.max_bond,
        cutoff=args.cutoff,
        unswap_threshold=args.unswap_threshold,
        early_stopping_gates=args.early_stopping_gates,
        center_ratio=args.center_ratio,
        equal=False,
        flip_freq=None,
        max_its=args.max_its,
        to_backend=_make_backend(args.backend),
        seed=args.seed,
        hows=("both", "left", "right"),
        sabre_trials=args.sabre_trials,
        post_sabre_trials=args.post_sabre_trials,
        post_sabre_seed=None,
        sabre_heuristic="decay",
        on_stats=handle_live_stats,
        max_unswap_cycles=args.max_unswap_cycles,
        max_work_gates=args.max_work_gates,
        abort_after_no_progress_unswap_cycles=(
            None
            if args.abort_after_no_progress_unswap_cycles is not None
            and args.abort_after_no_progress_unswap_cycles < 0
            else args.abort_after_no_progress_unswap_cycles
        ),
        force_absorb_tail_gates=args.force_absorb_tail_gates,
        virtualize_tail_swaps=args.virtualize_tail_swaps,
        absorb_score=args.absorb_score,
        parallel_absorb_probes=args.parallel_absorb_probes,
        parallel_rewire=args.parallel_rewire,
        adaptive_parallel_rewire=args.adaptive_parallel_rewire,
        adaptive_parallel_rewire_min_speedup=args.adaptive_parallel_rewire_min_speedup,
        unswap_probe_max_bond=args.unswap_probe_max_bond,
        unswap_probe_cutoff=args.unswap_probe_cutoff,
        unswap_adaptive_stop_max_bond=args.unswap_adaptive_stop_max_bond,
        unswap_adaptive_stop_min_rel_improvement=args.unswap_adaptive_stop_min_rel_improvement,
        unswap_adaptive_stop_min_iteration=args.unswap_adaptive_stop_min_iteration,
        absorb_lookahead_depth=1,
        route_candidates=args.route_candidates,
        route_seed_stride=args.route_seed_stride,
        route_score=args.route_score,
        route_score_lookahead=args.route_score_lookahead,
        parallel_route_candidates=args.parallel_route_candidates,
        route_candidate_workers=args.route_candidate_workers,
        swap_apply_method="mpo",
        swap_gate_representation=args.swap_gate_representation,
        unswap_select_mode=args.unswap_select_mode,
        reuse_full_swap_probe=args.reuse_full_swap_probe,
        unswap_trigger_max_bond=args.unswap_trigger_max_bond,
        unswap_hot_bonds=args.unswap_hot_bonds,
        unswap_hot_radius=args.unswap_hot_radius,
        unswap_pair_lookahead_limit=args.unswap_pair_lookahead_limit,
        unswap_route_proxy_weight=args.unswap_route_proxy_weight,
        unswap_route_proxy_lookahead=args.unswap_route_proxy_lookahead,
        unswap_route_proxy_include_swaps=args.unswap_route_proxy_include_swaps,
        unswap_route_proxy_allow_nonbond=args.unswap_route_proxy_allow_nonbond,
        unswap_route_proxy_policy=args.unswap_route_proxy_policy,
        unswap_route_proxy_min_benefit=args.unswap_route_proxy_min_benefit,
        unswap_route_proxy_max_bond_loss=args.unswap_route_proxy_max_bond_loss,
        unswap_route_proxy_protect_gain=args.unswap_route_proxy_protect_gain,
        unswap_route_proxy_max_cycles=args.unswap_route_proxy_max_cycles,
        unswap_alignment_weight=args.unswap_alignment_weight,
        unswap_alignment_target=_alignment_target,
        unswap_alignment_metric=args.unswap_alignment_metric,
        unswap_alignment_protect_gain=args.unswap_alignment_protect_gain,
        unswap_alignment_max_replacements=args.unswap_alignment_max_replacements,
        unswap_alignment_tie_loss=args.unswap_alignment_tie_loss,
        unswap_apply_cutoff=args.unswap_apply_cutoff,
        absorb_swaps_as_perm=args.absorb_swaps_as_perm,
        swap_cutoff=args.swap_cutoff,
    )
    _route_perms = next((r for r in stats if r.get("stage") == "route_perms"), None)
    route_perm_left = _route_perms["perm_left_route"] if _route_perms else None
    route_perm_right = _route_perms["perm_right_route"] if _route_perms else None
    if fid10_tracker is not None:
        fid10_tracker.uninstall()
    compress_time = time.perf_counter() - compression_started

    write_rows_csv(stats, outdir / "stats.csv")
    write_json(outdir / "stats.json", stats)

    remaining_work_gates = int(last_value(stats, "remaining_work_gates") or 0)
    summary = make_summary(
        qasm_path=qasm_path,
        tag=tag,
        circuit=circuit,
        initial_ops=initial_ops,
        args=args,
        environment=environment,
        stats=stats,
        compress_time_s=compress_time,
        run_status="compression_complete",
        final_fields={
            "partial": False,
            "leftover_left_layers": len(layers_left),
            "leftover_right_layers": len(layers_right),
            "remaining_work_gates": remaining_work_gates,
            "final_max_bond": mpo.max_bond(),
            "final_total_elems": last_value(stats, "total_elems"),
        },
    )

    factor_interface = None
    if args.factor_interface_json:
        factor_interface = build_factor_interface(
            args.factor_interface_json, qasm_path, circuit, mpo, layers_left, layers_right
        )
        write_json(outdir / "factor_interface.json", factor_interface)

    if args.save_mpo:
        write_factor_bundle(
            outdir / "mpo_dump.pkl", mpo, layers_left, layers_right, factor_interface
        )
        summary["factor_bundle_path"] = "mpo_dump.pkl"
        summary["factor_bundle_saved"] = True
        summary["factor_bundle_remaining_work_gates"] = remaining_work_gates
        summary["factor_bundle_virtual_tail_frames"] = getattr(
            mpo, "_p9_tail_virtual_frames", None
        )
        if factor_interface is not None:
            summary["factor_interface_path"] = "factor_interface.json"

    expected = args.expected_bitstring or None
    terminal_reason = summary.get("termination_reason")
    terminal_failure = terminal_reason not in (None, "completed", "max_work_gates", "max_unswap_cycles", "early_stopping_gates")
    if args.skip_sampling or args.samples <= 0:
        summary["sampling_skipped_reason"] = "disabled"
    elif terminal_failure:
        summary["sampling_skipped_reason"] = f"terminated: {terminal_reason}"
    elif args.max_unswap_cycles is not None or args.max_work_gates is not None:
        summary["sampling_skipped_reason"] = "partial run"
    else:
        logging.info("materializing MPS for sampling")
        materialize_started = time.perf_counter()
        mps, measurement_perm = mpo_to_mps(
            mpo,
            layers_left[:-2],
            layers_right,
            cutoff=args.cutoff,
            to_backend=_make_backend(args.backend),
            route_perm_left=route_perm_left,
            route_perm_right=route_perm_right,
        )
        materialize_time = time.perf_counter() - materialize_started

        logging.info("sampling %s shots", args.samples)
        sample_started = time.perf_counter()
        raw_sample_pairs = list(mps.sample(args.samples))
        sample_time = time.perf_counter() - sample_started

        raw_samples = ["".join(str(bit) for bit in bits) for bits, _ in raw_sample_pairs]
        permuted_samples = [
            "".join(raw[index] for index in measurement_perm)
            for raw in raw_samples
        ]
        raw_counts = Counter(raw_samples)
        permuted_counts = Counter(permuted_samples)
        top_permuted = permuted_counts.most_common(10)
        predicted = top_permuted[0][0] if top_permuted else None
        peak_count = top_permuted[0][1] if top_permuted else 0

        samples_path = outdir / "samples.tsv"
        with samples_path.open("w") as handle:
            handle.write("raw\tpermuted\n")
            for raw, permuted in zip(raw_samples, permuted_samples):
                handle.write(f"{raw}\t{permuted}\n")

        summary.update({
            "materialize_time_s": materialize_time,
            "sample_time_s": sample_time,
            "sample_total_time_s": materialize_time + sample_time,
            "samples": args.samples,
            "sample_unique_raw": len(raw_counts),
            "sample_unique_permuted": len(permuted_counts),
            "sample_peak_count": peak_count,
            "sample_peak_fraction": peak_count / args.samples,
            "predicted_bitstring": predicted,
            "top_permuted_samples": top_permuted,
            "measurement_perm": list(measurement_perm),
            "samples_path": str(samples_path),
        })
        if expected is not None:
            summary["expected_bitstring"] = expected
            summary["matches_expected_bitstring"] = predicted == expected
        logging.info(
            "predicted=%s peak=%s/%s matches_expected=%s",
            predicted,
            peak_count,
            args.samples,
            summary.get("matches_expected_bitstring"),
        )

    summary["run_status"] = "complete"
    write_json(outdir / "summary.json", summary)
    maybe_render_live_plot(force=True)
    render_samples_plot()

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
