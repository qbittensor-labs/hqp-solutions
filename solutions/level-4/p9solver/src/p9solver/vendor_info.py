"""Vendoring marker and effective-config block (peakpilot local change).

The effective config states every resolved parameter plus the numerical
choices that argparse does not show: SVD routine, cutoff semantics, dtype,
backend, extraction bond, non-finite policy, and the vendored commit.
"""

VENDORED_FROM = "github.com/alexgalda-m/peaked-mpo-solver (local checkout ../peaked-mpo-solver, Apache-2.0)"
VENDORED_COMMIT = "47c8057dae1809eb18a391a548362e20660221e4"
# Bump when vendor/p9solver/VENDORED.md "Local changes" gains an entry.
LOCAL_CHANGES = "peakpilot-p4"

# mpo_to_mps() default; the CLI does not override it.
EXTRACTION_MAX_BOND = 4096

# quimb 1.11.2 MatrixProductOperator.apply(compress=True) reaches the SVD split
# with cutoff_mode=4 ("rsum2") and renorm=0 (checked by
# tests/test_peakpilot_local_changes.py::test_quimb_cutoff_mode_is_rsum2).
CUTOFF_MODE = "rsum2"
CUTOFF_MEANING = (
    "quimb cutoff_mode='rsum2': at each SVD the smallest singular values are "
    "discarded while their summed squares stay below cutoff * sum(s**2) "
    "(relative discarded weight per bond), then capped at max_bond; renorm=0 "
    "(no renormalisation after truncation)"
)


def _svd_routine(args):
    if args.backend == "numpy":
        if args.track_fid10:
            if args.fid10_svd_driver == "native":
                return (
                    "retention.tracked_svd (quimb svd_truncated body, instrumented): "
                    "numpy.linalg.svd via autoray (LAPACK gesdd from the numpy build); "
                    "on LinAlgError -> scipy gesvd -> QR-preconditioned gesvd -> "
                    "1e-12 relative jitter (finite inputs only; non-finite raises)"
                )
            return (
                "retention.tracked_svd (quimb svd_truncated body, instrumented): "
                "scipy.linalg.svd lapack_driver=gesvd, same fallback chain"
            )
        return (
            "quimb svd_truncated_numpy: numba-jitted numpy.linalg.svd (gesdd), "
            "scipy gesvd fallback on ValueError"
        )
    base = (
        "torch.linalg.svd patched by torch_hardening.rsvd (CUDA driver from "
        "$P9_SVD_DRIVER/$HQP_SVD_DRIVER, default gesvd; CPU: upcast to complex128, "
        "scipy gesvd fallback)"
    )
    if args.track_fid10:
        return "retention.tracked_svd wrapping " + base
    return base


def _dtype(args):
    return {
        "numpy": "complex128 (quimb gate arrays, numpy default)",
        "torch-c64": "complex64",
        "torch-c128": "complex128",
    }[args.backend]


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def build_effective_config(args):
    device = "cpu"
    if args.backend != "numpy":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "unavailable (torch not installed)"
    return {
        "vendored_from": VENDORED_FROM,
        "vendored_commit": VENDORED_COMMIT,
        "local_changes": LOCAL_CHANGES,
        "local_changes_doc": "vendor/p9solver/VENDORED.md",
        "backend": args.backend,
        "device": device,
        "dtype": _dtype(args),
        "svd_routine": _svd_routine(args),
        "cutoff": {
            "value": args.cutoff,
            "mode": CUTOFF_MODE,
            "meaning": CUTOFF_MEANING,
            "schedule": args.cutoff_schedule,
            "unswap_probe_cutoff": args.unswap_probe_cutoff,
            "unswap_apply_cutoff": args.unswap_apply_cutoff,
            "swap_cutoff": args.swap_cutoff,
            "extraction_cutoff": args.cutoff,
        },
        "max_bond": args.max_bond,
        "extraction_max_bond": EXTRACTION_MAX_BOND,
        "early_stopping_gates": args.early_stopping_gates,
        "nonfinite_policy": "raise FloatingPointError (no nan_to_num)",
        "parameters": {k: _jsonable(v) for k, v in sorted(vars(args).items())},
    }
