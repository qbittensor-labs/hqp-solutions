"""Loud non-finite detection (peakpilot local change).

Replaces the upstream ``nan_to_num`` sanitizing: a NaN or inf anywhere in the
tensor pipeline is a fault, never something to zero out and continue from.
The pipeline records where it is (work gates consumed so far, phase) so the
error names the site, the gate position and the array shape.
"""

_CONTEXT = {"phase": None, "work_gates_consumed": None}


def set_context(phase=None, work_gates_consumed=None):
    _CONTEXT["phase"] = phase
    _CONTEXT["work_gates_consumed"] = work_gates_consumed


def raise_nonfinite(site, x):
    shape = tuple(getattr(x, "shape", ()))
    dtype = getattr(x, "dtype", None)
    raise FloatingPointError(
        f"non-finite values (NaN/inf) at {site}: "
        f"phase={_CONTEXT['phase']}, "
        f"work_gates_consumed={_CONTEXT['work_gates_consumed']}, "
        f"shape={shape}, dtype={dtype}"
    )


def is_finite_array(x):
    """True when every entry of a numpy or torch array is finite."""
    if type(x).__module__.startswith("torch"):
        import torch
        return bool(torch.isfinite(x).all())
    import numpy as np
    return bool(np.isfinite(np.asarray(x)).all())
