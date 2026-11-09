"""Local retained-SVD-norm telemetry for compression diagnostics.

This is observational: it records the product of retained local Frobenius
fractions, conventionally shown as ``fid10 = log10(product)``. It does not
alter the solver's selected ranks or tensors.
"""

from __future__ import annotations

import math
import os
import numpy as np
from scipy import linalg as scipy_linalg
from autoray import do
from autoray.autoray import backend_like
from quimb.tensor import decomp
from quimb.tensor import tensor_core

from p9solver.nonfinite import is_finite_array, raise_nonfinite

def _as_np(x, dtype=None):
    """numpy view of x; torch tensors (cuda or cpu) detach->cpu first."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    import numpy as _np
    return _np.asarray(x, dtype=dtype) if dtype is not None else _np.asarray(x)


_ACTIVE_TRACKER = None


def capture_fid10_state():
    """Capture telemetry before a speculative MPO operation."""
    if _ACTIVE_TRACKER is None:
        return None
    return (
        _ACTIVE_TRACKER.log10_retained,
        _ACTIVE_TRACKER.events,
        _ACTIVE_TRACKER.fallbacks,
    )


def restore_fid10_state(state):
    """Restore telemetry after discarding a speculative MPO operation."""
    if state is None or _ACTIVE_TRACKER is None:
        return
    (
        _ACTIVE_TRACKER.log10_retained,
        _ACTIVE_TRACKER.events,
        _ACTIVE_TRACKER.fallbacks,
    ) = state


class TruncationFid10Tracker:
    """Track retained squared singular-value fractions across MPO splits."""

    def __init__(self, svd_driver="native"):
        self.log10_retained = 0.0
        self.events = 0
        self.fallbacks = 0
        self._original = None
        self.svd_driver = svd_driver

    @property
    def retained_fraction(self):
        return 10.0 ** self.log10_retained

    def install(self):
        global _ACTIVE_TRACKER
        if self._original is not None:
            return self
        # ``tensor_compress_bond`` reaches Quimb's SVD implementation after
        # QR/LQ reduction.  The old tracker ran a *second*, untruncated SVD on
        # copies to see the discarded tail; that doubles the expensive step
        # and can OOM at an unswap peak.  Instrument the SVD Quimb already
        # performs instead, immediately before its normal trim.
        self._original = tensor_core._SPLIT_FNS["svd"]
        original = self._original
        tracker = self

        def tracked_svd(x, cutoff=-1.0, cutoff_mode=4, max_bond=-1,
                        absorb=0, renorm=0, backend=None):
            # This is Quimb's ``svd_truncated`` body with one observation of
            # ``s`` before ``_trim_and_renorm_svd_result``.  It performs the
            # exact same one SVD as the uninstrumented solver.
            absorb_code = decomp._ABSORB_MAP[absorb]
            # [peakpilot] loud NaN: a non-finite SVD input used to reach the
            # gesvd -> QR -> jitter fallback chain (numpy) or nan_to_num
            # (torch) and come out as garbage factors. Now fatal, all backends.
            if not is_finite_array(x):
                raise_nonfinite("retention.tracked_svd (SVD input)", x)
            if type(x).__module__.startswith("torch"):
                # complex64 scale management: probe apply-chains grow norms
                # geometrically and overflow fp32 (~3.4e38), poisoning SVD with
                # inf/nan -- observed as deterministic silent deaths ~70s into
                # any real GPU workload (cuSOLVER err 4, then gesvd
                # non-convergence). Keep magnitudes bounded by persistently
                # down-scaling; global operator scale is irrelevant to the
                # kept-fraction ledger, argmax extraction, and sampling.
                import torch
                _m = float(x.abs().max())
                if _m > 1e4:
                    x = x / _m
                    tracker.rescales = getattr(tracker, "rescales", 0) + 1
            with backend_like(backend):
                try:
                    if tracker.svd_driver == "native":
                        if type(x).__module__.startswith("torch") \
                                and getattr(x, "is_cuda", False):
                            # cuSOLVER's default Jacobi SVD can emit NaN
                            # silently on non-convergence (observed: 36
                            # tainted tensors per whole-circuit run even with
                            # all inputs bounded <=1e3). The GPU engine always
                            # pinned the QR-based gesvd driver for exactly
                            # this reason (gpu_engine/fp32_patch.py).
                            import torch
                            U, s, VH = torch.linalg.svd(
                                x, full_matrices=False,
                                driver=os.environ.get(
                                    "P9_CUDA_SVD_DRIVER", "gesvd"),
                            )
                        else:
                            # Preserve Quimb's exact production path. On this
                            # laptop NumPy dispatches to Apple Accelerate
                            # LAPACK.
                            U, s, VH = do(
                                "linalg.svd", x, full_matrices=False
                            )
                    else:
                        U, s, VH = scipy_linalg.svd(
                            _as_np(x),
                            full_matrices=False,
                            check_finite=False,
                            lapack_driver="gesvd",
                        )
                except (np.linalg.LinAlgError, ValueError, RuntimeError) as err:
                    # Classical gesvd remains the convergence fallback.
                    # RuntimeError covers torch._C._LinAlgError: cuSOLVER's
                    # default driver fails to converge on ill-conditioned
                    # complex64 inputs (error code 4) -- observed on Blackwell
                    # within ~70s of any real d3 workload. The proven remedy
                    # (gpu_engine/fp32_patch.py) is the QR-based gesvd path on
                    # CPU; results are cast back to the source device/dtype so
                    # the torch network stays homogeneous.
                    is_torch = type(x).__module__.startswith("torch")
                    if not is_torch and isinstance(err, RuntimeError)                             and not isinstance(err, np.linalg.LinAlgError):
                        raise
                    try:
                        U, s, VH = scipy_linalg.svd(
                            _as_np(x, np.complex128),
                            full_matrices=False,
                            check_finite=False,
                            lapack_driver="gesvd",
                        )
                    except np.linalg.LinAlgError:
                        # gesvd itself can fail to converge on ill-conditioned
                        # blocks (fatal in seeded MPO+unswap runs). QR-precondition
                        # first — svd of the small triangular factor is far
                        # better conditioned and accuracy-preserving — then a
                        # norm-scaled 1e-12 jitter as last resort (relative
                        # perturbation far below any working cutoff).
                        xn = _as_np(x, np.complex128)
                        try:
                            Q, R = scipy_linalg.qr(xn, mode="economic")
                            U_r, s, VH = scipy_linalg.svd(
                                R, full_matrices=False, check_finite=False,
                                lapack_driver="gesvd",
                            )
                            U = Q @ U_r
                        except np.linalg.LinAlgError:
                            scale = np.linalg.norm(xn) / max(xn.size, 1)
                            rng = np.random.default_rng(0)
                            xj = xn + (1e-12 * scale) * (
                                rng.standard_normal(xn.shape)
                                + 1j * rng.standard_normal(xn.shape)
                            )
                            U, s, VH = scipy_linalg.svd(
                                xj, full_matrices=False, check_finite=False,
                                lapack_driver="gesvd",
                            )
                    if is_torch:
                        import torch
                        real = (torch.float32
                                if x.dtype == torch.complex64 else torch.float64)
                        U = torch.as_tensor(U, dtype=x.dtype, device=x.device)
                        s = torch.as_tensor(s, dtype=real, device=x.device)
                        VH = torch.as_tensor(VH, dtype=x.dtype, device=x.device)
                    tracker.fallbacks += 1
                # [peakpilot] cuSOLVER can emit NaN from finite input on
                # non-convergence; never let that through silently.
                if not is_finite_array(s):
                    raise_nonfinite("retention.tracked_svd (singular values)", x)
                full_power = float(np.vdot(_as_np(s), _as_np(s)).real)
                result = decomp._trim_and_renorm_svd_result(
                    U, s, VH, cutoff, cutoff_mode, max_bond, absorb_code, renorm
                )
            # The trimmed factors' shared dimension is the kept rank.
            if full_power > 0.0:
                left = result[0]
                rank = int(np.shape(left)[-1])
                kept = _as_np(s)[:rank]
                kept_power = float(np.vdot(kept, kept).real)
                fraction = min(1.0, max(kept_power / full_power, np.finfo(float).tiny))
                tracker.log10_retained += math.log10(fraction)
            tracker.events += 1
            return result

        tensor_core._SPLIT_FNS["svd"] = tracked_svd
        _ACTIVE_TRACKER = self
        return self

    def uninstall(self):
        global _ACTIVE_TRACKER
        if self._original is not None:
            tensor_core._SPLIT_FNS["svd"] = self._original
            self._original = None
        if _ACTIVE_TRACKER is self:
            _ACTIVE_TRACKER = None
