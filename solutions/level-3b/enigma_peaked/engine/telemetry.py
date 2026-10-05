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
import contextlib
import math
import warnings
import numpy as np

class Ledger:
    __slots__ = ('log_retained', 'truncation_count')

    def __init__(self, log_retained: float=0.0, truncation_count: int=0) -> None:
        self.log_retained = log_retained
        self.truncation_count = truncation_count

    def snapshot(self) -> dict[str, float | int]:
        return {'log_retained_ln': self.log_retained, 'truncation_count': self.truncation_count, 'svd_fallbacks': _svd_fallbacks}
REAL = Ledger()
_capture_stack: list[Ledger] = []
_installed: bool = False
_originals: dict[str, object] = {}
_svd_fallbacks: int = 0

def _record(total: float, discarded: float) -> None:
    if total > 0.0 and discarded > 0.0:
        kept = 1.0 - min(discarded / total, 1.0 - 1e-300)
        ledger = _capture_stack[-1] if _capture_stack else REAL
        ledger.log_retained += math.log(kept)
        ledger.truncation_count += 1

@contextlib.contextmanager
def capture():
    ledger = Ledger()
    _capture_stack.append(ledger)
    try:
        yield ledger
    finally:
        _capture_stack.pop()

def commit(ledger: Ledger | None) -> None:
    if ledger is None:
        return
    REAL.log_retained += ledger.log_retained
    REAL.truncation_count += ledger.truncation_count

def retained_local_frobenius_log10() -> float:
    return REAL.log_retained / math.log(10)

def reset(log_retained: float=0.0, truncation_count: int=0, svd_fallbacks: int=0) -> None:
    global _svd_fallbacks
    REAL.log_retained = log_retained
    REAL.truncation_count = truncation_count
    _svd_fallbacks = svd_fallbacks

def svd_fallback_count() -> int:
    return _svd_fallbacks

def install() -> None:
    global _installed
    if _installed:
        return
    import quimb.tensor.decomp as _qd
    original_trim = _qd._trim_and_renorm_svd_result
    original_numba = _qd.svd_truncated_numba
    original_numba_trim = _qd._trim_and_renorm_svd_result_numba
    _originals['trim'] = original_trim
    _originals['numba'] = original_numba

    def _wrapped_trim(U, s, VH, *args, **kwargs):
        result = original_trim(U, s, VH, *args, **kwargs)
        try:
            n_kept = result[0].shape[-1]
            sd = s.detach().cpu().numpy() if hasattr(s, 'detach') else np.asarray(s)
            s2 = np.abs(sd) ** 2
            _record(float(s2.sum()), float(s2[n_kept:].sum()))
        except Exception:
            pass
        return result
    import inspect
    try:
        _numba_sig = inspect.signature(getattr(original_numba, 'py_func', original_numba))
        _has_calc_error = 'calc_error' in _numba_sig.parameters
    except (TypeError, ValueError):
        _has_calc_error = True

    def _wrapped_numba_legacy(x, cutoff=-1.0, cutoff_mode=4, max_bond=-1, absorb=0, renorm=0):
        global _svd_fallbacks
        fallback_reason = None
        try:
            U0, full_s, VH0 = np.linalg.svd(x, full_matrices=False)
        except (ValueError, np.linalg.LinAlgError) as exc:
            fallback_reason = exc
        if fallback_reason is None and (not all((np.isfinite(value).all() for value in (U0, full_s, VH0)))):
            fallback_reason = ValueError('accelerated SVD returned non-finite factors')
        if fallback_reason is not None:
            import scipy.linalg as sla
            _svd_fallbacks += 1
            warnings.warn(f'Got: {fallback_reason}, falling back to scipy gesvd driver.', UserWarning, stacklevel=2)
            U0, full_s, VH0 = sla.svd(x, full_matrices=False, lapack_driver='gesvd')
        U, s, VH = original_numba_trim(U0, full_s, VH0, cutoff, cutoff_mode, max_bond, absorb, renorm)
        try:
            n_kept = U.shape[-1]
            s2 = np.abs(full_s) ** 2
            _record(float(s2.sum()), float(s2[n_kept:].sum()))
        except Exception:
            pass
        return (U, s, VH)

    def _wrapped_numba(x, cutoff=-1.0, cutoff_mode=getattr(_qd, 'cutoff_mode_rsum2', 4), max_bond=-1, absorb=getattr(_qd, 'get_Usq_sqVH', 0), renorm=0, calc_error=False):
        global _svd_fallbacks
        fallback_reason = None
        try:
            U, s, VH, error = original_numba(x, cutoff, cutoff_mode, max_bond, absorb, renorm, calc_error=True)
        except ValueError as exc:
            fallback_reason = exc
        if fallback_reason is None:
            factors = (U, s, VH)
            if not all((value is None or np.isfinite(value).all() for value in factors)):
                fallback_reason = ValueError('accelerated SVD returned non-finite factors')
        if fallback_reason is not None:
            import scipy.linalg as sla
            _svd_fallbacks += 1
            warnings.warn(f'Got: {fallback_reason}, falling back to scipy gesvd driver.', UserWarning, stacklevel=2)
            U, full_s, VH = sla.svd(x, full_matrices=False, lapack_driver='gesvd')
            U, s, VH, error = original_numba_trim(U, full_s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm, calc_error=True)
            try:
                s2 = np.abs(full_s) ** 2
                _record(float(s2.sum()), float(error) ** 2)
            except Exception:
                pass
            return (U, s, VH, error if calc_error else None)
        try:
            if error is not None and error > 0.0:
                total = float(np.sum(np.abs(x) ** 2))
                _record(total, float(error) ** 2)
        except Exception:
            pass
        return (U, s, VH, error if calc_error else None)
    _qd._trim_and_renorm_svd_result = _wrapped_trim
    _qd.svd_truncated_numba = _wrapped_numba if _has_calc_error else _wrapped_numba_legacy
    _installed = True

def uninstall() -> None:
    global _installed
    if not _installed:
        return
    import quimb.tensor.decomp as _qd
    _qd._trim_and_renorm_svd_result = _originals.pop('trim')
    _qd.svd_truncated_numba = _originals.pop('numba')
    _installed = False
