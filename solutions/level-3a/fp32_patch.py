# Copyright (C) 2026 qBitTensor Labs.
# Original author: Charlie (Enigma / Hardening Quantum Proof competition).
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

"""Robust SVD/QR patch — replaces hqp_v3_solution/fp32_patch.py (mount over /app/fp32_patch.py).

Bug in the original: the complex64 path did

    try:    _svd(A, driver='gesvd')
    except: _svd(A.to(complex128), driver='gesvd')

i.e. the fallback retries THE SAME DRIVER that just failed, only in higher precision. cuSOLVER's
gesvd failure here is a dimension/workspace limit (CUSOLVER_STATUS_INVALID_VALUE from
gesvd_bufferSize), not a precision problem, so the retry fails identically and the exception
escapes. That killed the cutoff=6e-4 run outright — a "fallback" that never changed the failing
condition. Same family as the other fail-quiet boundaries: the recovery mechanism exists but
does not recover.

This version cascades through genuinely DIFFERENT strategies and says out loud which one it
used, so a degraded path can never be mistaken for a clean one.
"""
import os
import torch, autoray
import quimb.tensor.decomp as _qd

# Primary GPU SVD strategy (the v7 speed lever). Profiling the D-measurement showed ~79% of its wall
# inside cuSOLVER gesvd on SMALL matrices (bond 64-512), where gesvd is latency-bound. Candidates:
#   gesvd   -- unchanged default (QR-based bidiagonalisation)
#   gesvdj  -- cuSOLVER Jacobi; no condition-number squaring, same accuracy class as gesvd
#   eigh128 -- Gram eigh trick with the Gram FORMED AND DIAGONALISED IN complex128. The complex64
#              eigh trick (fast_svd.py) is NOT safe here: quimb truncates by rsum2 at 2e-4 and the
#              fp32 Gram's eigenvalue error summed over ~512 values is ~6e-5, i.e. ~30% of that
#              budget. In complex128 the error is ~1e-16 per value.
# Every candidate falls through to the unchanged robust cascade below on any failure.
SVD_MODE = os.environ.get("HQP_SVD_MODE", "gesvd").strip().lower()
_EIGH_KEEP = float(os.environ.get("HQP_EIGH_KEEP", "1e-7"))     # zero columns with s < this*s_max
#   hybrid  -- CPU LAPACK below HQP_SVD_CPU_MAXDIM, GPU gesvd at/above it. MEASURED on a captured
#              D-measurement workload (147K calls, 75% with min-dim < 32): CPU is 5.7x faster <32,
#              5.9x at 32-64, 3.1x at 64-128, 2.0x at 128-256, SLOWER from 256 up; 0 rsum2 truncation
#              mismatches at 6e-4/2e-4 vs a complex128 reference. The GPU cost of a tiny SVD is pure
#              launch/sync latency, so the CPU skips it entirely.
_CPU_MAXDIM = int(os.environ.get("HQP_SVD_CPU_MAXDIM", "256"))
# Threads for the CPU SVD path. Small SVDs are fastest single/dual-threaded (measured 1 thr best <256),
# and the validator caps the container at 24 CPUs on hosts that may have many more cores, where
# torch's default (= host cores) would oversubscribe the quota.
_CPU_THREADS = int(os.environ.get("HQP_SVD_CPU_THREADS", "2"))
_threads_set = [False]

def _sgn(x):
    xp = _qd.get_namespace(x); ax = xp.abs(x); x0 = ax < 1e-12
    return (x + x0) / (ax + x0)
_qd.sgn = _sgn

_svd = torch.linalg.svd
_qr = torch.linalg.qr
_STATS = {}

def _report(tag):
    _STATS[tag] = _STATS.get(tag, 0) + 1
    if _STATS[tag] in (1, 10, 100, 1000):
        print(f"  [svd] using fallback '{tag}' (occurrence #{_STATS[tag]})", flush=True)

def _eigh128(A):
    """Truncation-grade SVD via the Gram matrix in complex128. None if the result is not finite."""
    A2 = A.to(torch.complex128)
    a, b = A2.shape
    if a >= b:
        w, V = torch.linalg.eigh(A2.mH @ A2)
        w = w.flip(0).clamp_min(0); V = V.flip(1); s = w.sqrt()
        keep = s > _EIGH_KEEP * s[0].clamp_min(1e-300)
        inv = torch.where(keep, 1.0 / s.clamp_min(1e-300), torch.zeros_like(s))
        U = (A2 @ V) * inv.to(A2.dtype); Vh = V.mH
    else:
        w, U = torch.linalg.eigh(A2 @ A2.mH)
        w = w.flip(0).clamp_min(0); U = U.flip(1); s = w.sqrt()
        keep = s > _EIGH_KEEP * s[0].clamp_min(1e-300)
        inv = torch.where(keep, 1.0 / s.clamp_min(1e-300), torch.zeros_like(s))
        Vh = (U.mH @ A2) * inv.to(A2.dtype).unsqueeze(1)
    s = torch.where(keep, s, torch.zeros_like(s))
    if not (torch.isfinite(s).all() and torch.isfinite(U.real).all() and torch.isfinite(Vh.real).all()):
        return None
    return U.to(A.dtype), s.to(A.real.dtype), Vh.to(A.dtype)

def rsvd(A, full_matrices=False, **k):
    cuda = A.is_cuda
    c64 = A.dtype == torch.complex64
    # 0) optional faster primary (HQP_SVD_MODE); any failure drops to the unchanged cascade
    if cuda and SVD_MODE == 'gesvdj':
        try:
            return _svd(A, full_matrices=False, driver='gesvdj')
        except Exception:
            _report('gesvdj->cascade')
    elif cuda and SVD_MODE == 'hybrid' and A.ndim == 2 and min(A.shape) < _CPU_MAXDIM:
        try:
            if not _threads_set[0]:
                torch.set_num_threads(max(1, _CPU_THREADS)); _threads_set[0] = True
            U, s, Vh = _svd(A.detach().cpu(), full_matrices=False)
            return U.to(A.device), s.to(A.device), Vh.to(A.device)
        except Exception:
            _report('hybrid-cpu->cascade')
    elif cuda and SVD_MODE == 'eigh128' and A.ndim == 2:
        try:
            r = _eigh128(A)
            if r is not None:
                return r
            _report('eigh128-nonfinite->cascade')
        except Exception:
            _report('eigh128->cascade')
    # 1) gesvd: robust QR-based path, correct for most sizes
    if cuda:
        try:
            return _svd(A, full_matrices=False, driver='gesvd')
        except Exception:
            pass
        # 2) let cuSOLVER/torch choose (gesvdj etc.) -- different code path, different limits
        try:
            _report('default-driver')
            return _svd(A, full_matrices=False)
        except Exception:
            pass
        # 3) gesvda: approximate, designed for tall-skinny, different workspace rules
        try:
            _report('gesvda')
            return _svd(A, full_matrices=False, driver='gesvda')
        except Exception:
            pass
        # 4) higher precision, default driver (precision AND driver both change)
        try:
            _report('c128-default')
            U, s, Vh = _svd(A.to(torch.complex128), full_matrices=False)
            return (U.to(A.dtype), s.to(torch.float32 if c64 else torch.float64), Vh.to(A.dtype))
        except Exception:
            pass
        # 5) CPU LAPACK: slow but has no GPU workspace limit. Never silent.
        _report('CPU-LAPACK (slow; GPU SVD exhausted)')
        Ac = A.detach().to('cpu', torch.complex128)
        U, s, Vh = _svd(Ac, full_matrices=False)
        return (U.to(A.device, A.dtype), s.to(A.device, torch.float32 if c64 else torch.float64),
                Vh.to(A.device, A.dtype))
    # CPU tensors (D-measurement on the CPU backend: MEASURED 2026-09-19 ~10x faster than the GPU path while
    # the MPO bond is <= 256, because nothing pays kernel-launch/sync latency). LAPACK gesdd can fail to
    # converge on ill-conditioned inputs (it did, 45 s into d3_s1 B2) -> same idea as the GPU cascade:
    # change precision, then change the algorithm (gesvd = QR iteration, no divide-and-conquer).
    # CPU SVD is done in complex128 (HQP_CPU_SVD_C128=1): the matrices that live on the CPU are tiny (bond <= 64),
    # and complex64 LAPACK gesdd there returned non-finite factors WITHOUT raising -- the NaNs surfaced several
    # steps later as "array must not contain infs or NaNs" and killed a D worker at 658 s (d3_s2 B2, 2026-09-19).
    if not torch.isfinite(A).all():
        raise FloatingPointError("rsvd: non-finite input matrix (upstream NaN/Inf)")
    if os.environ.get("HQP_CPU_SVD_C128", "1") == "1" and A.dtype == torch.complex64:
        try:
            U, s, Vh = _svd(A.to(torch.complex128), full_matrices=False)
            if torch.isfinite(U).all() and torch.isfinite(s).all() and torch.isfinite(Vh).all():
                return U.to(A.dtype), s.to(torch.float32), Vh.to(A.dtype)
            _report('cpu-c128-nonfinite')
        except Exception:
            _report('cpu-c128-failed')
    else:
        try:
            U, s, Vh = _svd(A, full_matrices=False)
            if torch.isfinite(U).all() and torch.isfinite(s).all() and torch.isfinite(Vh).all():
                return U, s, Vh
            _report('cpu-nonfinite')
        except Exception:
            pass
    try:
        _report('cpu-c128')
        U, s, Vh = _svd(A.to(torch.complex128), full_matrices=False)
        return (U.to(A.dtype), s.to(torch.float32 if c64 else torch.float64), Vh.to(A.dtype))
    except Exception:
        pass
    _report('cpu-scipy-gesvd')
    import scipy.linalg as _sl
    U, s, Vh = _sl.svd(A.detach().to(torch.complex128).numpy(), full_matrices=False, lapack_driver='gesvd')
    return (torch.from_numpy(U).to(A.dtype), torch.from_numpy(s).to(torch.float32 if c64 else torch.float64),
            torch.from_numpy(Vh).to(A.dtype))

def rqr(A, *a, **k):
    if not A.is_cuda and A.dtype == torch.complex64 and os.environ.get("HQP_CPU_SVD_C128", "1") == "1":
        Q, R = _qr(A.to(torch.complex128), *a, **k)             # same policy as the CPU SVD: tiny matrices, do it in c128
        return Q.to(torch.complex64), R.to(torch.complex64)
    try:
        return _qr(A, *a, **k)
    except Exception:
        if A.dtype == torch.complex64:
            _report('qr-c128')
            Q, R = _qr(A.to(torch.complex128), *a, **k)
            return Q.to(torch.complex64), R.to(torch.complex64)
        _report('qr-cpu')
        Q, R = _qr(A.detach().cpu(), *a, **k)
        return Q.to(A.device), R.to(A.device)

torch.linalg.svd = rsvd
torch.linalg.qr = rqr
autoray.register_function('torch', 'linalg.svd', rsvd)
autoray.register_function('torch', 'linalg.qr', rqr)

import atexit
@atexit.register
def _summary():
    if _STATS:
        print(f"  [svd] fallback usage summary: {_STATS}", flush=True)
    else:
        print(f"  [svd] no fallbacks used (primary {SVD_MODE} path throughout)", flush=True)
