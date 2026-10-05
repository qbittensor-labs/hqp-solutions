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
import atexit
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

class BackendError(RuntimeError):
    pass

@dataclass(frozen=True)
class Backend:
    name: str
    dtype: str
    device: str

    def describe(self) -> dict[str, str]:
        return {'name': self.name, 'dtype': self.dtype, 'device': self.device}

    def to_backend(self) -> Callable[[Any], Any] | None:
        if self.name == 'numpy':
            if self.dtype == 'complex128':
                return None
            import numpy as np
            return lambda x: np.asarray(x, dtype=np.complex64)
        torch = _import_torch()
        dtype = torch.complex64 if self.dtype == 'complex64' else torch.complex128
        device = self.device
        return lambda x: torch.as_tensor(_as_numpy(x), dtype=dtype, device=device)

def _as_numpy(x: Any):
    import numpy as np
    if hasattr(x, 'cpu'):
        x = x.cpu()
    return np.asarray(x)

def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise BackendError("the torch backend requires the optional 'gpu' dependency group") from exc
    return torch

def make_backend(name: str, dtype: str, device: str) -> Backend:
    if name not in {'numpy', 'torch'}:
        raise BackendError(f'unknown backend {name!r}')
    if dtype not in {'complex64', 'complex128'}:
        raise BackendError(f'unknown dtype {dtype!r}')
    if name == 'numpy':
        if device != 'cpu':
            raise BackendError("the numpy backend supports device='cpu' only")
    else:
        torch = _import_torch()
        if device.startswith('cuda') and (not torch.cuda.is_available()):
            raise BackendError(f'device {device!r} requested but CUDA is unavailable')
    return Backend(name=name, dtype=dtype, device=device)
_TORCH_PATCH_HANDLE: dict[str, Any] | None = None

def install_torch_linalg_patch(backend: Backend) -> Callable[[], None]:
    global _TORCH_PATCH_HANDLE
    if backend.name != 'torch':
        raise BackendError('the torch linalg patch applies only to the torch backend')
    if _TORCH_PATCH_HANDLE is not None:
        return _TORCH_PATCH_HANDLE['uninstall']
    import autoray
    import quimb.tensor.decomp as _qd
    import scipy.linalg as _sla
    torch = _import_torch()
    _SVD_CHECK = os.environ.get('HQP_SVD_CHECK', '1') == '1'
    _SVD_CHECK_MISSES = [0]

    def _report_misses():
        if _SVD_CHECK_MISSES[0]:
            print(f'[svd-check] {_SVD_CHECK_MISSES[0]} wrong cuda SVDs recomputed on CPU', file=sys.stderr, flush=True)
    atexit.register(_report_misses)
    original_svd = torch.linalg.svd
    original_qr = torch.linalg.qr
    original_sgn = _qd.sgn

    def _sgn(x):
        xp = _qd.get_namespace(x)
        ax = xp.abs(x)
        x0 = ax < 1e-12
        return (x + x0) / (ax + x0)

    def robust_svd(A, full_matrices=False, **kwargs):
        if A.dtype == torch.complex64:
            if not A.is_cuda:
                try:
                    U, s, Vh = original_svd(A.to(torch.complex128), full_matrices=False)
                except Exception:
                    u, sv, vh = _sla.svd(A.to(torch.complex128).numpy(), full_matrices=False, lapack_driver='gesvd')
                    U, s, Vh = (torch.from_numpy(u), torch.from_numpy(sv), torch.from_numpy(vh))
                return (U.to(torch.complex64), s.to(torch.float32), Vh.to(torch.complex64))
            mn = min(A.shape[-2:])
            driver = 'gesvd' if 256 < mn <= 2048 else 'gesvdj'
            try:
                U, s, Vh = original_svd(A, full_matrices=False, driver=driver)
                if torch.isfinite(s).all() and torch.isfinite(U).all() and torch.isfinite(Vh).all():
                    if mn > 256 or not _SVD_CHECK:
                        return (U, s, Vh)
                    rec = U * s.to(U.dtype).unsqueeze(-2) @ Vh
                    err = torch.linalg.matrix_norm(rec - A) / torch.linalg.matrix_norm(A).clamp_min(1e-30)
                    if bool((err < 0.001).all()):
                        return (U, s, Vh)
                    _SVD_CHECK_MISSES[0] += 1
                    _dump = os.environ.get('HQP_SVD_DUMP')
                    if _dump and _SVD_CHECK_MISSES[0] <= 8:
                        import numpy as _np
                        os.makedirs(_dump, exist_ok=True)
                        _np.savez(os.path.join(_dump, f'miss{_SVD_CHECK_MISSES[0]}.npz'), A=A.detach().resolve_conj().cpu().numpy(), is_conj=bool(A.is_conj()), is_neg=bool(A.is_neg()), contiguous=bool(A.is_contiguous()), stride=_np.array(A.stride()), shape=_np.array(A.shape), driver=driver, err=float(err.max()), torch_version=torch.__version__, cuda_version=str(torch.version.cuda))
                    if _SVD_CHECK_MISSES[0] <= 3:
                        print(f'[svd-check] wrong cuda SVD (shape {tuple(A.shape)}, rel err {float(err.max()):.2e}); recomputed on CPU (miss {_SVD_CHECK_MISSES[0]})', file=sys.stderr, flush=True)
            except Exception:
                pass
            u, sv, vh = _sla.svd(A.detach().to('cpu', torch.complex128).resolve_conj().numpy(), full_matrices=False, lapack_driver='gesvd')
            dev = A.device
            return (torch.from_numpy(u).to(dev, torch.complex64), torch.from_numpy(sv).to(dev, torch.float32), torch.from_numpy(vh).to(dev, torch.complex64))
        if A.dtype == torch.complex128 and A.is_cuda:
            try:
                return original_svd(A, full_matrices=False, driver='gesvd')
            except Exception:
                return original_svd(A, full_matrices=False)
        return original_svd(A, full_matrices=False)

    def robust_qr(A, *args, **kwargs):
        if A.dtype == torch.complex64:
            try:
                return original_qr(A, *args, **kwargs)
            except Exception:
                Q, R = original_qr(A.to(torch.complex128), *args, **kwargs)
                return (Q.to(torch.complex64), R.to(torch.complex64))
        return original_qr(A, *args, **kwargs)
    torch.linalg.svd = robust_svd
    torch.linalg.qr = robust_qr
    autoray.register_function('torch', 'linalg.svd', robust_svd)
    autoray.register_function('torch', 'linalg.qr', robust_qr)
    _qd.sgn = _sgn

    def uninstall() -> None:
        global _TORCH_PATCH_HANDLE
        torch.linalg.svd = original_svd
        torch.linalg.qr = original_qr
        autoray.register_function('torch', 'linalg.svd', original_svd)
        autoray.register_function('torch', 'linalg.qr', original_qr)
        _qd.sgn = original_sgn
        _TORCH_PATCH_HANDLE = None
    _TORCH_PATCH_HANDLE = {'uninstall': uninstall}
    return uninstall
