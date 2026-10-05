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
STATS = {'gram_calls': 0, 'fallbacks': 0}

def gram_svd(A, base_svd, *, min_dim: int=256, rel_tiny: float=1e-06, check_tol: float=0.01, full_matrices=False, **kwargs):
    import torch
    if A.ndim != 2 or min(A.shape) <= min_dim or (not (A.is_complex() or A.is_floating_point())):
        return base_svd(A, full_matrices=False, **kwargs)
    try:
        m, n = A.shape
        AH = A.mH
        if m <= n:
            w, W = torch.linalg.eigh(A @ AH)
        else:
            w, W = torch.linalg.eigh(AH @ A)
        w = torch.flip(w, (0,)).clamp(min=0)
        W = torch.flip(W, (1,))
        s = torch.sqrt(w)
        tiny = s <= s[0] * rel_tiny
        inv = torch.where(tiny, torch.zeros_like(s), 1.0 / torch.where(tiny, torch.ones_like(s), s)).to(A.dtype)
        if m <= n:
            U = W
            Vh = W.mH @ A * inv[:, None]
        else:
            Vh = W.mH
            U = A @ W * inv[None, :]
        x = torch.randn(n, dtype=A.dtype, device=A.device)
        ax = A @ x
        r = ax - U @ (s.to(A.dtype) * (Vh @ x))
        ok = bool(torch.isfinite(s).all()) and bool(torch.isfinite(U).all()) and bool(torch.isfinite(Vh).all())
        ok = ok and float(torch.linalg.vector_norm(r)) <= check_tol * max(float(torch.linalg.vector_norm(ax)), 1e-300)
        if ok:
            STATS['gram_calls'] += 1
            return (U, s, Vh)
    except Exception:
        pass
    STATS['fallbacks'] += 1
    return base_svd(A, full_matrices=False, **kwargs)
SPECTRUM = {'path': None, 'min_dim': 64, 'every': 1, 'seen': 0, 'written': 0, 'fh': None}
_TAPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
_THRESHOLDS = (0.01, 0.001, 0.0005, 0.0001)

def record_spectrum(s):
    cfg = SPECTRUM
    if cfg['path'] is None:
        return
    try:
        import json, math
        k = int(s.shape[0])
        if k < cfg['min_dim']:
            return
        cfg['seen'] += 1
        if cfg['seen'] % cfg['every']:
            return
        top = float(s[0])
        if not top > 0.0 or not math.isfinite(top):
            return
        v = (s / s[0]).detach().to('cpu').float().numpy()
        taps = {str(t): float(v[t]) if t < k else None for t in _TAPS}
        counts = {f'n>{t:g}': int((v > t).sum()) for t in _THRESHOLDS}
        n = min(64, k)
        if n >= 8:
            import numpy as _np
            x = _np.log(_np.arange(1, n + 1))
            y = _np.log(_np.maximum(v[:n], 1e-300))
            slope = float(_np.polyfit(x, y, 1)[0])
        else:
            slope = None
        if cfg['fh'] is None:
            cfg['fh'] = open(cfg['path'], 'a', buffering=1)
        cfg['fh'].write(json.dumps({'k': k, 'top': top, 'taps': taps, 'counts': counts, 'loglog_slope': slope}) + '\n')
        cfg['written'] += 1
    except Exception:
        pass

def install(min_dim: int=256):
    import autoray
    import torch
    base = torch.linalg.svd

    def svd(A, full_matrices=False, **kwargs):
        out = gram_svd(A, base, min_dim=min_dim, **kwargs)
        if SPECTRUM['path'] is not None:
            try:
                record_spectrum(out[1])
            except Exception:
                pass
        return out
    torch.linalg.svd = svd
    autoray.register_function('torch', 'linalg.svd', svd)
    return base
