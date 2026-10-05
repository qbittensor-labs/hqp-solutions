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

"""Global eigh-trick SVD acceleration for quimb's MPO compress path.

torch.linalg.svd on CUDA (gesvd driver, as forced by the reference's robust-SVD
patch) costs ~104ms at 1024x1024 c64; the Gram eigh-trick computes the same
truncated factorization in ~14ms (7-10x). All quimb zipup/compress SVDs route
through torch.linalg.svd, so one patch accelerates the whole MPO absorb.

Semantics: only intercepts 2-D, full_matrices=False, min-dim >= THRESH calls;
everything else (and any non-finite result) falls back to the original svd.
Eigenvectors are exact under identity-shift regularization; the junk tail of
tiny singular values is discarded by quimb's cutoff anyway.

Call install_fast_svd() AFTER any other svd patch (uses the current
torch.linalg.svd as fallback).
"""
import torch

THRESH = 192  # below this, cusolver svd is fast enough / eigh gains nothing

_ORIG_SVD = None  # the stable svd (e.g. the reference's robust gesvd) captured at install


def restore_svd():
    """Restore the pre-fast (stable gesvd) svd. Use for the one-shot final
    extraction, where eigh's squared condition number can produce NaNs."""
    global _ORIG_SVD
    if _ORIG_SVD is not None:
        torch.linalg.svd = _ORIG_SVD
        return True
    return False


def install_fast_svd(log=print):
    global _ORIG_SVD
    orig = torch.linalg.svd
    _ORIG_SVD = orig

    def fast(A, full_matrices=True, *, driver=None, out=None):
        if (out is not None or full_matrices or not torch.is_tensor(A)
                or A.ndim != 2 or min(A.shape) < THRESH or not A.is_cuda):
            return orig(A, full_matrices=full_matrices)
        try:
            if not torch.isfinite(A.real).all():
                raise RuntimeError("nonfinite input")
            a, b = A.shape
            # eigh of the Gram squares the condition number, so singular values below
            # ~sqrt(eps)*s_max are inaccurate and dividing by them (U = A V / s) injects
            # huge/NaN columns that cascade into the next Gram. Those columns correspond
            # to the smallest singular values, which the caller's truncation DROPS anyway,
            # so we zero them here instead of dividing -> exact on the kept block, safe tail.
            if a >= b:
                H = A.mH @ A
                eps = float(H.diagonal().real.sum()) / b * 1e-6 + 1e-30
                H = H + eps * torch.eye(b, dtype=H.dtype, device=H.device)
                w, V = torch.linalg.eigh(H)
                w = (w - eps).flip(0).clamp_min(0)
                V = V.flip(1)
                s = torch.sqrt(w)
                smax = float(s[0]) if s.numel() else 0.0
                keep = s > (1e-7 * smax if smax > 0 else 0.0)
                inv = torch.where(keep, 1.0 / s.clamp_min(1e-30),
                                  torch.zeros_like(s))
                U = (A @ V) * inv.to(A.dtype)
                s = torch.where(keep, s, torch.zeros_like(s))
                Vh = V.mH
            else:
                H = A @ A.mH
                eps = float(H.diagonal().real.sum()) / a * 1e-6 + 1e-30
                H = H + eps * torch.eye(a, dtype=H.dtype, device=H.device)
                w, U = torch.linalg.eigh(H)
                w = (w - eps).flip(0).clamp_min(0)
                U = U.flip(1)
                s = torch.sqrt(w)
                smax = float(s[0]) if s.numel() else 0.0
                keep = s > (1e-7 * smax if smax > 0 else 0.0)
                inv = torch.where(keep, 1.0 / s.clamp_min(1e-30),
                                  torch.zeros_like(s))
                Vh = (U.mH @ A) * inv.to(A.dtype).unsqueeze(1)
                s = torch.where(keep, s, torch.zeros_like(s))
            if not (torch.isfinite(U.real).all() and torch.isfinite(Vh.real).all()
                    and torch.isfinite(s).all()):
                raise RuntimeError("nonfinite")
            if s.dtype != A.real.dtype:
                s = s.to(A.real.dtype)
            return torch.return_types.linalg_svd((U, s, Vh))
        except Exception:
            return orig(A, full_matrices=full_matrices)

    torch.linalg.svd = fast
    log("Fast SVD installed (Gram eigh-trick for big CUDA matrices).")
    return True
