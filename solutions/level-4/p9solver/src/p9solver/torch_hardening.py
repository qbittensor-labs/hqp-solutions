"""July-proven GPU hardening, verbatim from gpu_engine/fp32_patch.py.

Applied GLOBALLY (torch.linalg.svd/qr + autoray registration) the moment the
torch backend is selected: gesvd driver pinning, c128-upcast retries on BOTH
svd and qr, and the sgn guard. This stack ran the Blackwell box at 83%% GPU
for hours in July; piecemeal ports of it kept missing paths (QR!).
"""
import os, torch, autoray
import quimb.tensor.decomp as _qd

# quimb >= 1.11 removed quimb.tensor.decomp.get_namespace; provide a shim that
# resolves the array namespace via autoray (box-measured AttributeError).
def _ns(x):
    try:
        import quimb.tensor.decomp as _d
        if hasattr(_d, "get_namespace"):
            return _d.get_namespace(x)
    except Exception:
        pass
    try:
        from autoray import infer_backend, get_lib_fn
        import importlib
        be = infer_backend(x)
        return importlib.import_module({"numpy": "numpy", "builtins": "numpy"}.get(be, be))
    except Exception:
        import numpy
        return numpy

# GPU SVD driver. Default 'gesvd' = robust QR path (Box B: ~2.2h/ordering). On Blackwell
# sm_120 gesvd is ~3x slower (unoptimized cuSOLVER kernels) — set HQP_SVD_DRIVER=gesvdj
# (or the fastest CORRECT driver from gpu/svd_bench.py) to switch WITHOUT a rebuild.
# 'default' -> torch/cuSOLVER auto. Keep gesvd unless the bench proves another is correct.
_GD = os.environ.get("P9_SVD_DRIVER", os.environ.get("HQP_SVD_DRIVER", "gesvd"))
_GDRV = None if _GD == "default" else _GD
def _sgn(x):
    xp= _ns(x); ax=xp.abs(x); x0=ax<1e-12
    return (x+x0)/(ax+x0)
_qd.sgn=_sgn
_svd=torch.linalg.svd
def rsvd(A, full_matrices=False, **k):
    if A.dtype==torch.complex64:
        # fp32 overflow safety independent of the fid10 tracker (FFID10=0 runs
        # bypass retention.tracked_svd entirely): bound magnitudes and sanitize
        # non-finite inputs before cuSOLVER sees them.
        _m = A.abs().max()
        if not torch.isfinite(_m):
            # [peakpilot] was nan_to_num (silent zeroing); now fatal.
            from p9solver.nonfinite import raise_nonfinite
            raise_nonfinite("torch_hardening.rsvd (complex64 SVD input)", A)
        if float(_m) > 1e4:
            A = A / _m
        if not A.is_cuda:
            # driver= is CUDA/cuSOLVER-only; CPU LAPACK path: upcast, and on gesdd
            # non-convergence retry via scipy's QR-based gesvd — the exact CPU
            # analogue of the CUDA branch's driver='gesvd' (engine CPU fallback was
            # dead without both fixes — 2026-07-07 toy runs)
            try:
                U,s,Vh=_svd(A.to(torch.complex128), full_matrices=False)
            except Exception:
                import scipy.linalg as _sla
                u,sv,vh=_sla.svd(A.to(torch.complex128).numpy(), full_matrices=False,
                                 lapack_driver='gesvd')
                U,s,Vh=torch.from_numpy(u),torch.from_numpy(sv),torch.from_numpy(vh)
            return U.to(torch.complex64),s.to(torch.float32),Vh.to(torch.complex64)
        try: return _svd(A, full_matrices=False, driver=_GDRV)   # HQP_SVD_DRIVER (def gesvd)
        except Exception:
            U,s,Vh=_svd(A.to(torch.complex128), full_matrices=False, driver='gesvd')
            return U.to(torch.complex64),s.to(torch.float32),Vh.to(torch.complex64)
    if A.dtype==torch.complex128 and A.is_cuda:
        # c128 on GPU: cuSOLVER's default gesvdj can fail to converge on large bonds;
        # gesvd is the robust QR-based path (same reason the c64 branch uses it).
        try: return _svd(A, full_matrices=False, driver=_GDRV)
        except Exception: return _svd(A, full_matrices=False)
    return _svd(A, full_matrices=False)
_qr=torch.linalg.qr
def rqr(A,*a,**k):
    if A.dtype==torch.complex64:
        try: return _qr(A,*a,**k)
        except Exception:
            Q,R=_qr(A.to(torch.complex128),*a,**k); return Q.to(torch.complex64),R.to(torch.complex64)
    return _qr(A,*a,**k)
torch.linalg.svd=rsvd; torch.linalg.qr=rqr
autoray.register_function('torch','linalg.svd',rsvd)
autoray.register_function('torch','linalg.qr',rqr)
