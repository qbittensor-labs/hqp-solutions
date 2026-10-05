# Copyright (C) 2026 qBitTensor Labs.
# Original author: an anonymous competition participant (Enigma / Hardening Quantum Proof competition).
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

import torch, autoray
import quimb.tensor.decomp as _qd
def _sgn(x):
    xp=_qd.get_namespace(x); ax=xp.abs(x); x0=ax<1e-12
    return (x+x0)/(ax+x0)
_qd.sgn=_sgn
_svd=torch.linalg.svd
def rsvd(A, full_matrices=False, **k):
    if A.dtype==torch.complex64:
        try: return _svd(A, full_matrices=False, driver='gesvd')   # direct gesvd (no wasted gesvdj)
        except Exception:
            U,s,Vh=_svd(A.to(torch.complex128), full_matrices=False, driver='gesvd')
            return U.to(torch.complex64),s.to(torch.float32),Vh.to(torch.complex64)
    if A.dtype==torch.complex128 and A.is_cuda:
        # c128 on GPU: cuSOLVER's default gesvdj can fail to converge on large bonds;
        # gesvd is the robust QR-based path (same reason the c64 branch uses it).
        try: return _svd(A, full_matrices=False, driver='gesvd')
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
