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
import numpy as np

class TorchNP:

    def __init__(self, device: str='cuda:0'):
        import torch
        self.torch, self.device, self.linalg = (torch, device, torch.linalg)

    def asarray(self, x, dtype=None):
        t = self.torch
        if isinstance(x, t.Tensor):
            return x.to(device=self.device, dtype=dtype) if dtype is not None else x.to(self.device)
        return t.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def zeros(self, shape, dtype=None):
        return self.torch.zeros(shape, dtype=dtype, device=self.device)

    def ones(self, shape, dtype=None):
        return self.torch.ones(shape, dtype=dtype, device=self.device)

    def tensordot(self, a, b, axes=2):
        if isinstance(axes, (tuple, list)) and len(axes) == 2:
            x, y = axes
            axes = ([x] if isinstance(x, int) else list(x), [y] if isinstance(y, int) else list(y))
        return self.torch.tensordot(a, b, dims=axes)

    def einsum(self, eq, *ops):
        return self.torch.einsum(eq, *ops)

    def sqrt(self, x):
        return self.torch.sqrt(x) if isinstance(x, self.torch.Tensor) else np.sqrt(x)

    def maximum(self, a, b):
        return self.torch.clamp(a, min=b) if not isinstance(b, self.torch.Tensor) else self.torch.maximum(a, b)

    def argsort(self, x):
        return self.torch.argsort(x)

    def isfinite(self, x):
        return self.torch.isfinite(x)

    def where(self, cond, a, b):
        return self.torch.where(cond, a, b)

    def vdot(self, a, b):
        return self.torch.vdot(a.reshape(-1), b.reshape(-1))

    def asnumpy(self, x):
        return x.detach().cpu().numpy()

    def empty_cache(self):
        self.torch.cuda.empty_cache()
