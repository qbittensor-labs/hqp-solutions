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

from quimb.tensor import tensor_network_1d_compress, MatrixProductOperator, MatrixProductState, Circuit

from qiskit_quimb import quimb_circuit
from qiskit import QuantumCircuit
import os

# Not in the reference (env-gated, default = reference behaviour). quimb's compress defaults to
# cutoff_mode="rsum2" (trim while the DISCARDED sum of squares < cutoff * total): MEASURED on one
# spectrum, rsum2@2e-3 keeps 44 values and discards 1.9e-3 of the weight, rel@2e-3 keeps 81 and
# discards 1.7e-5. HQP_CUTOFF_MODE=rel|rsum2|abs|sum2 passes the mode through; unset passes nothing.
_COMPRESS_OPTS = {}
if os.environ.get("HQP_CUTOFF_MODE", "").strip():
    _COMPRESS_OPTS["cutoff_mode"] = os.environ["HQP_CUTOFF_MODE"].strip()

# ------------------------------------------------------------------
#  Constructors
# ------------------------------------------------------------------

def mpo_from_circuit(circ: Circuit):
    # add dummy rz to cover all sites
    for q in range(circ.N):
    #    circ.rz(0.0, q)
        circ.u3(0, 0, 0, q)
    tn_uni = circ.get_uni()

    # contract gates per site tag
    for st in list(tn_uni.site_tags):
        tn_uni ^= st

    # make sure bonds are simple 1D chain bonds
    tn_uni.fuse_multibonds_()  

    # cast as MatrixProductOperator
    mpo = tn_uni.view_as_(
        MatrixProductOperator,
        cyclic=False,
        L=circ.N,
    )

    mpo.ensure_bonds_exist()
    return mpo


# ------------------------------------------------------------------
#  MPO x MPO composition
# ------------------------------------------------------------------

def apply_mpo(mpo1: MatrixProductOperator, mpo2: MatrixProductOperator,
                side,
                max_bond=None,
                cutoff=0.0,
                contract=True,
                compress=True,
                **compress_opts):
    compress_opts = {**_COMPRESS_OPTS, **compress_opts}
    if side == "right":
        return mpo1.apply(
            mpo2,
            compress=compress,
            max_bond=max_bond,
            cutoff=cutoff,
            create_bond=True,
            contract=contract,
            **compress_opts,
        )
    elif side == "left":
        return mpo2.apply(
            mpo1,
            compress=compress,
            max_bond=max_bond,
            cutoff=cutoff,
            create_bond=True,
            contract=contract,
            **compress_opts,
        )
    else:
        raise ValueError("side must be 'left' or 'right'.")



# ------------------------------------------------------------------
#  Applying circuits to MPO
# ------------------------------------------------------------------


def apply_circuit(mpo, circ, side, max_bond=None, cutoff=0.0, contract=True, compress=True, **compress_opts):
    return apply_mpo(mpo, mpo_from_circuit(circ), side=side, max_bond=max_bond, cutoff=cutoff, contract=contract, compress=compress, **compress_opts)


def apply_swaps(mpo: MatrixProductOperator, swaps_l, swaps_r, max_bond=None, cutoff=0.0, to_backend=None, inplace=False):
    N = len(mpo.sites)
    qc_swaps_l = QuantumCircuit(N)
    qc_swaps_r = QuantumCircuit(N)

    for q0, q1 in swaps_l:
        qc_swaps_l.swap(q0, q1)

    for q0, q1 in swaps_r:
        qc_swaps_r.swap(q0, q1)

    mpo_out = mpo if inplace else mpo.copy()

    if len(swaps_l) > 0:
        circ_l = quimb_circuit(qc_swaps_l.inverse().decompose("swap"), Circuit, to_backend=to_backend)
        mpo_out = apply_circuit(mpo_out, circ_l, side="right", max_bond=max_bond, cutoff=cutoff)
    
    if len(swaps_r) > 0:
        circ_r = quimb_circuit(qc_swaps_r.decompose("swap"), Circuit, to_backend=to_backend)
        mpo_out = apply_circuit(mpo_out, circ_r, side="left", max_bond=max_bond, cutoff=cutoff) 

    return mpo_out

