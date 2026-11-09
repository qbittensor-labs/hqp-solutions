"""Helper for --absorb-swaps-as-perm: apply a routed layer to the MPO with routing SWAPs
handled as permutation updates (never materialized as gates), work gates remapped through
the running permutation.

Convention (validated by tests/test_swap_perm_principle.py, err=0):
  perm[wire] = MPO site currently holding routed-wire `wire`; start identity.
  - a routed layer's work gates act on routed wires; apply them to MPO sites perm[wire]
    (using perm BEFORE this layer's swaps, since routing puts a layer's swaps in service of
    the NEXT layer's gates).
  - then update perm with this layer's SWAP pairs.
The MPO stays in its fixed frame => routing SWAPs cost ZERO bond. Work gates may land on
non-adjacent sites (paid once, the fundamental routing cost) instead of churning.
"""
from qiskit import QuantumCircuit


def split_layer(layer):
    """Return (swap_pairs, has_work). swap_pairs are (i,j) routed-wire indices."""
    swaps = []
    has_work = False
    for ci in layer.data:
        qs = [layer.find_bit(q).index for q in ci.qubits]
        name = ci.operation.name
        if name == "swap":
            swaps.append((qs[0], qs[1]))
        elif name not in ("barrier", "measure", "delay"):
            has_work = True
    return swaps, has_work


def remap_work_layer(layer, perm):
    """Build a QuantumCircuit with this layer's WORK gates only, remapped onto MPO sites
    via perm[wire]. Returns (work_circuit_or_None, new_perm)."""
    n = layer.num_qubits
    work = QuantumCircuit(n)
    any_work = False
    swaps = []
    for ci in layer.data:
        qs = [layer.find_bit(q).index for q in ci.qubits]
        name = ci.operation.name
        if name == "swap":
            swaps.append((qs[0], qs[1]))
        elif name in ("barrier", "measure", "delay"):
            continue
        else:
            work.append(ci.operation, [perm[q] for q in qs])
            any_work = True
    new_perm = list(perm)
    for (i, j) in swaps:
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return (work if any_work else None), new_perm
