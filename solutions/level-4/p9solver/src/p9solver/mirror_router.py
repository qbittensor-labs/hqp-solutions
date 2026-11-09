"""σ-mirror line router: route the late stream so each swept pair meets on the
same line sites where its early partner executed.

Rationale (2026-07-22, enigma-peaked KREMER_PAPER_GAP_ANALYSIS): the S2 module
windows carry ~160 σ-paired CZs whose implicit cancellation is the whole game.
Sabre routes each half independently, so partners land on unrelated sites and
the MPO must transport them together — the dominant unswap cost, and extra
truncation pressure. This router routes the EARLY stream with sabre as usual,
records the line sites where every paired work gate executed, then routes the
LATE stream with a target-seeking greedy: paired gates steer toward their
partner's recorded sites (σ-conjugated), unpaired mask gates route to nearest
adjacency. Only SWAP insertions are used and per-qubit gate order is preserved,
so the routed circuit is unitarily exact by construction.

Opt-in via ``p9solve --router mirror --sigma-pairs <sigma_pairs.json>``;
default routing remains sabre everywhere (parity preserved).
"""

from __future__ import annotations

import json
import logging

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate

from .qiskit_utils import iter_layers, merge_layers


def _work_gate_sites(layers):
    """Walk routed line layers; return site pairs for each 2q work gate in
    stream order, tracking SWAP-induced wire movement."""
    qc = merge_layers(layers)
    nq = qc.num_qubits
    # position of logical wire on the line; routed circuits act on line sites
    # directly, so sites ARE the qubit indices of each instruction.
    sites = []
    for inst in qc.data:
        name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name in ("measure", "barrier"):
            continue
        if name == "swap" and inst.operation.label == "route":
            continue
        if len(qubits) == 2:
            sites.append(tuple(sorted(qubits)))
    return sites


def _greedy_route_to_targets(layers, targets, num_qubits):
    """Route an (unrouted) stream to line connectivity. ``targets`` maps the
    stream 2q-rank to a preferred site pair or None. Greedy: bubble the two
    logical wires toward the target (or toward each other when no target),
    emitting SWAPs on the line; gates then execute on adjacent sites."""
    qc = merge_layers(layers)
    nq = num_qubits
    pos = list(range(nq))          # logical wire -> line site
    wire_at = list(range(nq))      # line site -> logical wire
    out = QuantumCircuit(nq)
    routed_swaps = 0

    def do_swap(site_a, site_b):
        nonlocal routed_swaps
        out.append(SwapGate(label="route"), [out.qubits[site_a], out.qubits[site_b]])
        routed_swaps += 1
        wa, wb = wire_at[site_a], wire_at[site_b]
        wire_at[site_a], wire_at[site_b] = wb, wa
        pos[wa], pos[wb] = site_b, site_a

    def bubble(wire, dest):
        while pos[wire] != dest:
            step = 1 if dest > pos[wire] else -1
            do_swap(pos[wire], pos[wire] + step)

    rank = 0
    for inst in qc.data:
        name = inst.operation.name
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name in ("measure", "barrier"):
            continue
        if len(qubits) == 1:
            out.append(inst.operation, [out.qubits[pos[qubits[0]]]])
            continue
        a, b = qubits
        target = targets.get(rank)
        rank += 1
        if target is not None:
            lo, hi = target
            # steer the pair onto the partner's sites; order within the pair
            # does not matter for cancellation locality. Bubbling one wire can
            # displace the other from its cell, so iterate to a fixed point
            # (bounded: each pass leaves the displaced wire adjacent).
            if abs(pos[a] - lo) + abs(pos[b] - hi) <= abs(pos[a] - hi) + abs(pos[b] - lo):
                ta, tb = lo, hi
            else:
                ta, tb = hi, lo
            for _ in range(6):
                if pos[a] != ta:
                    bubble(a, ta)
                if pos[b] != tb:
                    bubble(b, tb)
                if pos[a] == ta and pos[b] == tb:
                    break
            else:
                raise RuntimeError("mirror-router target placement did not converge")
        else:
            # no target: contract the pair to adjacency at the midpoint
            while abs(pos[a] - pos[b]) > 1:
                if pos[a] < pos[b]:
                    do_swap(pos[a], pos[a] + 1)
                    if abs(pos[a] - pos[b]) > 1:
                        do_swap(pos[b], pos[b] - 1)
                else:
                    do_swap(pos[b], pos[b] + 1)
                    if abs(pos[a] - pos[b]) > 1:
                        do_swap(pos[a], pos[a] - 1)
        out.append(inst.operation, [out.qubits[pos[a]], out.qubits[pos[b]]])
    # final wire permutation is implicit (pos); the caller's measurement
    # relabel handles it exactly as it does for sabre's final permutation.
    out.measure_all()
    logging.info(
        f"    [mirror-router] routed {rank} work gates with {routed_swaps} swaps"
    )
    return list(iter_layers(out)), pos


def load_pair_targets(sigma_pairs_path):
    """sigma_pairs.json (diagnostics/sigma_pair_odometer.py) -> rank maps.

    Pairs are stored as (lrank, rrank), 1-based absorbed counts; stream 2q
    rank is rank-1. Returns {rrank0: lrank0} for the late stream.
    """
    sidecar = json.loads(open(sigma_pairs_path).read())
    return {rr - 1: lr - 1 for lr, rr in sidecar["pairs"]}


def mirror_route_right(layers_left_routed, layers_right, pair_map, num_qubits):
    """Route the late stream so paired gates land on their partner's sites."""
    left_sites = _work_gate_sites(layers_left_routed)
    targets = {}
    for rrank, lrank in pair_map.items():
        if 0 <= lrank < len(left_sites):
            targets[rrank] = left_sites[lrank]
    logging.info(
        f"    [mirror-router] {len(targets)} of {len(pair_map)} pair targets "
        f"resolved from the routed early stream"
    )
    return _greedy_route_to_targets(layers_right, targets, num_qubits)
