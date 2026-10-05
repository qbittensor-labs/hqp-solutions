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
import math
from dataclasses import dataclass, field
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks

def load_circuit(qasm_file: str) -> QuantumCircuit:
    with open(qasm_file) as fh:
        head = fh.readline()
    if '3.0' in head:
        import qiskit.qasm3 as qasm3
        qc = qasm3.load(qasm_file)
    else:
        qc = QuantumCircuit.from_qasm_file(qasm_file)
    qc.remove_final_measurements(inplace=True)
    return qc

def consolidate(region: QuantumCircuit) -> QuantumCircuit:
    return PassManager([Collect2qBlocks(), ConsolidateBlocks(force_consolidate=True)]).run(region)

def blocks_before(raw: QuantumCircuit, cut: int) -> int:
    sub = QuantumCircuit(raw.num_qubits)
    for inst in raw.data[:cut]:
        sub.append(inst.operation, [raw.find_bit(q).index for q in inst.qubits])
    return sum((1 for inst in consolidate(sub).data if len(inst.qubits) == 2))

def two_q_ordinal(raw: QuantumCircuit) -> np.ndarray:
    out = np.zeros(len(raw.data) + 1, dtype=int)
    for i, inst in enumerate(raw.data):
        out[i + 1] = out[i] + (1 if len(inst.qubits) == 2 else 0)
    return out

def seam_density(raw: QuantumCircuit, n_blocks: int, window_frac: float=0.0125, tol: float=0.05):
    ord2 = two_q_ordinal(raw)
    n2 = max(1, int(ord2[-1]))
    pos = []
    for i, inst in enumerate(raw.data):
        if len(inst.qubits) == 1 and inst.operation.name == 'u':
            th = float(inst.operation.params[0])
            if abs(abs(th) - math.pi / 2) < tol:
                pos.append(ord2[i] / n2)
    pos = np.asarray(pos)
    grid = np.linspace(0, 1, 401)
    dens = np.array([((pos > g - window_frac) & (pos <= g + window_frac)).sum() for g in grid])
    return (grid, dens)

def seam_peaks(grid, dens, n_blocks: int, min_sep_frac: float=0.15):
    if not np.any(dens > 0):
        return []
    order = np.argsort(-dens)
    picks = []
    for i in order:
        g = grid[i]
        if all((abs(g - p) >= min_sep_frac for p in picks)):
            picks.append(g)
        if len(picks) == 2:
            break
    return [int(round(p * n_blocks)) for p in sorted(picks)]

def oneq_raw_positions(raw: QuantumCircuit, oneq) -> list[int]:
    per_q: dict[int, list[int]] = {}
    for i, inst in enumerate(raw.data):
        if len(inst.qubits) == 1 and inst.operation.name not in ('barrier', 'delay', 'measure', 'reset'):
            per_q.setdefault(raw.find_bit(inst.qubits[0]).index, []).append(i)
    seen: dict[int, int] = {}
    out = []
    for g in oneq:
        k = seen.get(g.qubit, 0)
        seen[g.qubit] = k + 1
        out.append(per_q[g.qubit][k])
    return out

def block_map_grid(raw: QuantumCircuit, step: int=25) -> np.ndarray:
    n = len(raw.data)
    ord2 = two_q_ordinal(raw)
    grid = list(range(0, n + 1, step))
    if grid[-1] != n:
        grid.append(n)
    Bg = [blocks_before(raw, g) for g in grid]
    B = np.zeros(n + 1, dtype=float)
    for (g0, b0), (g1, b1) in zip(zip(grid, Bg), zip(grid[1:], Bg[1:])):
        o0, o1 = (ord2[g0], ord2[g1])
        for r in range(g0, g1 + 1):
            B[r] = b0 if o1 == o0 else b0 + (b1 - b0) * (ord2[r] - o0) / (o1 - o0)
    return B

def cz_records(raw: QuantumCircuit, st) -> list[dict]:
    per_edge: dict[tuple[int, int], list[int]] = {}
    for i, inst in enumerate(raw.data):
        if len(inst.qubits) == 2 and inst.operation.name == 'cz':
            a, b = sorted((raw.find_bit(q).index for q in inst.qubits))
            per_edge.setdefault((a, b), []).append(i)
    seen: dict[tuple[int, int], int] = {}
    out = []
    for g in st.cz:
        k = seen.get(g.edge, 0)
        seen[g.edge] = k + 1
        lst = per_edge.get(g.edge, [])
        if k < len(lst):
            out.append(dict(layer=g.layer, edge=g.edge, raw=lst[k]))
    return out

def pair_scan(czs, perm, centre_layer, e_span, l_span, tol=8):
    early = [c for c in czs if e_span[0] <= c['layer'] <= e_span[1]]
    late = [c for c in czs if l_span[0] <= c['layer'] <= l_span[1]]
    by_edge: dict[tuple[int, int], list[dict]] = {}
    for c in late:
        by_edge.setdefault(c['edge'], []).append(c)
    used: set[int] = set()
    pairs = []
    for c in sorted(early, key=lambda c: -c['layer']):
        me = tuple(sorted((int(perm[c['edge'][0]]), int(perm[c['edge'][1]]))))
        pred = 2 * centre_layer - c['layer']
        cands = [d for d in by_edge.get(me, []) if abs(d['layer'] - pred) <= tol and d['raw'] not in used]
        if not cands:
            continue
        d = min(cands, key=lambda d: (abs(d['layer'] - pred), d['raw']))
        used.add(d['raw'])
        pairs.append((c, d))
    return pairs

@dataclass
class Module:
    module_id: int
    centre_block: int
    raw_centre: int
    raw_early: int
    raw_late: int
    pair_count: int
    cz_timed_rate: float
    seam_centre_block: int | None = None
    cz_pairs: int = 0
    centre_1q: int | None = None
    span_blocks: tuple[int, int] | None = None
    cz_span: tuple[int, int] | None = None
    notes: list[str] = field(default_factory=list)

@dataclass
class Structure:
    n_qubits: int
    n_raw: int
    n_blocks: int
    modules: list[Module]
    seam_centres: list[int]
    method: str

def analyse(raw: QuantumCircuit) -> Structure:
    from enigma_peaked.structure.gadgets import analyze_circuit, collect_structure, discover_unique_inverse_pairs, cluster_local_pairs, AnalysisConfig
    n_blocks = sum((1 for inst in consolidate(raw).data if len(inst.qubits) == 2))
    grid, dens = seam_density(raw, n_blocks)
    seams = seam_peaks(grid, dens, n_blocks)
    modules: list[Module] = []
    method = 'gadgets'
    try:
        cfg = AnalysisConfig()
        st = collect_structure(raw)
        disc = discover_unique_inverse_pairs(st.oneq, tolerance=cfg.inverse_tol, batch_size=cfg.inverse_batch)
        clusters = cluster_local_pairs(disc.pairs, max_midpoint_gap=cfg.cluster_gap, min_pairs=cfg.min_pairs)
        rep = analyze_circuit(raw, cfg)
        B = block_map_grid(raw)
        czs = cz_records(raw, st)
        for m, cluster in zip(rep.modules, clusters):
            rawi = oneq_raw_positions(raw, st.oneq)
            ord2 = two_q_ordinal(raw)
            mids_blk = [(ord2[rawi[p.early_index]] + ord2[rawi[p.late_index]]) / 2.0 for p in cluster]
            target = float(np.mean(mids_blk))
            raw_c = int(np.searchsorted(ord2, target))
            raw_e = int(min((rawi[p.early_index] for p in cluster)))
            raw_l = int(max((rawi[p.late_index] for p in cluster)))
            c_1q = blocks_before(raw, raw_c)
            pairs = pair_scan(czs, m.permutation, float(m.center), (m.early_layer_min, m.early_layer_max), (m.late_layer_min, m.late_layer_max), tol=8)
            if len(pairs) >= 20:
                c_blk = int(round(float(np.mean([(B[c['raw']] + B[d['raw']]) / 2 for c, d in pairs]))))
            else:
                c_blk = c_1q + 11
            mod = Module(m.module_id, c_blk, raw_c, raw_e, raw_l, int(m.pair_count), float(m.cz_timed_rate))
            mod.cz_pairs, mod.centre_1q = (len(pairs), c_1q)
            if pairs:
                mod.cz_span = (int(round(min((B[c['raw']] for c, d in pairs)))), int(round(max((B[d['raw']] for c, d in pairs)))))
            mod.span_blocks = (int(round(B[raw_e])), int(round(B[min(raw_l + 1, len(B) - 1)])))
            modules.append(mod)
    except Exception as exc:
        method = f'gadgets-failed:{type(exc).__name__}'
    modules.sort(key=lambda m: m.centre_block)
    if len(modules) < 2:
        method += '+seams'
        centres = seams if len(seams) == 2 else [n_blocks // 3, 2 * n_blocks // 3]
        modules = [Module(i, c, -1, -1, -1, 0, 0.0) for i, c in enumerate(centres)]
    for m in modules:
        near = [s for s in seams if abs(s - m.centre_block) < 0.12 * n_blocks]
        m.seam_centre_block = min(near, key=lambda s: abs(s - m.centre_block)) if near else None
    return Structure(raw.num_qubits, len(raw.data), n_blocks, modules[:2], seams, method)
if __name__ == '__main__':
    import sys, time
    for f in sys.argv[1:]:
        t0 = time.time()
        qc = load_circuit(f)
        s = analyse(qc)
        print(f'{f}: {s.n_qubits}q {s.n_raw} raw {s.n_blocks} blocks, method {s.method}, seams {s.seam_centres}, {time.time() - t0:.1f}s')
        for m in s.modules:
            print(f'   module {m.module_id}: centre block {m.centre_block} (pairs {m.cz_pairs}; 1q-pair estimate {m.centre_1q}; block span {m.span_blocks}), pairs {m.pair_count}, cz_timed {m.cz_timed_rate:.2f}, seam {m.seam_centre_block}')
