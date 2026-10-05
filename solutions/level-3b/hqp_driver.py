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
import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'src'))
sys.path.insert(0, str(HERE / 'scripts'))
START = time.time()
LOG_LINES: list[str] = []

def log(msg: str) -> None:
    line = f'[+{time.time() - START:7.1f}s] {msg}'
    LOG_LINES.append(line)
    print(line, flush=True)
RECORD = dict(max_bond=4096, cutoff=0.0001, final_cutoff=0.0001, unswap_threshold=4000000.0, max_its=20, beam_size=512, topk=32, sabre_trials=90, post_sabre_trials=50)
RECORD_FLAGS = ['--gram-min-dim', '999999', '--adaptive-unswap', '4,1.0,4000000', '--tail-starve-cap', '4', '--absorb-offload-elems', '200000000']
SOLVE = dict(RECORD, unswap_threshold=1000000.0)
SOLVE_FLAGS = ['--gram-min-dim', '999999', '--adaptive-unswap', '4,1.0,1000000', '--tail-starve-cap', '4', '--absorb-offload-elems', '200000000']

def scale_unswap(cfg: dict, flags: list[str], factor: float) -> tuple[dict, list[str]]:
    if factor == 1.0:
        return (cfg, flags)
    cfg = dict(cfg, unswap_threshold=max(1000.0, cfg['unswap_threshold'] * factor))
    out = list(flags)
    i = out.index('--adaptive-unswap')
    g, f, n = out[i + 1].split(',')
    out[i + 1] = f'{g},{f},{max(1000, int(float(n) * factor))}'
    return (cfg, out)
TAIL_STOP = '512,150'
EARLY_STOP_GATES = 100
RACE_A_OFFSETS = [0]
RACE_B_OFFSETS = [-4, 0, 4]
SCREEN_OFFSETS = RACE_B_OFFSETS
MAX_FACTOR = 5
MARGIN = 5.0
MIN_OBJECT_FRACTION = 0.15
RACE_SECONDS = 720.0
GAP_MARGIN = 45
ANCHOR_MARGIN = 48
OFFSET_LADDER = [('A', 3, 1.0), ('A', 4, 1.0), ('B', 0, 1.0), ('B', 3, 1.0), ('A', 5, 1.0), ('B', -3, 1.0), ('A', 2, 1.0), ('B', 4, 1.0), ('A', 6, 1.0), ('B', 1, 1.0), ('A', 3, 1.02), ('B', -1, 1.0), ('A', 4, 1.02), ('B', 0, 1.02)]
CONT_MIN_LEFT = 2400
VOTE_MIN_RATIO = 1.3
LADDER_MIN_LEFT = 3000
SOLVE_MAX_DRAINS = 5
DISK_MIN_FREE = 2500000000.0
ORDERING_SEED = int(os.environ.get('HQP_ORDERING_SEED', '123'))
POOL_MAX = 40
POOL_WORKERS = 8
POOL_SECONDS = 600
OBJECT_CANDIDATES = 3

def effective_cpus() -> float:
    try:
        q, per = open('/sys/fs/cgroup/cpu.max').read().split()[:2]
        if q != 'max':
            return float(q) / float(per)
    except (OSError, ValueError):
        pass
    try:
        q = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read())
        per = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
        if q > 0:
            return q / per
    except (OSError, ValueError):
        pass
    try:
        return float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return float(os.cpu_count() or 1)

def capped_env(share: int=1) -> dict:
    env = dict(os.environ)
    vis, quota = (os.cpu_count() or 1, effective_cpus())
    if vis > 1.5 * quota:
        n = str(max(1, int(quota // max(1, share))))
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
            env.setdefault(k, n)
    env['PYTHONPATH'] = str(HERE) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env
PARALLEL_HUNT_ARMS = int(os.environ.get('HQP_ARMS', '0')) or 1
PORTFOLIO_GRACE = 600
READOUT_CHI = 256
READOUT_CHI_FALLBACK = 128
READOUT_MIN_FRACTION = 0.4
READOUT_SECONDS = 900
TS_TOP = 8
TS_ACCEPT = 8.0
TS_SECONDS = 600
FB_CHI = 64
FB_BEAM = 4096
FB_ORDERINGS = 5
FB_AGREE = 2
FB_SECONDS = 1500
MAX_BG_READOUTS = 2
READOUT_BG_THREADS = 2
SOLVE_KEEP = 2 if PARALLEL_HUNT_ARMS >= 2 else 1
ATTEMPT1_SECONDS = float(os.environ.get('HQP_ATTEMPT1_SECONDS', '6600'))
ATTEMPT2_UNSWAP = 2000000
ATTEMPT2_MARGIN_SCALE = 1.5
ATTEMPT2_CUTOFF_SCALE = 1.02
ATTEMPT2_RACE_B_OFFSETS = [-8, -4, 0, 4, 8]
HUNT_ABORT_BOND = 2048
HUNT_ABORT_HOLD = 120.0
EARLY_CAP_FRAC = 0.5
WALL_ABORT_FRAC = (0.55, 0.78)
KEEP_ALL = os.environ.get('HQP_KEEP_ALL', '0') == '1'
HUNT_GPU_CAP_GB = 82.0
MODULE_RACE_SECONDS = 840.0
RACE_FIRST_ELIM = 360.0
RACE_ELIM_EVERY = 120.0

def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_plan(path: Path, *, qasm: Path, plan_id: str, cfg: dict, centre: int, window: tuple[int, int], early_stop: int, ckpt_layers: int, device: str, dtype: str, wall: float) -> Path:
    txt = f'''schema_version = "1.0"\nplan_id = "{plan_id}"\ninstance_id = "local"\ntarget_access = "blind"\npurpose = "autonomous solve"\nqasm_path = "{qasm}"\nqasm_sha256 = "{sha256_file(qasm)}"\n\n[engine]\nname = "mpo-unswap"\nmode = "baseline"\nmax_bond = {int(cfg['max_bond'])}\ncutoff = {cfg['cutoff']}\nfinal_cutoff = {cfg['final_cutoff']}\nunswap_threshold = {float(cfg['unswap_threshold'])}\nmax_its = {int(cfg['max_its'])}\nbeam_size = {int(cfg['beam_size'])}\ntopk = {int(cfg['topk'])}\nordering_seeds = [{int(cfg.get('ordering_seed', ORDERING_SEED))}]\nearly_stopping_gates = {int(early_stop)}\nsabre_trials = {int(cfg['sabre_trials'])}\npost_sabre_trials = {int(cfg['post_sabre_trials'])}\ncenter_ratio = {int(centre)}\nstaged_transpilation = false\nabsorb_window = [{int(window[0])}, {int(window[1])}]\ncheckpoint_every_seconds = 60.0\ncheckpoint_every_layers = {int(ckpt_layers)}\n\n[execution]\nenabled = true\nwall_seconds = {float(wall)}\ndtype = "{dtype}"\ndevice = "{device}"\nbackend = "{('torch' if device != 'cpu' else 'numpy')}"\n'''
    path.write_text(txt)
    return path

class EngineRun:

    def __init__(self, tag: str, plan: Path, qasm: Path, out_root: Path, flags: list[str], *, waist_dir: Path | None, waist_maxbond: int, resume_ckpt: Path | None=None):
        self.tag, self.plan, self.qasm, self.out_root, self.flags = (tag, plan, qasm, out_root, flags)
        self.waist_dir, self.waist_maxbond, self.resume_ckpt = (waist_dir, waist_maxbond, resume_ckpt)
        self.proc: subprocess.Popen | None = None
        self.t0 = 0.0
        self.events_path: Path | None = None
        self._ev_pos = 0
        self.stats = dict(blocks=0, L=0, R=0, bond=0, elems=0.0, retained=None, drains=0, cycles=0, elapsed=0.0, termination=None, early_stop=None, layers=0, last_side=None)
        self.hist: list[tuple[float, int, int]] = []
        self._prev_abs = None

    def start(self) -> None:
        env = capped_env(PARALLEL_HUNT_ARMS)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        if self.waist_dir is not None:
            self.waist_dir.mkdir(parents=True, exist_ok=True)
            env['WAIST_KEEP'] = str(self.waist_dir)
            env['WAIST_KEEP_MAXBOND'] = str(self.waist_maxbond)
        else:
            env.pop('WAIST_KEEP', None)
        self.out_root.mkdir(parents=True, exist_ok=True)
        runner = HERE / 'scripts' / 'engine_run_gram.py'
        if self.resume_ckpt is not None:
            cmd = [sys.executable, str(runner), *self.flags, 'resume', str(self.resume_ckpt), '--plan', str(self.plan), '--qasm', str(self.qasm), '--out-root', str(self.out_root)]
        else:
            cmd = [sys.executable, str(runner), *self.flags, 'run', str(self.plan), '--qasm', str(self.qasm), '--out-root', str(self.out_root)]
        self.stdout = open(self.out_root / f'{self.tag}.stdout.log', 'ab')
        self.proc = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=self.stdout, stderr=subprocess.STDOUT, start_new_session=True)
        self.t0 = time.time()
        log(f"[{self.tag}] started pid {self.proc.pid}: {' '.join(cmd[2:])}")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def kill(self, why: str) -> None:
        if not self.alive():
            return
        log(f'[{self.tag}] stopping ({why})')
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            for _ in range(30):
                if self.proc.poll() is not None:
                    break
                time.sleep(1)
            if self.proc.poll() is None:
                os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()

    def _find_events(self) -> Path | None:
        if self.events_path and self.events_path.exists():
            return self.events_path
        cands = sorted(self.out_root.glob('**/events.jsonl'), key=lambda p: p.stat().st_mtime)
        cands = [c for c in cands if 'core_compress' not in str(c)]
        self.events_path = cands[-1] if cands else None
        return self.events_path

    def poll(self) -> dict:
        ev = self._find_events()
        if ev is None:
            return self.stats
        with open(ev, 'rb') as fh:
            fh.seek(self._ev_pos)
            chunk = fh.read()
            self._ev_pos = fh.tell()
        s = self.stats
        for raw in chunk.splitlines():
            try:
                d = json.loads(raw)
            except ValueError:
                continue
            k = d.get('event')
            if k == 'starvation_guard_drain':
                s['drains'] += 1
            elif k == 'unswap_cycle_end':
                s['cycles'] += 1
            elif k == 'early_stop':
                s['early_stop'] = d.get('reason', 'count')
            elif k == 'layer_absorbed':
                s['layers'] += 1
                b, side = (d.get('absorbed_total'), d.get('side'))
                if isinstance(b, int):
                    if self._prev_abs is not None and b > self._prev_abs:
                        if side == 'left':
                            s['L'] += b - self._prev_abs
                        elif side == 'right':
                            s['R'] += b - self._prev_abs
                        else:
                            s['L'] += (b - self._prev_abs) / 2
                            s['R'] += (b - self._prev_abs) / 2
                    self._prev_abs = b
                    s['blocks'] = b
                s['bond'] = int(d.get('max_bond') or 0)
                s['elems'] = float(d.get('total_elems') or 0)
                if d.get('retained_local_frobenius_log10') is not None:
                    s['retained'] = float(d['retained_local_frobenius_log10'])
                s['elapsed'] = float(d.get('elapsed_seconds') or 0)
                s['last_side'] = side
                self.hist.append((time.time() - self.t0, s['blocks'], s['bond']))
            if 'termination' in d and d.get('termination') not in (None, 'in_progress'):
                s['termination'] = d['termination']
        return s

    def pace(self, seconds: float) -> float | None:
        if not self.hist:
            return None
        _, b_end, _ = self.hist[-1]
        t_end = time.time() - self.t0
        if seconds >= t_end:
            t0, b0 = (0.0, 0)
        else:
            older = [h for h in self.hist if h[0] <= t_end - seconds]
            if not older:
                return None
            t0, b0, _ = older[-1]
        return 60.0 * (b_end - b0) / max(1e-09, t_end - t0)

    def candidates(self) -> Path | None:
        c = sorted(self.out_root.glob('**/candidates.json'))
        return c[-1] if c else None

    def checkpoints(self) -> list[Path]:
        return sorted(self.out_root.glob('**/checkpoints/seed*.ckpt'), key=lambda p: p.stat().st_mtime)

    def summary(self) -> str:
        s = self.stats
        return f"blk {s['blocks']} (L{int(s['L'])}/R{int(s['R'])}) bond {s['bond']} elems {s['elems']:.2e} ret {(None if s['retained'] is None else round(s['retained'], 4))} drains {s['drains']} cycles {s['cycles']} t {s['elapsed'] / 60:.1f}m"

def scan_object(path: Path, max_factor: int) -> dict | None:
    try:
        return _scan_object(path, max_factor)
    except Exception as exc:
        log(f'[scan] {Path(path).name}: {type(exc).__name__}: {str(exc)[:160]}; skipped')
        return None

def _scan_object(path: Path, max_factor: int) -> dict | None:
    from peaked.core_mpo import portable_to_chain
    from checkpoint_to_reduced_qasm import spectra, regions
    try:
        pay = pickle.load(open(path, 'rb'))
    except Exception:
        return None
    c = pay.get('counters') or {}
    try:
        chain = portable_to_chain(pay)
    except Exception:
        return None
    sp = spectra(chain)
    strong = set()
    for i, s in enumerate(sp):
        t = float((s ** 2).sum())
        if t > 0 and len(s) > 1 and (float((s[1:] ** 2).sum()) / t > 1e-09):
            strong.add(i)
    regs = regions(strong, len(chain))
    lf = max((len(r) for r in regs))
    if lf > max_factor:
        from checkpoint_to_reduced_qasm import region_gate
        from factor_split import split_factor
        sizes = []
        for r in regs:
            if len(r) > max_factor and len(r) <= 8:
                sizes.append(max((len(pc[0]) for pc in split_factor(region_gate(chain, r), len(r), 1e-06))))
            else:
                sizes.append(len(r))
        lf = max(sizes)
    return dict(path=str(path), work=int(c.get('work_ops_absorbed_total') or 0), layers=int(c.get('layers_absorbed') or 0), bond=max((int(a.shape[0]) for a in chain)), largest_factor=lf, n_multi=sum((1 for r in regs if len(r) > 1)), usable=lf <= max_factor)

class WaistScanner:

    def __init__(self, waist_dir: Path, max_factor: int, min_work: int=0):
        self.dir, self.max_factor, self.min_work = (waist_dir, max_factor, min_work)
        self.seen: set[str] = set()
        self.objects: list[dict] = []

    def scan(self, max_bond_to_load: int=8) -> None:
        idx = self.dir / 'index.tsv'
        if not idx.exists():
            return
        for line in idx.read_text(errors='replace').splitlines():
            f = line.split('\t')
            if len(f) < 7:
                continue
            path = f[6].strip()
            try:
                mb = int(f[4])
            except ValueError:
                continue
            if path in self.seen or mb > max_bond_to_load or (not Path(path).exists()):
                continue
            self.seen.add(path)
            o = scan_object(Path(path), self.max_factor)
            if o:
                self.objects.append(o)

    def best(self) -> dict | None:
        ok = [o for o in self.objects if o['usable'] and o['work'] >= max(1, self.min_work)]
        if not ok:
            return None
        return max(ok, key=lambda o: (o['work'], -o['largest_factor'], -o['bond']))

    def candidates(self, k: int=3, gap: int=8) -> list[dict]:
        ok = [o for o in self.objects if o['usable'] and o['work'] >= max(1, self.min_work)]
        ok.sort(key=lambda o: (-o['work'], o['largest_factor'], o['bond']))
        out: list[dict] = []
        for o in ok:
            if all((abs(o['work'] - c['work']) > gap for c in out)):
                out.append(o)
            if len(out) >= k:
                break
        return out

def pick_per_collapse(ranked: list[dict], n: int) -> list[dict]:
    groups: list[list[dict]] = []
    for c in ranked:
        for g in groups:
            if abs(int(c['work']) - int(g[0]['work'])) <= 8:
                g.append(c)
                break
        else:
            groups.append([c])
    out: list[dict] = []
    for g in groups:
        out.append(min(g, key=lambda c: (int(c.get('bond', 1 << 30)), ranked.index(c))))
        if len(out) >= n:
            break
    return out

class Solver:

    def __init__(self, qasm: Path, workdir: Path, *, wall: float, reserve: float, device: str, dtype: str, test_scale: float=1.0, unswap_scale: float=1.0):
        self.qasm, self.work = (Path(qasm).resolve(), Path(workdir).resolve())
        self.RECORD, self.RECORD_FLAGS = scale_unswap(RECORD, RECORD_FLAGS, unswap_scale)
        self._unswap_scale = unswap_scale
        self.SOLVE, self.SOLVE_FLAGS = scale_unswap(SOLVE, SOLVE_FLAGS, unswap_scale)
        self.deadline = START + wall - reserve
        self._n_solves = 0
        self.device, self.dtype = (device, dtype)
        self.ts = test_scale
        self.info: dict = {'stages': [], 'qasm_sha256': sha256_file(qasm)}
        self.best: dict | None = None
        self._cand_lock = threading.RLock()
        self.work.mkdir(parents=True, exist_ok=True)
        for d in ('schemas', 'evidence'):
            (HERE / d).mkdir(exist_ok=True)

    def left(self) -> float:
        return self.deadline - time.time()

    def record(self, **kw) -> None:
        kw['t'] = round(time.time() - START, 1)
        self.info['stages'].append(kw)

    def consider(self, bits: str, w1: float, w2: float, source: str) -> None:
        with self.__dict__.setdefault('_cand_lock', threading.RLock()):
            self._consider(bits, w1, w2, source)

    def _consider(self, bits: str, w1: float, w2: float, source: str) -> None:
        if not (math.isfinite(w1) and math.isfinite(w2)) or w1 <= 0:
            ratio = 0.0
        else:
            ratio = w1 / w2 if w2 > 0 else 1.0
        prev = getattr(self, '_cands', []) if self.best is not None else []
        self._cands = prev + [{'bits': bits, 'w1': w1, 'w2': w2, 'ratio': ratio, 'source': source}]
        top = max(self._cands, key=lambda c: c['ratio'])
        if top['ratio'] >= MARGIN:
            self.best = dict(top, agree=sum((1 for c in self._cands if c['bits'] == top['bits'])))
        else:
            groups: dict[str, list[dict]] = {}
            for c in self._cands:
                if c['ratio'] >= VOTE_MIN_RATIO:
                    groups.setdefault(c['bits'], []).append(c)
            g = max(groups.values(), key=lambda g: (len(g), max((c['ratio'] for c in g)))) if groups else [top]
            if top['ratio'] >= 2.0 * max((c['ratio'] for c in g)):
                g = [top]
            self.best = dict(max(g, key=lambda c: c['ratio']), agree=len(g))
        log(f"candidate from {source}: top1/top2 {ratio:.2f} (weight {w1:.3g}); best so far from {self.best['source']} at {self.best['ratio']:.2f} (agreed by {self.best['agree']} source(s))")

    def forward_beam_stage(self) -> bool:
        bits_file = self.work / 'beam0.bits.json'
        bits_file.unlink(missing_ok=True)
        cmd = [sys.executable, str(HERE / 'scripts' / 'forward_beam_stage.py'), str(self.qasm), '--chi', str(FB_CHI), '--beam', str(FB_BEAM), '--orderings', str(FB_ORDERINGS), '--emit-bits', str(bits_file), '--out', str(self.work / 'beam0.json')]
        limit = max(60.0, min(FB_SECONDS * self.ts, self.left() - 120 * self.ts))
        log(f'[beam0] no CZ-mirror module found: forward beam at chi {FB_CHI}, {FB_ORDERINGS} orderings (<= {limit:.0f} s)')
        try:
            r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, env=capped_env(), timeout=limit)
        except subprocess.TimeoutExpired:
            log(f'[beam0] exceeded {limit:.0f} s; skipped')
            return False
        for line in r.stdout.splitlines():
            if line.startswith('[fb]'):
                log('  ' + line[:200])
        if r.returncode != 0 or not bits_file.exists():
            log(f'[beam0] FAILED rc {r.returncode}: {r.stderr[-300:]}')
            return False
        try:
            d = json.loads(bits_file.read_text())
            bits, w, agree, stable = (d['bits'], d['weights'], int(d.get('agree', 0)), bool(d.get('stable', False)))
            if not bits or not w:
                raise ValueError('empty beam')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log(f'[beam0] output unusable: {exc}')
            return False
        w1, w2 = (float(w[0]), float(w[1]) if len(w) > 1 else 0.0)
        self.consider(bits[0], w1, w2, 'beam0')
        self.record(stage='beam0', agree=agree, stable=stable, gap=d.get('gap'), ratio=w1 / w2 if w2 > 0 else None)
        if agree >= 2 or (stable and agree >= 1):
            with self._cand_lock:
                self.best = dict(bits=bits[0], w1=w1, w2=w2, ratio=max(MARGIN, w1 / w2 if w2 > 0 else MARGIN), source='beam0', agree=agree, accepted='cross-ordering product, leave-one-out stable')
            log(f'[beam0] ACCEPTED: product winner (rank-1 in {agree}/{FB_ORDERINGS} orderings, leave-one-out stable {stable})')
            return True
        log(f'[beam0] product winner not leave-one-out stable (rank-1 in {agree}/{FB_ORDERINGS}); kept as a candidate, continuing with the hunt')
        return False

    def structure(self):
        from hqp_structure import load_circuit, analyse
        raw = load_circuit(str(self.qasm))
        st = analyse(raw)
        self.n, self.N = (st.n_qubits, st.n_blocks)
        self.modules = st.modules
        log(f'structure: {st.n_qubits} qubits, {st.n_raw} raw ops, {st.n_blocks} blocks, method {st.method}; module centres {[m.centre_block for m in st.modules]} (seams {st.seam_centres})')
        self.record(stage='structure', n_qubits=st.n_qubits, blocks=st.n_blocks, method=st.method, centres=[m.centre_block for m in st.modules], seams=st.seam_centres)

    def launch(self, tag: str, qasm: Path, cfg: dict, flags: list[str], centre: int, window: tuple[int, int], *, seconds: float, early_stop: int, ckpt_layers: int, waist_maxbond: int | None, tail_stop: bool, resume: Path | None=None) -> EngineRun:
        d = self.work / tag
        k = 1
        while d.exists():
            k += 1
            d = self.work / f'{tag}_r{k}'
        tag = d.name
        d.mkdir(parents=True)
        plan = write_plan(d / 'plan.toml', qasm=qasm, plan_id=f'plan.local.hqp.{tag}', cfg=cfg, centre=centre, window=window, early_stop=early_stop, ckpt_layers=ckpt_layers, device=self.device, dtype=self.dtype, wall=max(120.0, seconds + 600))
        fl = list(flags) + (['--tail-stop', TAIL_STOP] if tail_stop else [])
        run = EngineRun(tag, plan, qasm, d / 'out', fl, waist_dir=d / 'waists' if waist_maxbond else None, waist_maxbond=waist_maxbond or 0, resume_ckpt=resume)
        run.start()
        self._runs = getattr(self, '_runs', []) + [run]
        return run

    def discard(self, runs) -> None:
        if KEEP_ALL:
            return
        keep = getattr(self, '_keep_paths', set())
        freed = 0
        for r in runs if isinstance(runs, (list, tuple)) else [runs]:
            if r is None or r.alive():
                continue
            root = Path(r.out_root).parent
            if not root.exists():
                continue
            for f in sorted(root.rglob('*')):
                if f.is_file() and str(f.resolve()) not in keep and (f.suffix in ('.ckpt', '.pkl', '.npz', '.pt')):
                    try:
                        freed += f.stat().st_size
                        f.unlink()
                    except OSError:
                        pass
        if freed:
            log(f'[disk] discarded {freed / 1000000000.0:.2f} GB of finished runs')

    def disk_note(self) -> None:
        try:
            used = sum((f.stat().st_size for f in self.work.rglob('*') if f.is_file()))
            free = shutil.disk_usage(str(self.work)).free
        except OSError:
            return
        if abs(used - getattr(self, '_disk_last', -1e+18)) >= 500000000.0:
            self._disk_last = used
            self._disk_peak = max(used, getattr(self, '_disk_peak', 0))
            self.info['disk_peak_gb'] = round(self._disk_peak / 1000000000.0, 2)
            log(f'[disk] workdir {used / 1000000000.0:.2f} GB (peak {self._disk_peak / 1000000000.0:.2f} GB), free {free / 1000000000.0:.1f} GB')

    def guard_disk(self, live: EngineRun | None=None) -> None:
        self.disk_note()
        if KEEP_ALL:
            return
        try:
            free = shutil.disk_usage(str(self.work)).free
        except OSError:
            return
        if free >= DISK_MIN_FREE:
            return
        lives = live if isinstance(live, (list, tuple)) else [live] if live is not None else []
        self.discard([r for r in getattr(self, '_runs', []) if r not in lives])
        live_dirs = [Path(r.waist_dir).resolve() for r in lives if getattr(r, 'waist_dir', None)]
        live_dir = live_dirs[0] if live_dirs else None
        freed = 0
        for d in sorted(self.work.glob('**/waists')):
            if d.resolve() in live_dirs:
                continue
            for f in d.glob('seq*.ckpt'):
                if str(f.resolve()) in getattr(self, '_keep_paths', set()):
                    continue
                freed += f.stat().st_size
                f.unlink(missing_ok=True)
        for live_dir in live_dirs:
            if shutil.disk_usage(str(self.work)).free >= DISK_MIN_FREE:
                break
            files = sorted(Path(live_dir).glob('seq*.ckpt'))
            for f in files[:-40]:
                try:
                    if int(f.stem.rsplit('bond', 1)[-1]) <= 16:
                        continue
                except ValueError:
                    pass
                freed += f.stat().st_size
                f.unlink(missing_ok=True)
        log(f'[disk] free {free / 1000000000.0:.1f} GB < {DISK_MIN_FREE / 1000000000.0:.1f} GB: pruned {freed / 1000000000.0:.2f} GB of waists')

    def monitor(self, run: EngineRun, seconds: float, on_tick=None) -> EngineRun:
        t_end = min(time.time() + seconds, self.deadline)
        last_log = 0.0
        while run.alive():
            time.sleep(5)
            run.poll()
            now = time.time()
            if now - last_log >= 60:
                log(f'[{run.tag}] {run.summary()}')
                last_log = now
                self.guard_disk(run)
            if on_tick is not None and on_tick(run):
                run.kill('stop rule')
                break
            if now >= t_end:
                run.kill('time box')
                break
        run.poll()
        log(f"[{run.tag}] ended: {run.summary()} termination {run.stats['termination']} early_stop {run.stats['early_stop']}")
        return run

    def run_engine(self, tag, qasm, cfg, flags, centre, window, *, seconds, early_stop, ckpt_layers, waist_maxbond, tail_stop, on_tick=None, resume=None) -> EngineRun:
        run = self.launch(tag, qasm, cfg, flags, centre, window, seconds=seconds, early_stop=early_stop, ckpt_layers=ckpt_layers, waist_maxbond=waist_maxbond, tail_stop=tail_stop, resume=resume)
        return self.monitor(run, seconds, on_tick)

    @staticmethod
    def gpu_used_gb() -> float:
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            return float(out[0]) / 1024.0 if out else 0.0
        except Exception:
            return 0.0

    def race(self, tag: str, qasm: Path, cfg: dict, flags: list[str], centres: list[int], window: tuple[int, int], *, seconds: float, first_elim: float, elim_every: float, early_stop: int, waist_maxbond: int, tail_stop: bool, N: int, gpu_cap_gb: float=70.0, windows: dict | None=None, keep: int=1) -> tuple[int, EngineRun]:
        runs: dict[int, EngineRun] = {}
        for c in centres:
            c = max(window[0] + 5, min(window[1] - 5, c))
            if c in runs:
                continue
            runs[c] = self.launch(f'{tag}_c{c}', qasm, cfg, flags, c, (windows or {}).get(c, window), seconds=seconds + 3600, early_stop=early_stop, ckpt_layers=1, waist_maxbond=waist_maxbond, tail_stop=tail_stop)
            time.sleep(2)
        t0 = time.time()
        next_elim = t0 + first_elim
        last_log = 0.0

        def score(r: EngineRun) -> float:
            st = r.stats
            if st['retained'] is not None and (not math.isfinite(st['retained'])):
                return -1e+18
            pen = (1000000000.0 if st['bond'] >= cfg['max_bond'] else 0) + (1000000.0 if (st['retained'] or 0) < -0.3 else 0)
            pen += 1000000.0 * st['drains']
            return st['blocks'] - pen - 1e-07 * st['elems'] - (1000000000.0 if not r.alive() and r.candidates() is None else 0)
        while True:
            time.sleep(5)
            if self.best and self.best['ratio'] >= MARGIN:
                break
            for r in runs.values():
                r.poll()
                ret = r.stats['retained']
                if r.alive() and ret is not None and (not math.isfinite(ret)):
                    r.kill('non-finite state')
                if r.candidates() is not None and r.stats['termination'] in ('completed', 'early_stopped'):
                    self.read_candidates(r, f'{r.tag}-engine')
            alive = [c for c, r in runs.items() if r.alive()]
            now = time.time()
            if now - last_log >= 60:
                log(f'[{tag}] ' + ' | '.join((f"c{c}: {runs[c].stats['blocks']}b bond {runs[c].stats['bond']}" for c in sorted(runs))) + f' | gpu {self.gpu_used_gb():.0f} GB')
                last_log = now
                self.guard_disk([runs[c] for c in alive])
            if len(alive) <= max(1, keep) or now - t0 >= seconds or now >= self.deadline:
                break
            if self.gpu_used_gb() > gpu_cap_gb or now >= next_elim:
                worst = min(alive, key=lambda c: score(runs[c]))
                runs[worst].kill(f"race elimination ({runs[worst].stats['blocks']} blocks)")
                next_elim = now + elim_every
        ranked = sorted(runs, key=lambda c: -score(runs[c]))
        winner = ranked[0]
        kept = [c for c in ranked if runs[c].alive()][:max(1, keep)]
        if winner not in kept:
            kept = [winner] + kept[:max(0, keep - 1)]
        for c in ranked:
            if c not in kept and runs[c].alive():
                runs[c].kill('race over')
        self.record(stage=tag, results=[(c, runs[c].stats['blocks'], runs[c].stats['bond'], runs[c].stats['retained']) for c in ranked], winner=winner, kept=kept)
        log(f'[{tag}] winner centre {winner}: {runs[winner].summary()}' + (f'; also kept {kept[1:]}' if len(kept) > 1 else ''))
        for c in ranked:
            if c not in kept:
                if not KEEP_ALL:
                    shutil.rmtree(runs[c].out_root.parent, ignore_errors=True)
        self._race_kept = [runs[c] for c in kept]
        return (winner, runs[winner])

    def hunt(self, run: EngineRun, seconds: float) -> dict | None:
        return self.hunt_many([run], seconds)

    def hunt_many(self, runs: list[EngineRun], seconds: float, spawn=None) -> dict | None:
        mf = getattr(self, 'max_factor', MAX_FACTOR)
        minw = max(30, int(MIN_OBJECT_FRACTION * self.N))
        arms = [(r, WaistScanner(r.waist_dir, mf, min_work=minw), {'t_obj': None, 'ret_obj': None, 'best': None}) for r in runs]

        def tick(r: EngineRun, scanner: WaistScanner, state: dict) -> bool:
            scanner.scan()
            b = scanner.best()
            st = r.stats
            if b and (state['best'] is None or b['work'] > state['best']['work']):
                state['best'] = b
                state['t_obj'], state['ret_obj'] = (time.time(), st['retained'])
                log(f"[hunt] usable object ({r.tag}): work {b['work']} bond {b['bond']} largest factor {b['largest_factor']} ({Path(b['path']).name})")
            if getattr(self, 'hunt_only', 0):
                return st['drains'] >= 6 or (st['bond'] >= RECORD['max_bond'] and (st['retained'] or 0) < -0.5)
            if state['best'] is None:
                if st['bond'] >= HUNT_ABORT_BOND:
                    state['t_big'] = state.get('t_big') or time.time()
                    if time.time() - state['t_big'] >= HUNT_ABORT_HOLD * self.ts:
                        log(f"[hunt] {r.tag}: bond {st['bond']} >= {HUNT_ABORT_BOND} for {HUNT_ABORT_HOLD * self.ts:.0f} s without a usable object: early abort")
                        r.cap_abort = True
                        return True
                else:
                    state['t_big'] = None
                p8 = r.pace(480 * self.ts)
                if p8 is not None and p8 < 1.0 and (time.time() - r.t0 > 1200 * self.ts):
                    return True
                return st['drains'] >= 6 or (st['bond'] >= RECORD['max_bond'] and (st['retained'] or 0) < -0.5)
            p5, pm = (r.pace(300 * self.ts), r.pace(1000000000.0))
            slow = p5 is not None and pm is not None and (pm > 0) and (p5 < 0.25 * pm) and (time.time() - state['t_obj'] > 120 * self.ts)
            big = st['bond'] >= RECORD['max_bond'] // 2
            lossy = state['ret_obj'] is not None and st['retained'] is not None and (st['retained'] < state['ret_obj'] - 0.1)
            return slow or big or lossy or (st['drains'] >= 3)
        t_end = min(time.time() + seconds, self.deadline)
        last_log = 0.0
        t_first_obj = None

        def fill() -> None:
            if spawn is None or t_first_obj is not None:
                return
            while sum((1 for r, _, _ in arms if r.alive())) < PARALLEL_HUNT_ARMS:
                r = spawn()
                if r is None:
                    return
                arms.append((r, WaistScanner(r.waist_dir, mf, min_work=minw), {'t_obj': None, 'ret_obj': None, 'best': None}))
        fill()
        while any((r.alive() for r, _, _ in arms)):
            time.sleep(5)
            now = time.time()
            if self.best and self.best['ratio'] >= MARGIN:
                for r, _, _ in arms:
                    if r.alive():
                        r.kill('answer at the margin')
                break
            for r, sc, stt in arms:
                if r.alive():
                    r.poll()
                    if tick(r, sc, stt):
                        r.kill('stop rule')
            if t_first_obj is None and any((stt['best'] is not None for _, _, stt in arms)):
                t_first_obj = now
                log(f'[portfolio] first usable object; other arms get {PORTFOLIO_GRACE * self.ts:.0f} s to find a deeper one')
            if t_first_obj is not None and now - t_first_obj > PORTFOLIO_GRACE * self.ts:
                for r, _, stt in arms:
                    if r.alive() and stt['best'] is None:
                        r.kill('portfolio grace over')
            fill()
            alive = [(r, stt) for r, _, stt in arms if r.alive()]
            if now - last_log >= 60:
                for r, _ in alive:
                    log(f'[{r.tag}] {r.summary()}')
                last_log = now
                self.guard_disk([r for r, _ in alive])
                if len(alive) > 1 and self.gpu_used_gb() > HUNT_GPU_CAP_GB:
                    victim = max(alive, key=lambda x: (x[1]['best'] is None, x[0].stats['bond']))[0]
                    victim.kill(f'GPU memory {self.gpu_used_gb():.0f} GB > {HUNT_GPU_CAP_GB:.0f} GB with two arms')
            if now >= t_end:
                for r, _ in alive:
                    r.kill('time box')
                break
        merged: list[dict] = []
        for r, sc, stt in arms:
            r.poll()
            sc.scan(max_bond_to_load=16)
            for c in sc.candidates(OBJECT_CANDIDATES):
                c['dst_centre'] = getattr(r, 'dst_centre', None)
                merged.append(c)
            log(f"[{r.tag}] ended: {r.summary()} termination {r.stats['termination']}; objects {len(sc.objects)}")
        merged.sort(key=lambda o: (-o['work'], o['largest_factor'], o['bond']))
        uniq: list[dict] = []
        for o in merged:
            if all((abs(o['work'] - u['work']) > 2 or o['bond'] != u['bond'] or o['largest_factor'] != u['largest_factor'] for u in uniq)):
                uniq.append(o)
        self._candidates = uniq[:OBJECT_CANDIDATES]
        pool: dict[str, dict] = {}
        for r, sc, stt in arms:
            for o in sc.objects:
                if o['usable'] and o['work'] >= max(1, sc.min_work):
                    pool.setdefault(str(Path(o['path']).resolve()), dict(o, dst_centre=getattr(r, 'dst_centre', None)))
        self._pool = sorted(pool.values(), key=lambda o: (-o['work'], o['bond']))[:POOL_MAX]
        with self.__dict__.setdefault('_cand_lock', threading.RLock()):
            self._keep_paths = set(getattr(self, '_keep_paths', set())) | {str(Path(o['path']).resolve()) for o in self._pool}
            self._keep_paths = set(self._keep_paths) | {str(Path(c['path']).resolve()) for c in self._candidates}
        best = self._candidates[0] if self._candidates else None
        self.record(stage='hunt', centre=[r.tag for r, _, _ in arms], run=[r.summary() for r, _, _ in arms], objects=sum((len(sc.objects) for _, sc, _ in arms)), best=best, candidates=[{k: c[k] for k in ('work', 'bond', 'largest_factor')} for c in self._candidates])
        if len(self._candidates) > 1:
            log('[hunt] object candidates (distinct collapses): ' + ', '.join((f"{c['work']} blocks/bond {c['bond']}/factor {c['largest_factor']}" for c in self._candidates)))
        return best

    def reduce(self, obj: dict, locate: list[int], out: Path | None=None, *, timeout: float | None=None, share: int=1) -> tuple[Path, dict] | None:
        out = out or self.work / 'reduced.qasm'
        cmd = [sys.executable, str(HERE / 'scripts' / 'checkpoint_to_reduced_qasm.py'), obj['path'], '--qasm', str(self.qasm), '--out', str(out), '--locate', ','.join((str(x) for x in locate)), '--max-factor', str(getattr(self, 'max_factor', MAX_FACTOR))]
        log(f"[reduce] {Path(obj['path']).name} (work {obj['work']}, bond {obj['bond']}, factor {obj['largest_factor']})")
        limit = max(60.0, min(1200.0 * self.ts, self.left() - 120 * self.ts))
        if timeout is not None:
            limit = min(limit, timeout)
        if limit < 30.0:
            log('[reduce] no time left in the ranking budget; object skipped')
            return None
        try:
            r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, env=capped_env(share), timeout=limit)
        except subprocess.TimeoutExpired:
            log(f'[reduce] exceeded {limit:.0f} s; object skipped')
            return None
        for line in r.stdout.splitlines():
            if line.startswith('[red]'):
                log('  ' + line[:300])
        side = out.with_suffix('.qasm.json')
        if r.returncode != 0 or not out.exists() or (not side.exists()):
            log(f'[reduce] FAILED rc {r.returncode}: {r.stderr[-500:]}')
            return None
        try:
            meta = json.loads(side.read_text())
        except (OSError, ValueError) as exc:
            log(f'[reduce] side file unreadable: {exc}')
            return None
        if not meta.get('ok'):
            log(f"[reduce] checks failed: {meta.get('checks')}")
            return None
        self.record(stage='reduce', object=Path(obj['path']).name, reduced_blocks=meta.get('reduced_blocks'), object_block=meta.get('object_reduced_block'), located=meta.get('located'), checks=meta.get('checks'))
        return (out, meta)

    def read_candidates(self, run: EngineRun, source: str) -> bool:
        cj = run.candidates()
        if cj is None:
            return False
        key = (str(cj.resolve()), cj.stat().st_mtime_ns)
        seen = getattr(self, '_read_cands', set())
        if key in seen:
            return True
        self._read_cands = seen | {key}
        try:
            rows = json.load(open(cj))['orderings'][0]['candidates']
        except Exception as exc:
            log(f'[{source}] candidates unreadable: {exc}')
            return False
        if not rows:
            return False
        w = [float(r.get('engine_weight') or 0) for r in rows]
        self.consider(rows[0]['logical_bits'], w[0], w[1] if len(w) > 1 else 0.0, source)
        return True

    def readout(self, ckpt: Path, source: str, chi: int=0, threads: int=0) -> bool:
        chi = chi or READOUT_CHI
        bits_file = self.work / f'{source}.bits.json'
        cmd = [sys.executable, str(HERE / 'scripts' / 'checkpoint_readout.py'), str(ckpt), '--chi', str(chi), '--beam', '4096', '--emit-bits', str(bits_file), '--out', str(self.work / f'{source}.readout.json')]
        log(f'[{source}] state readout of {ckpt.name} at chi {chi}')
        limit = max(60.0, min(READOUT_SECONDS * self.ts, self.left() - 120 * self.ts))
        bits_file.unlink(missing_ok=True)
        try:
            env = capped_env()
            if threads:
                for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
                    env[k] = str(threads)
            r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, env=env, timeout=limit)
        except subprocess.TimeoutExpired:
            if chi > READOUT_CHI_FALLBACK and self.left() > 300 * self.ts:
                log(f'[{source}] readout exceeded {limit:.0f} s at chi {chi}; retrying at chi {READOUT_CHI_FALLBACK}')
                return self.readout(ckpt, source, chi=READOUT_CHI_FALLBACK, threads=threads)
            log(f'[{source}] readout exceeded {limit:.0f} s; skipped')
            return False
        for line in r.stdout.splitlines():
            if line.startswith('[ro]'):
                log('  ' + line[:200])
        if not bits_file.exists():
            log(f'[{source}] readout FAILED rc {r.returncode}: {r.stderr[-400:]}')
            return False
        if r.returncode != 0:
            log(f'[{source}] readout rc {r.returncode} after writing its bits (kept): {r.stderr[-200:]}')
        try:
            d = json.loads(bits_file.read_text())
            bits, wts = (d['bits'], d['weights'])
            if not bits or not wts:
                raise ValueError('empty beam')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log(f'[{source}] readout output unusable: {exc}')
            return False
        self.consider(bits[0], wts[0], wts[1] if len(wts) > 1 else 0.0, source)
        return True

    def readout_async(self, ckpt: Path, source: str) -> None:
        alive = [t for t in getattr(self, '_readouts', []) if t.is_alive()]
        if len(alive) >= MAX_BG_READOUTS:
            self.join_readouts()
        keep = str(Path(ckpt).resolve())
        lock = self.__dict__.setdefault('_cand_lock', threading.RLock())
        with lock:
            self._keep_paths = set(getattr(self, '_keep_paths', set())) | {keep}

        def work():
            try:
                self.readout(ckpt, source, threads=READOUT_BG_THREADS)
            except Exception as exc:
                log(f'[{source}] readout error: {type(exc).__name__}: {exc}')
            finally:
                with lock:
                    self._keep_paths = set(self._keep_paths) - {keep}
                if not KEEP_ALL:
                    Path(keep).unlink(missing_ok=True)
        th = threading.Thread(target=work, name=source, daemon=True)
        th.start()
        self._readouts = [t for t in getattr(self, '_readouts', []) if t.is_alive()] + [th]
        log(f'[{source}] readout started in the background ({READOUT_BG_THREADS} CPU threads)')

    def join_readouts(self, reserve: float=60.0) -> None:
        for th in getattr(self, '_readouts', []):
            if th.is_alive():
                th.join(timeout=max(0.0, self.left() - reserve * self.ts))
        self._readouts = [t for t in getattr(self, '_readouts', []) if t.is_alive()]

    def twosided_rescan(self) -> None:
        if self.best is not None and self.best['ratio'] >= VOTE_MIN_RATIO:
            return
        if self.left() < 180 * self.ts:
            return
        best_src, best_work = (None, -1)
        for rj in self.work.glob('*-readout.readout.json'):
            try:
                d = json.loads(rj.read_text())
            except (OSError, ValueError):
                continue
            src = rj.name[:-len('.readout.json')]
            bits_file = self.work / f'{src}.bits.json'
            ck = d.get('checkpoint')
            if not bits_file.exists() or not ck or (not Path(ck).exists()):
                continue
            work = int((d.get('counters') or {}).get('work_ops_absorbed_total') or 0)
            if work > best_work:
                best_src, best_work = ((src, ck), work)
        if best_src is None:
            return
        src, ck = best_src
        out = self.work / f'{src}.twosided.json'
        out.unlink(missing_ok=True)
        dev = self.device if str(self.device).startswith('cuda') else 'cpu'
        cmd = [sys.executable, str(HERE / 'scripts' / 'checkpoint_twosided.py'), str(ck), '--chi', str(READOUT_CHI), '--bits', str(self.work / f'{src}.bits.json'), '--top', str(TS_TOP), '--device', dev, '--out', str(out)]
        limit = max(60.0, min(TS_SECONDS * self.ts, self.left() - 120 * self.ts))
        log(f"[twosided] best candidate is junk-class: rescoring {src}'s top {TS_TOP} through the pending gates (<= {limit:.0f} s, {dev})")
        try:
            r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, env=capped_env(), timeout=limit)
        except subprocess.TimeoutExpired:
            log(f'[twosided] exceeded {limit:.0f} s; skipped')
            return
        if r.returncode != 0 or not out.exists():
            log(f"[twosided] FAILED rc {r.returncode}: {(r.stderr or '')[-200:].strip()}")
            return
        try:
            res = json.loads(out.read_text())
            rows = json.loads((self.work / f'{src}.bits.json').read_text())['bits'][:TS_TOP]
            rel = [x if x is not None else -1.0 for x in res['scores_rel']]
        except (OSError, ValueError, KeyError) as exc:
            log(f'[twosided] unreadable result: {exc}')
            return
        ratio = res.get('twosided_ratio') or 0.0
        win = max(range(len(rel)), key=lambda i: rel[i]) if rel else 0
        log(f"[twosided] two-sided top1/top2 {ratio:.2f}; beam rank-1 is two-sided rank {res.get('twosided_rank_of_beam_rank1')}; winner = beam rank {win + 1}")
        self.record(stage='twosided', source=src, ratio=ratio, winner_beam_rank=win + 1, beam_rank1_ts_rank=res.get('twosided_rank_of_beam_rank1'))
        if ratio >= TS_ACCEPT and win < len(rows):
            self._consider(rows[win], float(ratio), 1.0, f'{src}-twosided')
        else:
            log(f"[twosided] below the accept bar ({TS_ACCEPT}): the beam's pick stands")

    def best_waist(self, run: EngineRun, maxbond: int=512) -> Path | None:
        idx = run.waist_dir / 'index.tsv' if run.waist_dir else None
        best = None
        if idx and idx.exists():
            for line in idx.read_text(errors='replace').splitlines():
                f = line.split('\t')
                if len(f) < 7:
                    continue
                try:
                    seq, mb = (int(f[0]), int(f[4]))
                except ValueError:
                    continue
                if mb <= maxbond and Path(f[6].strip()).exists() and (best is None or seq > best[0]):
                    best = (seq, Path(f[6].strip()))
        if best:
            return best[1]
        cks = run.checkpoints()
        return cks[-1] if cks else None

    def solve_reduced(self, run, seconds: float, tag: str='solve') -> None:
        runs = run if isinstance(run, (list, tuple)) else [run]

        def stop_rule(r: EngineRun) -> bool:
            st = r.stats
            p8 = r.pace(480 * self.ts)
            stalled = p8 is not None and p8 < 1.0 and (time.time() - r.t0 > 900 * self.ts)
            ret = st['retained']
            if ret is not None and (not math.isfinite(ret)):
                return True
            hi = getattr(r, '_t_hi_bond', None)
            if st['bond'] >= HUNT_ABORT_BOND:
                if hi is None:
                    r._t_hi_bond = time.time()
                elif time.time() - hi >= HUNT_ABORT_HOLD * self.ts:
                    log(f"[{r.tag}] bond {st['bond']} >= {HUNT_ABORT_BOND} for {HUNT_ABORT_HOLD * self.ts:.0f} s: stop")
                    return True
            else:
                r._t_hi_bond = None
            return st['bond'] >= SOLVE['max_bond'] and (ret or 0) < -0.5 or st['drains'] >= SOLVE_MAX_DRAINS or (stalled and st['drains'] >= 3)
        t_end = min(time.time() + seconds, self.deadline)
        last_log, read = (0.0, set())
        while any((r.alive() for r in runs)):
            time.sleep(5)
            now = time.time()
            for r in runs:
                if not r.alive():
                    continue
                r.poll()
                if stop_rule(r):
                    r.kill('stop rule')
            for i, r in enumerate(runs):
                if i not in read and (not r.alive()):
                    r.poll()
                    read.add(i)
                    self.read_candidates(r, f'{tag}-engine' + (f'-arm{i}' if len(runs) > 1 else ''))
            if self.best and self.best['ratio'] >= MARGIN:
                for r in runs:
                    if r.alive():
                        r.kill(f"{self.best['source']} reached the margin")
                break
            if now - last_log >= 60:
                for r in runs:
                    if r.alive():
                        log(f'[{r.tag}] {r.summary()}')
                last_log = now
                self.guard_disk([r for r in runs if r.alive()])
            if now >= t_end:
                for r in runs:
                    if r.alive():
                        r.kill('time box')
                break
        for i, r in enumerate(runs):
            r.poll()
            if i not in read:
                self.read_candidates(r, f'{tag}-engine' + (f'-arm{i}' if len(runs) > 1 else ''))
            self.record(stage=tag, centre=r.tag, run=r.summary(), termination=r.stats['termination'], early_stop=r.stats['early_stop'])
        if self.best and self.best['ratio'] >= MARGIN:
            return
        lead = max(runs, key=lambda r: r.stats.get('blocks') or 0)
        frac = (lead.stats.get('blocks') or 0) / max(1, getattr(self, '_n_red', 0) or 1)
        if getattr(self, '_n_red', 0) and frac < READOUT_MIN_FRACTION and (self.best is not None):
            log(f'[{tag}] no readout: the solve absorbed {frac:.0%} of the reduced circuit (< {READOUT_MIN_FRACTION:.0%})')
            return
        ck = self.best_waist(lead)
        if ck is not None and self.left() > 300 * self.ts:
            self.readout_async(ck, f'{tag}-readout')

    def anchor_window(self, src, dst, N: int) -> tuple[int, int]:
        if dst.cz_span is None:
            return self.gap_window(src, dst, N)
        if src.centre_block < dst.centre_block:
            return (0, min(N, dst.cz_span[0] + self.anchor_margin(N)))
        return (max(0, dst.cz_span[1] - self.anchor_margin(N)), N)

    def anchor_margin(self, N: int) -> int:
        return max(16, int(round(ANCHOR_MARGIN * getattr(self, 'margin_scale', 1.0))))

    def next_ticket_after_cap(self, mod: str) -> None:
        other = 'B' if mod == 'A' else 'A'
        self._cap_module = mod
        for i, t in enumerate(self._ladder):
            if t[0] == other:
                if i:
                    self._ladder.insert(0, self._ladder.pop(i))
                    log(f'[hunt] module {mod} aborted at the bond cap: module {other} ticket {t[1]:+d} moved to the front of the queue')
                return
        log(f'[hunt] module {mod} aborted at the bond cap, but no module-{other} ticket is queued')

    def next_ticket_same_module(self, mod: str) -> None:
        for i, t in enumerate(self._ladder):
            if t[0] == mod:
                if i:
                    self._ladder.insert(0, self._ladder.pop(i))
                    log(f'[hunt] module {mod} ticket failed without the wrong-module signature: module {mod} ticket {t[1]:+d} moved to the front of the queue (v5)')
                return

    def early_cap(self, blocks: int, window) -> bool:
        if not window or len(window) != 2 or window[1] - window[0] <= 0:
            return False
        return blocks / (window[1] - window[0]) < EARLY_CAP_FRAC

    def walled_short(self, blocks: int, window) -> bool:
        if not window or len(window) != 2 or window[1] - window[0] <= 0:
            return False
        f = blocks / (window[1] - window[0])
        return WALL_ABORT_FRAC[0] <= f <= WALL_ABORT_FRAC[1]

    def refill_ladder(self) -> bool:
        if self.left() < LADDER_MIN_LEFT * self.ts or (self.best and self.best['ratio'] >= MARGIN):
            return False
        if not getattr(self, '_attempt2', False):
            self.maybe_attempt2(force=True)
            if getattr(self, '_ladder', None):
                return True
        self._ladder_pass = getattr(self, '_ladder_pass', 1) + 1
        self._ladder = [(m, o, j * ATTEMPT2_CUTOFF_SCALE ** self._ladder_pass) for m, o, j in OFFSET_LADDER]
        log(f'=== ticket queue empty: pass {self._ladder_pass} with cutoff x{ATTEMPT2_CUTOFF_SCALE ** self._ladder_pass:.3f} ({len(self._ladder)} tickets, {self.left():.0f} s left)')
        return True

    def maybe_attempt2(self, force: bool=False) -> None:
        if getattr(self, '_attempt2', False) or (self.best and self.best['ratio'] >= MARGIN):
            return
        if not force and time.time() - START < ATTEMPT1_SECONDS * self.ts or self.left() < LADDER_MIN_LEFT * self.ts:
            return
        self._attempt2 = True
        thr = max(1, int(ATTEMPT2_UNSWAP * getattr(self, '_unswap_scale', 1.0)))
        self.RECORD = dict(self.RECORD, unswap_threshold=float(thr))
        fl = list(self.RECORD_FLAGS)
        if '--adaptive-unswap' in fl:
            i = fl.index('--adaptive-unswap')
            g, f, _ = fl[i + 1].split(',')
            fl[i + 1] = f'{g},{f},{thr}'
        self.RECORD_FLAGS = fl
        self.margin_scale = ATTEMPT2_MARGIN_SCALE
        self.race_b_offsets = list(ATTEMPT2_RACE_B_OFFSETS)
        if getattr(self, '_ladder', None) is not None:
            self._ladder = [(m, o, j * ATTEMPT2_CUTOFF_SCALE) for m, o, j in OFFSET_LADDER]
            if getattr(self, '_cap_module', None):
                self.next_ticket_after_cap(self._cap_module)
        log(f'=== attempt 2 (no answer after {time.time() - START:.0f} s): unswap {thr}, window margin x{ATTEMPT2_MARGIN_SCALE}, cutoff x{ATTEMPT2_CUTOFF_SCALE}, M1 seeds {self.race_b_offsets}; ticket queue restarted')

    def gap_window(self, src, dst, N: int) -> tuple[int, int]:
        m = GAP_MARGIN
        if src.span_blocks is None or dst.span_blocks is None:
            return (0, dst.centre_block) if src.centre_block < dst.centre_block else (dst.centre_block, N)
        if src.centre_block < dst.centre_block:
            return (0, min(N, src.span_blocks[1] + m))
        return (max(0, src.span_blocks[0] - m), N)

    def prepare(self) -> Path | None:
        out = self.work / 'prep'
        cmd = [sys.executable, str(HERE / 'scripts' / 'prep.py'), '--source', str(self.qasm), '--output', str(out), '--allow-derived-source']
        try:
            r = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True, env=capped_env(), timeout=900)
        except subprocess.TimeoutExpired:
            log('[prep] exceeded 900 s; solving the circuit as given')
            return None
        red = out / 'reduced.qasm'
        if r.returncode != 0 or not red.exists():
            log(f'[prep] failed rc {r.returncode}: {r.stderr[-300:]}')
            return None
        try:
            summ = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
        except Exception:
            summ = {}
        log(f"[prep] wrote {red} ({summ.get('pairs_removed', summ.get('n_pairs', '?'))} pairs removed)")
        self.record(stage='prepare', qasm=str(red), summary={k: v for k, v in summ.items() if isinstance(v, (int, float, str))})
        self._prep_applied = True
        return red

    def solve_from_reduced(self, red: Path, centres2: list[int], attempt: int=0) -> None:
        from hqp_structure import load_circuit, consolidate
        N_red = sum((1 for inst in consolidate(load_circuit(str(red))).data if len(inst.qubits) == 2))
        self._n_red = N_red
        c_solve, leader2 = self.race(f'raceB{attempt}', red, self.SOLVE, self.SOLVE_FLAGS, centres2, (0, N_red), seconds=RACE_SECONDS * self.ts, first_elim=RACE_FIRST_ELIM * self.ts, elim_every=RACE_ELIM_EVERY * self.ts, early_stop=EARLY_STOP_GATES, waist_maxbond=512, tail_stop=True, N=N_red, keep=SOLVE_KEEP)
        runs = getattr(self, '_race_kept', None) or [leader2]
        self.solve_reduced(runs if len(runs) > 1 else leader2, min(3600 * self.ts, self.left() - 900 * self.ts), tag=f'solve{attempt}')
        self.discard(runs)
        for k, c in enumerate(sorted(set(centres2) - {c_solve}, key=lambda c: (abs(c - c_solve), c))):
            if self.best and self.best['ratio'] >= MARGIN:
                return
            if self.left() < CONT_MIN_LEFT * self.ts:
                break
            log(f"=== M1 solve from {c_solve} did not reach the margin; full solve from the race's other seed {c} (same reduced circuit)")
            _c, lead = self.race(f'raceB{attempt}s{k + 1}', red, self.SOLVE, self.SOLVE_FLAGS, [c], (0, N_red), seconds=RACE_SECONDS * self.ts, first_elim=RACE_FIRST_ELIM * self.ts, elim_every=RACE_ELIM_EVERY * self.ts, early_stop=EARLY_STOP_GATES, waist_maxbond=512, tail_stop=True, N=N_red, keep=1)
            self.solve_reduced(lead, min(3600 * self.ts, self.left() - 900 * self.ts), tag=f'solve{attempt}s{k + 1}')
            self.discard(lead)

    def rank_by_reduced(self, cands: list[dict], dst) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor
        pool = getattr(self, '_pool', None) or cands
        self._prereduced = {}
        jobs = []
        for i, c in enumerate(pool):
            d_c = int(c.get('dst_centre') or getattr(dst, 'rule_centre', dst.centre_block))
            locate = sorted({d_c + o for o in getattr(self, 'race_b_offsets', RACE_B_OFFSETS)} | {d_c})
            jobs.append((i, c, [x for x in locate if 0 <= x < self.N]))
        if self.left() < LADDER_MIN_LEFT * self.ts:
            return cands
        workers = max(1, min(POOL_WORKERS, int(effective_cpus()) - 1))
        log(f'[reduce] ranking {len(jobs)} low-bond objects by reduced CZ count ({workers} CPU workers)')
        t0 = time.time()
        t_pool = t0 + POOL_SECONDS * self.ts

        def one(j):
            try:
                return (j, self.reduce(j[1], j[2], out=self.work / f'reduced_o{j[0]}.qasm', timeout=t_pool - time.time(), share=workers))
            except Exception as exc:
                log(f'[reduce] object {j[0]} failed: {type(exc).__name__}: {exc}')
                return (j, None)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(one, jobs))
        sized = []
        for (i, c, _loc), r in res:
            if r is None:
                continue
            try:
                cz = sum((1 for ln in open(r[0]) if ln.startswith('cz ')))
            except OSError:
                continue
            self._prereduced[str(Path(c['path']).resolve())] = r
            sized.append((cz, -int(c['work']), i, c))
        if not sized:
            return cands
        sized.sort(key=lambda t: t[:3])
        log(f'[reduce] {len(sized)}/{len(jobs)} objects reduced in {time.time() - t0:.0f} s; fewest CZ: ' + ', '.join((f"{c['work']} blocks/bond {c['bond']} -> {cz} CZ" for cz, _w, _i, c in sized[:5])))
        self.record(stage='rank_objects', ranked=[{'work': c['work'], 'bond': c['bond'], 'cz': cz} for cz, _w, _i, c in sized])
        out = pick_per_collapse([c for *_x, c in sized], OBJECT_CANDIDATES)
        log('[reduce] solve order (one per collapse, lowest bond within a collapse): ' + ', '.join((f"{c['work']} blocks/bond {c['bond']}" for c in out)))
        return out

    def solve_from_object(self, obj_path: Path, dst_centre: int, attempt: int=0) -> bool:
        obj = scan_object(Path(obj_path), getattr(self, 'max_factor', MAX_FACTOR)) or {'path': str(obj_path), 'work': -1, 'bond': -1, 'largest_factor': -1}
        locate = sorted({dst_centre + o for o in getattr(self, 'race_b_offsets', RACE_B_OFFSETS)} | {dst_centre})
        r = (getattr(self, '_prereduced', None) or {}).get(str(Path(obj_path).resolve()))
        if r is None:
            r = self.reduce(obj, [x for x in locate if 0 <= x < self.N])
        if r is None:
            return False
        red, meta = r
        mapped = {int(k): v for k, v in (meta.get('located') or {}).items() if isinstance(v, dict) and 'reduced_block' in v}
        if not mapped:
            log('[reduce] no mapped centre for the solve module (all inside the object?)')
            return False
        centres2 = sorted({int(v['reduced_block']) for v in mapped.values()})
        self.solve_from_reduced(red, centres2, attempt)
        return True

    def solve(self, *, object_path: Path | None=None, reduced: Path | None=None, centre: int | None=None) -> str | None:
        self.structure()
        A, B = (self.modules[0], self.modules[1])
        N = self.N
        if reduced is None and object_path is None and (not getattr(self, 'no_beam0', False)) and (not any((getattr(m, 'cz_pairs', 0) for m in self.modules))):
            try:
                if self.forward_beam_stage():
                    return self.best['bits']
            except Exception as exc:
                log(f'[beam0] error: {type(exc).__name__}: {exc}; continuing with the hunt')
        if reduced is not None:
            c = centre if centre is not None else B.centre_block
            self.solve_from_reduced(Path(reduced), sorted({c + o for o in RACE_B_OFFSETS}), 0)
        elif object_path is not None:
            self.solve_from_object(Path(object_path), centre if centre is not None else B.centre_block, 0)
        else:
            first_B = getattr(self, 'object_module', 'A') == 'B'
            plan = [(0, B, A, self.qasm), (1, A, B, self.qasm)] if first_B else [(0, A, B, self.qasm), (1, B, A, self.qasm)]
            if getattr(self, 'module_race', False):
                plan = [(0, None, None, self.qasm), (1, None, None, self.qasm)]
            prep_done = False
            while plan:
                if self.best and self.best['ratio'] >= MARGIN:
                    break
                attempt = None
                try:
                    attempt, src, dst, qasm = plan.pop(0)
                    self.maybe_attempt2()
                    arms = None
                    self._spawner = None
                    if attempt >= 1 and self.left() < LADDER_MIN_LEFT * self.ts:
                        break
                    if src is None and (not getattr(self, 'module_race', False)):
                        src, dst = (A, B) if attempt % 2 == 0 else (B, A)
                    if qasm != self.qasm:
                        self.qasm = qasm
                        self.structure()
                        A, B = (self.modules[0], self.modules[1])
                        N = self.N
                        src, dst = ((B, A) if attempt % 2 == 0 else (A, B)) if first_B else (A, B) if attempt % 2 == 0 else (B, A)
                    if getattr(self, 'module_race', False) and src is None:
                        wins = {A.centre_block: self.gap_window(A, B, N), B.centre_block: self.gap_window(B, A, N)} if getattr(self, 'hunt_window', 'full') == 'mid' else {A.centre_block: (0, N), B.centre_block: (0, N)}
                        if attempt == 1:
                            log('=== attempt 1: the module race already used both centres; nothing new to try on this circuit')
                            continue
                        log(f'=== attempt {attempt}: module race A {A.centre_block} vs B {B.centre_block} ({qasm.name})')
                        c_hunt, leader = self.race(f'raceM{attempt}', self.qasm, self.RECORD, self.RECORD_FLAGS, [A.centre_block, B.centre_block], (0, N), seconds=MODULE_RACE_SECONDS * self.ts, first_elim=MODULE_RACE_SECONDS * self.ts, elim_every=RACE_ELIM_EVERY * self.ts, early_stop=0, waist_maxbond=getattr(self, 'waist_maxbond', 16), tail_stop=False, N=N, windows=wins)
                        src, dst = (A, B) if c_hunt == A.centre_block else (B, A)
                        log(f"module race winner: {('A' if src is A else 'B')} (centre {c_hunt}); solve module centre {dst.centre_block}")
                    else:
                        log(f'=== attempt {attempt}: object module at ~{src.centre_block}, solve module at ~{dst.centre_block} ({qasm.name})')
                        if getattr(self, 'object_centre', None) is not None:
                            src.centre_block = int(self.object_centre)
                        elif getattr(self, 'object_centre_offset', 0):
                            if not hasattr(A, 'rule_centre'):
                                A.rule_centre, B.rule_centre = (A.centre_block, B.centre_block)
                                first = ('A', int(self.object_centre_offset), 1.0)
                                self._ladder = [first] + [t for t in OFFSET_LADDER if t != first]
                            if not self._ladder and (not self.refill_ladder()):
                                log('=== no time left for another hunt')
                                continue
                            mod, off, _jit = self._ladder.pop(0)
                            src, dst = (A, B) if mod == 'A' else (B, A)
                            self._ticket_n = getattr(self, '_ticket_n', 0) + 1
                            self._ticket_cfg = dict(self.RECORD, cutoff=self.RECORD['cutoff'] * _jit, final_cutoff=self.RECORD['final_cutoff'] * _jit)
                            src.centre_block = int(src.rule_centre + off)
                            log(f'object module {mod}: centre = rule {src.rule_centre} {off:+d} -> {src.centre_block} (tickets left {len(self._ladder)})')
                        centres = [src.centre_block + o for o in RACE_A_OFFSETS]
                        hw = {'full': (0, N), 'mid': None, 'anchor': None}.get(getattr(self, 'hunt_window', 'full'), (0, N))
                        if getattr(self, 'hunt_window', 'full') == 'mid':
                            hw = self.gap_window(src, dst, N)
                        elif getattr(self, 'hunt_window', 'full') == 'anchor':
                            hw = self.anchor_window(src, dst, N)
                        if getattr(self, 'hunt_window_abs', None):
                            hw = tuple(self.hunt_window_abs)
                        log(f'hunt window {hw}')
                        c_hunt, leader = self.race(f"raceA{attempt}t{getattr(self, '_ticket_n', 0)}", self.qasm, getattr(self, '_ticket_cfg', None) or self.RECORD, self.RECORD_FLAGS, centres, hw, seconds=RACE_SECONDS * self.ts, first_elim=RACE_FIRST_ELIM * self.ts, elim_every=RACE_ELIM_EVERY * self.ts, early_stop=0, waist_maxbond=getattr(self, 'waist_maxbond', 16), tail_stop=False, N=N)
                        arms = [leader]
                        leader.window = hw
                        leader.dst_centre = getattr(dst, 'rule_centre', dst.centre_block)
                        if PARALLEL_HUNT_ARMS >= 2 and hasattr(src, 'rule_centre'):

                            def spawner(_src=src, _hw=hw, _attempt=attempt):
                                self.maybe_attempt2()
                                if self.left() < LADDER_MIN_LEFT * self.ts or not (getattr(self, '_ladder', None) or self.refill_ladder()):
                                    return None
                                mod2, off2, jit = self._ladder.pop(0)
                                m_src, m_dst = (A, B) if mod2 == 'A' else (B, A)
                                c2 = int(m_src.rule_centre + off2)
                                _hw = self.anchor_window(m_src, m_dst, N) if getattr(self, 'hunt_window', 'full') == 'anchor' else _hw
                                self._arm_n = getattr(self, '_arm_n', 0) + 1
                                cfg = dict(self.RECORD, cutoff=self.RECORD['cutoff'] * jit, final_cutoff=self.RECORD['final_cutoff'] * jit)
                                log(f'portfolio arm {self._arm_n}: module {mod2} rule {m_src.rule_centre} {off2:+d} -> {c2}, cutoff x{jit:g} (tickets left {len(self._ladder)})')
                                r_new = self.launch(f'raceA{_attempt}_c{c2}_arm{self._arm_n}', self.qasm, cfg, self.RECORD_FLAGS, c2, _hw, seconds=RACE_SECONDS * self.ts + 3600, early_stop=0, ckpt_layers=1, waist_maxbond=getattr(self, 'waist_maxbond', 16), tail_stop=False)
                                r_new.dst_centre = getattr(m_dst, 'rule_centre', m_dst.centre_block)
                                return r_new
                            self._spawner = spawner
                    if getattr(self, 'hunt_only', 0):
                        obj = self.hunt_many(arms or [leader], float(self.hunt_only) * self.ts, spawn=getattr(self, '_spawner', None))
                        log(f'[hunt-only] stop after the hunt; best usable object: {obj}')
                        for r in arms or [leader]:
                            r.kill('hunt-only done')
                        break
                    obj = self.hunt_many(arms or [leader], min(3600 * self.ts, self.left() - 1800 * self.ts), spawn=getattr(self, '_spawner', None))
                    self._spawner = None
                    self.discard([r for r in getattr(self, '_runs', []) if not r.alive()])
                    if obj is None:
                        log('no usable object from this module')
                        if leader.alive():
                            leader.kill('no usable object')
                        if (getattr(self, '_ladder', None) or self.refill_ladder()) and (not getattr(self, 'module_race', False)) and (src is not None) and hasattr(src, 'rule_centre') and (self.left() > LADDER_MIN_LEFT * self.ts):
                            blk = leader.stats.get('blocks') or 0
                            if not getattr(leader, 'cap_abort', False) and self.walled_short(blk, getattr(leader, 'window', None)):
                                leader.cap_abort = True
                                log(f'[hunt] {leader.tag}: walled at {blk} blocks of window {leader.window} without an object: treated as a wrong-module abort (v5)')
                            elif getattr(leader, 'cap_abort', False) and self.early_cap(blk, getattr(leader, 'window', None)):
                                leader.cap_abort = False
                                log(f'[hunt] {leader.tag}: capped at {blk} blocks of window {leader.window}: an EARLY cap is a bad seed on the right module, not the wrong module (v5 #5); next offset on the same module')
                            if getattr(leader, 'cap_abort', False):
                                self.next_ticket_after_cap('A' if src is A else 'B')
                            else:
                                self.next_ticket_same_module('A' if src is A else 'B')
                            log(f'=== ticket queue: retry the object module (next ticket {self._ladder[0]})')
                            plan.insert(0, (attempt, src, dst, qasm))
                            continue
                        if not plan and (not prep_done) and (not getattr(self, '_prep_applied', False)) and (self.left() > LADDER_MIN_LEFT * self.ts):
                            prep_done = True
                            red = self.prepare()
                            if red is not None:
                                plan = [(2, None, None, red), (3, None, None, red)]
                        continue
                    cands = [c for c in getattr(self, '_candidates', None) or [] if c.get('path')] or [obj]
                    if len(getattr(self, '_pool', None) or cands) > 1:
                        cands = self.rank_by_reduced(cands, dst)
                    solved = False
                    for ci, cand in enumerate(cands):
                        if ci > 0:
                            if self.left() < CONT_MIN_LEFT * self.ts:
                                break
                            log(f"=== next object candidate {ci + 1}/{len(cands)}: {cand['work']} blocks, bond {cand['bond']}, factor {cand['largest_factor']} (the previous one did not reach the margin)")
                        d_c = cand.get('dst_centre') or getattr(dst, 'rule_centre', dst.centre_block)
                        try:
                            solved = self.solve_from_object(Path(cand['path']), int(d_c), 100 * attempt + 10 * self._n_solves + ci) or solved
                        except Exception as exc:
                            log(f'[solve] object candidate {ci + 1} failed: {type(exc).__name__}: {exc}; next candidate')
                            live = [r for r in getattr(self, '_runs', []) if r.alive()]
                            for r in live:
                                r.kill('candidate error')
                            self.discard(live)
                        if self.best and self.best['ratio'] >= MARGIN:
                            break
                    self._n_solves += 1
                    if self.best and self.best['ratio'] >= MARGIN:
                        break
                    if (getattr(self, '_ladder', None) or self.refill_ladder()) and (not getattr(self, 'module_race', False)) and (src is not None) and hasattr(src, 'rule_centre') and (self.left() > LADDER_MIN_LEFT * self.ts):
                        log(f"=== M1 solve did not reach the margin from a {obj.get('work')}-block object; ticket queue: re-hunt with {self._ladder[0]} (a deeper collapse)")
                        plan.insert(0, (attempt, src, dst, qasm))
                        continue
                    if not solved:
                        continue
                except Exception as exc:
                    import traceback
                    self._stage_errors = getattr(self, '_stage_errors', 0) + 1
                    log(f'=== stage error {self._stage_errors}: {type(exc).__name__}: {exc}; stopping its runs and moving on')
                    traceback.print_exc(file=sys.stdout)
                    live = [r for r in getattr(self, '_runs', []) if r.alive()]
                    for r in live:
                        r.kill('stage error')
                    self.discard(live)
                    if self._stage_errors >= 5 or self.left() < LADDER_MIN_LEFT * self.ts or (self.best and self.best['ratio'] >= MARGIN):
                        break
                    if attempt is not None and src is not None and hasattr(src, 'rule_centre') and (getattr(self, '_ladder', None) or self.refill_ladder()):
                        plan.insert(0, (attempt, src, dst, qasm))
        self.join_readouts()
        try:
            self.twosided_rescan()
        except Exception as exc:
            log(f"[twosided] error: {type(exc).__name__}: {exc}; the beam's pick stands")
        self.info['best'] = None if self.best is None else {k: v for k, v in self.best.items() if k != 'bits'}
        return None if self.best is None else self.best['bits']

def grade(bits: str, manifest: Path, instance: str) -> dict:
    man = json.loads(Path(manifest).read_text())
    truth = set(next((e for e in man['instances'] if e['qasm'] == instance))['truth_sha256'].values())
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    return {'instance': instance, 'match': h(bits) in truth or h(bits[::-1]) in truth}

def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--qasm', required=True)
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--wall', type=float, default=14400.0)
    ap.add_argument('--reserve', type=float, default=600.0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--dtype', default='complex64')
    ap.add_argument('--test-scale', type=float, default=1.0)
    ap.add_argument('--unswap-scale', type=float, default=1.0)
    ap.add_argument('--instance', default='')
    ap.add_argument('--manifest', default=str(HERE / 'truth_manifest.json'))
    ap.add_argument('--bits-out', default='')
    ap.add_argument('--object-module', choices=('A', 'B'), default='A')
    ap.add_argument('--module-race', action='store_true')
    ap.add_argument('--prep-first', action='store_true')
    ap.add_argument('--hunt-window-abs', default='')
    ap.add_argument('--object-centre', type=int, default=None)
    ap.add_argument('--hunt-window', choices=('full', 'mid', 'anchor'), default='full')
    ap.add_argument('--hunt-cutoff', type=float, default=None)
    ap.add_argument('--hunt-gram', action='store_true')
    ap.add_argument('--hunt-unswap', type=float, default=None)
    ap.add_argument('--object-centre-offset', type=int, default=0)
    ap.add_argument('--max-factor', type=int, default=MAX_FACTOR)
    ap.add_argument('--waist-maxbond', type=int, default=16)
    ap.add_argument('--no-beam0', action='store_true')
    ap.add_argument('--hunt-only', type=float, default=0.0)
    ap.add_argument('--object', default='')
    ap.add_argument('--reduced', default='')
    ap.add_argument('--centre', type=int, default=None)
    a = ap.parse_args()
    s = Solver(Path(a.qasm), Path(a.workdir), wall=a.wall, reserve=a.reserve, device=a.device, dtype=a.dtype, test_scale=a.test_scale, unswap_scale=a.unswap_scale)
    s.hunt_window = a.hunt_window
    s.hunt_window_abs = tuple((int(x) for x in a.hunt_window_abs.split(','))) if a.hunt_window_abs else None
    s.object_centre = a.object_centre
    s.waist_maxbond = a.waist_maxbond
    s.max_factor = a.max_factor
    s.object_centre_offset = a.object_centre_offset
    s.hunt_only = a.hunt_only
    s.no_beam0 = a.no_beam0
    s.object_module = a.object_module
    s.module_race = a.module_race
    if a.prep_first:
        red = s.prepare()
        if red is not None:
            s.qasm = red
    if a.hunt_cutoff is not None:
        s.RECORD = dict(s.RECORD, cutoff=a.hunt_cutoff, final_cutoff=a.hunt_cutoff)
    if a.hunt_unswap is not None:
        s.RECORD = dict(s.RECORD, unswap_threshold=float(a.hunt_unswap))
        fl = list(s.RECORD_FLAGS)
        i = fl.index('--adaptive-unswap')
        g, f, _ = fl[i + 1].split(',')
        fl[i + 1] = f'{g},{f},{int(a.hunt_unswap)}'
        s.RECORD_FLAGS = fl
    if a.hunt_gram:
        fl = list(s.RECORD_FLAGS)
        i = fl.index('--gram-min-dim')
        del fl[i:i + 2]
        s.RECORD_FLAGS = fl
    log(f"hunt config: window {s.hunt_window}, cutoff {s.RECORD['cutoff']}, unswap {s.RECORD['unswap_threshold']:.0f}, flags {' '.join(s.RECORD_FLAGS)}")
    bits = None
    try:
        bits = s.solve(object_path=Path(a.object) if a.object else None, reduced=Path(a.reduced) if a.reduced else None, centre=a.centre)
    except Exception as exc:
        import traceback
        log(f'solver error: {type(exc).__name__}: {exc}')
        traceback.print_exc()
        if s.best:
            bits = s.best['bits']
    (s.work / 'solve_info.json').write_text(json.dumps(s.info, indent=1, default=str))
    if bits and a.bits_out:
        Path(a.bits_out).write_text(bits + '\n')
    if bits and a.instance:
        g = grade(bits, Path(a.manifest), a.instance)
        log(f"GRADE {a.instance}: {('MATCH' if g['match'] else 'no match')} (best source {s.best['source']}, top1/top2 {s.best['ratio']:.2f})")
    elif bits:
        log(f"answer ready (source {s.best['source']}, top1/top2 {s.best['ratio']:.2f})")
    else:
        log('no answer')
    return 0 if bits else 1
if __name__ == '__main__':
    raise SystemExit(main())
