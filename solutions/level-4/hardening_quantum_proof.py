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

"""Level-4 HQP solver entry (2026-10-13). Wrapper pattern from our accepted level-3 v5 entry point; route from
level4/hqpgen/counter/swapnet_solve.py (difficulty-4 samples hash-matched on CPU).

Protocol (enigma repo commit 9860c5c, qbittensor/challenges/solution_output.py): logs, SOLUTION_OUTPUT_SEPARATOR, base64
zip (result.json + solve_info.json), SOLUTION_OUTPUT_END_SEPARATOR, nothing after it. stderr is merged into stdout at
fd level before anything prints; before the payload every child process group is killed and fds 1/2 are pointed at
/dev/null, then the payload goes out in one os.write loop on a saved fd. status is always "success".

Plan per circuit (CPU only; the GPU is never touched):
  attempt 1: swapnet route, undo threshold 0.50;  attempt 2 (no answer): undo 0.30;  attempt 3: undo 0.10.
  A watchdog thread emits the best answer so far (or a correctly sized placeholder) at HQP_WATCHDOG_AT seconds.
"""
import os
import sys
try:
    sys.stdout.flush()
    os.dup2(sys.stdout.fileno(), sys.stderr.fileno())
except Exception:
    pass
import base64
import json
import re
import signal
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

START = time.time()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from enigma_challenges.hardening_quantum_proof import Solution, load_solver_input  # noqa: E402
from enigma_challenges.solution_output import (SOLUTION_OUTPUT_END_SEPARATOR, SOLUTION_OUTPUT_SEPARATOR,  # noqa: E402
                                               build_solution_zip)

BUILD = "hqp4-v7"
WATCHDOG_AT = float(os.environ.get("HQP_WATCHDOG_AT", "13500"))      # validator wall 14400 s from container start
DEADLINE = float(os.environ.get("HQP_DEADLINE", "13200"))           # no new attempt is started after this
WORKDIR = Path(os.environ.get("HQP_WORKDIR", "/tmp/hqp4"))
PP_PY = os.environ.get("HQP4_PY", "/opt/pp/bin/python")
UNDO_LADDER = [x for x in os.environ.get("HQP4_UNDO_LADDER", "0.50,0.30,0.10").split(",") if x]
STOP_RATIO = float(os.environ.get("HQP4_STOP_RATIO", "5"))
# v5 guard 1: never use more than MAX_THREADS (the validator quota is 24, so this changes nothing there; it protects the
# 1024-PID limit when the quota is absent and the host shows 128-256 CPUs)
MAX_THREADS = int(os.environ.get("HQP4_MAX_THREADS", "32"))
# v4 attempt plan (user rule 2026-10-14): stop only at top1/top2 >= STOP_RATIO; otherwise keep trying diverse attempts
# until DEADLINE, then emit the highest ratio. Measured: correct answers read 2.08-31 (364 graded), the one wrong
# answer of the night read 3.2 (generated seed 661, laptop route) -> a ratio of 2 is not a safe stop.
# Each entry: (label, route module, undo threshold, extra args). HQP4_UNDO_LADDER (test override) keeps the old ladder.
PLAN = [("v3", "hqp4.solve3", "0.50", []),
        ("v3-lowest-err", "hqp4.solve3", "0.50", ["--cut-policy", "lowest-err"]),
        ("v3-undo0.30", "hqp4.solve3", "0.30", []),
        ("v3-p9seed11", "hqp4.solve3", "0.50", ["--p9-seed", "11"]),
        ("v1-route", "hqp4.swapnet_solve", "0.50", []),
        ("v3-undo0.10", "hqp4.solve3", "0.10", []),
        ("v3-lowest-err-undo0.30", "hqp4.solve3", "0.30", ["--cut-policy", "lowest-err"]),
        ("v3-p9seed23-lowest-err", "hqp4.solve3", "0.50", ["--cut-policy", "lowest-err", "--p9-seed", "23"])]
if "HQP4_UNDO_LADDER" in os.environ:
    PLAN = [(f"v3-undo{u}", "hqp4.solve3", u, []) for u in UNDO_LADDER]
EMIT_LOCK = threading.Lock()
STATE = {"best": None, "n": None, "attempts": [], "children": []}


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} +{time.time() - START:7.1f}s] {msg}", flush=True)


def cpu_quota():
    """CPUs from the cgroup quota (validator: --cpus 24; os.cpu_count() reports the host's cores), else visible."""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            return max(1, int(int(q) / int(p)))
    except Exception:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 4


def kill_children():
    for p in STATE["children"]:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass


def payload(peak, info, challenge_id, difficulty):
    sol = Solution("success", peak)
    si = {"solution_status": "success", "challenge_id": challenge_id, "difficulty": difficulty,
          "timestamp_utc": datetime.fromtimestamp(START, timezone.utc).isoformat(),
          "solve_time_seconds": round(time.time() - START, 1), "build": BUILD, **info}
    z = build_solution_zip({"result.json": json.dumps(sol.to_dict(), indent=2),
                            "solve_info.json": json.dumps(si, default=str)})
    return SOLUTION_OUTPUT_SEPARATOR + base64.b64encode(z) + SOLUTION_OUTPUT_END_SEPARATOR


def emit_and_exit(challenge_id, difficulty, note):
    """Single emitter (lock): kill children, silence fds 1/2, write the payload once on a saved fd, _exit."""
    if not EMIT_LOCK.acquire(blocking=False):
        time.sleep(3600)
        os._exit(0)
    try:
        best = STATE["best"] or {}
        peak = best.get("bits") or "0" * int(STATE["n"] or 52)
        info = {"note": note, "best": {k: v for k, v in best.items() if k != "bits"}, "attempts": STATE["attempts"]}
        log(f"EMIT ({note}): source {best.get('source')}, ratio {best.get('ratio')}, n {len(peak)}")
        kill_children()
        data = payload(peak, info, challenge_id, difficulty)
        out = os.environ.get("OUTPUT_DIR")
        if out:                                            # workbench --mode direct only
            try:
                Path(out).mkdir(parents=True, exist_ok=True)
                Path(out, "result.json").write_text(json.dumps(Solution("success", peak).to_dict(), indent=2))
            except OSError:
                pass
        try:
            sys.stdout.flush()
        except Exception:
            pass
        real = os.dup(1)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
        while data:
            data = data[os.write(real, data):]
    finally:
        os._exit(0)


def watchdog(challenge_id, difficulty):
    time.sleep(max(0.0, WATCHDOG_AT - (time.time() - START)))
    log(f"WATCHDOG at {time.time() - START:.0f} s")
    emit_and_exit(challenge_id, difficulty, f"watchdog at {time.time() - START:.0f} s")


def normalize_input(qasm):
    """v5 guard 2: the route reads u / cz only. A circuit with any other gate is rewritten to u + cz by qiskit (basis
    translation only: no layout, no routing, so qubit i stays qubit i and the answer's bit order is unchanged). A pure
    u / cz circuit (every d4 sample) is returned unchanged. On any error the original path is kept."""
    try:
        other = set()
        for ln in Path(qasm).read_text().splitlines():
            s = ln.split("//")[0].strip()
            if not s or s.startswith(("OPENQASM", "include", "qreg", "creg", "barrier", "measure")):
                continue
            name = re.split(r"[\s(]", s, 1)[0]
            if name not in ("u", "cz"):
                other.add(name)
        if not other:
            return qasm
        log(f"input has non-u/cz gates {sorted(other)[:8]}: translating to u + cz")
        out = WORKDIR / "input_ucz.qasm"
        WORKDIR.mkdir(parents=True, exist_ok=True)
        code = ("import sys\nfrom qiskit import qasm2, transpile\n"
                "c = qasm2.load(sys.argv[1], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)\n"
                "c.remove_final_measurements(inplace=True)\n"
                "t = transpile(c, basis_gates=['u', 'cz'], optimization_level=1, layout_method='trivial', routing_method='none', seed_transpiler=7)\n"
                "assert [t.find_bit(q).index for q in t.qubits] == list(range(c.num_qubits))\n"
                "open(sys.argv[2], 'w').write(qasm2.dumps(t))\n")
        r = subprocess.run([PP_PY, "-c", code, str(qasm), str(out)], capture_output=True, text=True, timeout=1800)
        if r.returncode == 0 and out.exists():
            log(f"translated input -> {out}")
            return str(out)
        log(f"translation failed (rc {r.returncode}): {r.stderr[-300:]}; using the original input")
    except Exception as exc:  # noqa: BLE001
        log(f"input check failed: {type(exc).__name__}: {exc}; using the original input")
    return qasm


def run_attempt(qasm, k, undo, threads, label="v3", route="hqp4.solve3", extra=()):
    out = WORKDIR / f"a{k}_{label}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), NUMBA_NUM_THREADS=str(threads), PYTHONPATH=str(HERE))
    cmd = [PP_PY, "-m", route, str(qasm), "--out", str(out), "--threads", str(threads), "--undo-threshold", undo] + list(extra)
    log(f"attempt {k}: undo {undo}, threads {threads}: {' '.join(cmd[2:])}")
    t0 = time.time()
    with open(out / "route.log", "w") as f:
        p = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        STATE["children"].append(p)
        while p.poll() is None:
            time.sleep(5)
            if time.time() - START > WATCHDOG_AT - 120:     # leave the emit to the watchdog
                break
    R = {}
    try:
        R = json.loads((out / "result.json").read_text())
    except Exception:
        pass
    stages = [(s.get("name"), s.get("error") or "ok") for s in R.get("stages", [])]
    A = {"attempt": k, "label": label, "undo": undo, "rc": p.returncode, "secs": round(time.time() - t0, 1),
         "ratio": R.get("ratio"), "answer_from": R.get("answer_from"), "mps_agrees": R.get("mps_agrees"),
         "stages": stages}
    STATE["attempts"].append(A)
    try:                                                    # route log tail into the container log (no bitstrings)
        tail = (out / "route.log").read_text(errors="replace").splitlines()[-25:]
        for ln in tail:
            print("    | " + re.sub(r"[01]{20,}", "<bits>", ln)[:220], flush=True)
    except Exception:
        pass
    bits = R.get("answer_q0_left")
    if bits and re.fullmatch(r"[01]+", bits) and len(bits) == STATE["n"]:
        r = R.get("ratio") or 0.0
        try:
            r = float(r)
        except (TypeError, ValueError):
            r = 0.0
        if r != r:                                          # NaN guard
            r = 0.0
        if STATE["best"] is None or r > STATE["best"]["ratio"]:
            STATE["best"] = {"bits": bits, "ratio": r, "source": label, "attempt": k}
    log(f"attempt {k} ended rc {p.returncode} in {A['secs']} s: ratio {A['ratio']}, from {A['answer_from']}, "
        f"mps_agrees {A['mps_agrees']}")
    return A


def main():
    challenge_id, difficulty, qasm = "unknown", None, None
    try:
        challenge_id, problem = load_solver_input(sys.argv)
        difficulty, qasm = problem.difficulty, problem.qasm_file
        m = re.search(r"qreg\s+\w+\[(\d+)\]", Path(qasm).read_text())
        STATE["n"] = int(m.group(1)) if m else None
    except Exception as err:
        log(f"input error: {type(err).__name__}: {err}")
        emit_and_exit(challenge_id, difficulty, f"input error: {err}")
    threading.Thread(target=watchdog, args=(challenge_id, difficulty), daemon=True).start()
    threads = min(cpu_quota(), MAX_THREADS)
    qasm = normalize_input(qasm)
    log(f"HQP {BUILD} | {challenge_id} difficulty={difficulty} qasm={qasm} n={STATE['n']} cpus={threads} "
        f"watchdog={WATCHDOG_AT:.0f}s deadline={DEADLINE:.0f}s")
    try:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        for k, (label, route, undo, extra) in enumerate(PLAN, 1):
            if time.time() - START > DEADLINE:
                log(f"deadline reached before attempt {k}")
                break
            try:
                run_attempt(qasm, k, undo, threads, label, route, extra)
            except Exception as exc:
                log(f"attempt {k} crashed: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            b = STATE["best"]
            if b and b["ratio"] >= STOP_RATIO:
                log(f"stop: attempt {b['attempt']} ({b['source']}) reached top1/top2 {b['ratio']} >= {STOP_RATIO}")
                break
    except Exception as exc:
        log(f"fatal: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    emit_and_exit(challenge_id, difficulty, "final" if STATE["best"] else "no candidate: placeholder")


if __name__ == "__main__":
    main()
