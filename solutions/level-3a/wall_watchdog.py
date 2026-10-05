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

"""Hard wall-clock backstop for the HQP solver.

WHY THIS EXISTS (measured 2026-09-15, Level-3 submission): the solver's wall guard is
COOPERATIVE -- every stage has to volunteer a `time_left()` check. A single long operation
(one chi rung of `mps_torch.evolve`, one MPO/MPS contraction, one SVD fallback cascade) therefore
sails straight past the budget. The validator then kills the container at
`max_solution_runtime_seconds` (14400 s for L3) and, because the ONLY output channel is the
base64 payload on stdout, `docker logs` carries NO payload at all -> WallTimeFailure, a guaranteed
zero. That is exactly what happened: the container ran the full 4 h and emitted nothing, while the
same code solves the two public samples in 22-38 minutes.

The contract with the validator makes partial/progressive output impossible: it splits stdout on
the FIRST occurrence of the separator and treats everything after it as the payload
(`run_solution._split_on_separator`), so a provisional payload followed by more log lines would
corrupt the real one. The payload must be emitted exactly ONCE and be the last thing printed.

So the backstop is: stages PUBLISH their best candidate as soon as they have one, and a daemon
thread emits that candidate and hard-exits before the validator's kill, whatever the main thread
is doing. Torch/quimb release the GIL inside their heavy kernels, so the thread is scheduled even
while a CUDA op is in flight.

Env (all defaulted and settable):
  HQP_WATCHDOG=1         master switch
  HQP_HARD_WALL=14400    the validator's max_solution_runtime_seconds for this milestone
  HQP_HARD_MARGIN=600    emit this many seconds BEFORE that kill
"""
import os
import sys
import threading
import time

ON = os.environ.get("HQP_WATCHDOG", "1").strip() not in ("0", "false", "False")
HARD_WALL = float(os.environ.get("HQP_HARD_WALL", os.environ.get("WALL_TIME", "14400")))
HARD_MARGIN = float(os.environ.get("HQP_HARD_MARGIN", "600"))

# Serialises log writes against the final emit: the payload must not have a log line interleaved
# into it. Every print in the solver goes through this, and the watchdog holds it while emitting.
emit_lock = threading.RLock()

_best = {"bits": None, "info": {}, "trusted": False, "score": float("-inf"), "stage": None,
         "key": (False, -1, float("-inf"))}
_best_lock = threading.Lock()
_emitted = threading.Event()
_silenced = threading.Event()


# A process start this far in the past is taken as real; anything outside [0, this] means the
# /proc arithmetic disagreed with time.time() and must not be trusted.
_MAX_PLAUSIBLE_AGE = float(os.environ.get("HQP_PROC_AGE_MAX", "3600"))


def _proc_start():
    """Wall-clock time this PROCESS started, from /proc/self/stat, or None if implausible.

    Anchoring on the process (not on module-import time) closes the blind spot between the
    container's StartedAt -- which is what the validator actually measures -- and the moment our
    module-level START is assigned, i.e. interpreter boot plus the numpy/qiskit imports.

    /proc/uptime is measured against the kernel's monotonic boot clock while time.time() is the
    (settable) wall clock, so any clock step -- an NTP correction, a container with a shifted
    clock -- makes this difference meaningless. MEASURED: in our own build container it came out
    62 DAYS in the future, which would have pushed the deadline past the heat death of the run and
    silently disabled the very backstop this module exists to provide. So the result is
    range-checked and a bad one is discarded LOUDLY rather than quietly used.
    """
    try:
        with open("/proc/self/stat", "rb") as f:
            fields = f.read().rsplit(b")", 1)[1].split()
        ticks = int(fields[19])                      # starttime, in clock ticks since boot
        hz = os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        est = time.time() - (uptime - ticks / hz)
    except Exception as e:
        return None, f"/proc unreadable ({type(e).__name__})"
    age = time.time() - est
    if not (0.0 <= age <= _MAX_PLAUSIBLE_AGE):
        return None, (f"/proc/uptime disagrees with the wall clock (implied process age "
                      f"{age:.0f}s, outside [0,{_MAX_PLAUSIBLE_AGE:.0f}]) -- clock skew")
    return est, "proc"


_ps, PROC_START_SOURCE = _proc_start()
# Falling back to import time is safe: it is LATER than the true start, so the deadline it yields
# is EARLIER, i.e. conservative against the validator's kill.
PROC_START = _ps if _ps is not None else time.time()
DEADLINE = PROC_START + HARD_WALL - HARD_MARGIN


def time_to_deadline():
    return DEADLINE - time.time()


def silenced():
    """True once the watchdog has begun emitting: callers must stop writing to stdout."""
    return _silenced.is_set()


def publish(bits, info=None, trusted=False, score=0.0, stage=None, prio=0):
    """Record a candidate answer. Safe to call from any stage, as often as it likes.

    Ranking is (trusted, prio, score), ties to the newer candidate. A trusted candidate is sticky.
    `prio` orders the STAGES, because their margins are not on a common scale: a 1.2 from the
    canonical ladder on a raw d3 circuit is a measured noise argmax (E3/E52/E80), while a 1.2 from
    the excision ladder is a real peak. Mirrors solve()'s own final preference: excision (2) over
    unswap (1) over canonical (0).
    """
    if not bits:
        return False
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    if score != score:                               # NaN margins are not a score
        score = 0.0
    key = (bool(trusted), int(prio), score)
    with _best_lock:
        if _best["bits"] is not None and key < _best["key"]:
            return False
        _best.update(bits=bits, info=dict(info or {}), trusted=bool(trusted),
                     score=score, stage=stage, key=key)
        return True


def best():
    with _best_lock:
        return _best["bits"], dict(_best["info"]), _best["trusted"], _best["stage"]


_claim_lock = threading.Lock()
_claimed = [False]


def claim_emit():
    """Atomically win the right to emit. Exactly one caller ever gets True.

    Event.set() is not a test-and-set, so the normal path and the watchdog could both decide to
    emit and print TWO separators -- and the validator splits on the FIRST one, which would leave
    log text and a second payload inside the region it decodes. One winner, always.
    """
    with _claim_lock:
        if _claimed[0]:
            return False
        _claimed[0] = True
        _emitted.set()
        return True


def mark_emitted():
    """Called by the normal exit path so the watchdog never double-emits."""
    _emitted.set()


def arm(emit_fn, log=None):
    """Start the backstop. `emit_fn(bits, info, reason)` must print the payload and not return."""
    if not ON:
        if log:
            log("watchdog: DISABLED (HQP_WATCHDOG=0) -- a stuck stage will produce no output")
        return None

    def _run():
        while True:
            dt = time_to_deadline()
            if dt <= 0:
                break
            time.sleep(min(dt, 30.0))                # short naps: a wedged main thread can't delay us
            if _emitted.is_set():
                return
        if _emitted.is_set():
            return
        # Take the log lock FIRST so no stage can print into the middle of the payload, and latch
        # the silence flag so anything that bypasses the lock stops writing too.
        _silenced.set()
        with emit_lock:
            if _emitted.is_set():
                return                               # the normal path won the race and is exiting
            # NB: do NOT claim here -- emit_fn claims. Claiming twice would make emit_fn lose to
            # us and silently print nothing, which is the exact failure this module prevents.
            bits, info, trusted, stage = best()
            info = dict(info)
            info["watchdog_fired"] = True
            info["watchdog_stage"] = stage
            info["watchdog_elapsed"] = round(time.time() - PROC_START, 1)
            try:
                emit_fn(bits, info, reason=f"wall watchdog at {info['watchdog_elapsed']:.0f}s "
                                           f"(hard wall {HARD_WALL:.0f}s, margin {HARD_MARGIN:.0f}s)")
            except BaseException:                    # never let the backstop itself hang the exit
                try:
                    sys.stdout.flush()
                except Exception:
                    pass
                os._exit(1)

    t = threading.Thread(target=_run, name="hqp-wall-watchdog", daemon=True)
    t.start()
    if log:
        if PROC_START_SOURCE != "proc":
            log(f"watchdog: anchoring on import time -- {PROC_START_SOURCE}")
        log(f"watchdog armed: hard emit at {HARD_WALL - HARD_MARGIN:.0f}s after process start "
            f"({time_to_deadline():.0f}s from now); validator kills at {HARD_WALL:.0f}s")
    return t
