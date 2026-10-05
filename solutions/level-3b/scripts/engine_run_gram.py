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
import os
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
argv = sys.argv[1:]
FAST_TAIL_K = 0
if '--fast-tail-drain' in argv:
    _i = argv.index('--fast-tail-drain')
    FAST_TAIL_K = int(argv[_i + 1])
    del argv[_i:_i + 2]
FREEZE_SIDE = None
if '--freeze-side' in argv:
    _i = argv.index('--freeze-side')
    FREEZE_SIDE = argv[_i + 1]
    del argv[_i:_i + 2]
    if FREEZE_SIDE not in ('left', 'right'):
        raise SystemExit('--freeze-side takes left or right')
TAIL_CAP = 0
if '--tail-starve-cap' in argv:
    _i = argv.index('--tail-starve-cap')
    TAIL_CAP = int(argv[_i + 1])
    del argv[_i:_i + 2]
    FAST_TAIL_K = FAST_TAIL_K or 32
STARVE_CAP_N = 0
if '--starve-cap' in argv:
    _i = argv.index('--starve-cap')
    STARVE_CAP_N = int(argv[_i + 1])
    del argv[_i:_i + 2]
    if STARVE_CAP_N < 1:
        raise SystemExit('--starve-cap needs N >= 1')
BOND_ROUTE = None
if '--bond-route' in argv:
    _i = argv.index('--bond-route')
    BOND_ROUTE = tuple((int(v) for v in argv[_i + 1].split(',')))
    del argv[_i:_i + 2]
STOP_AT_BLOCK = 0
ADAPT = None
if '--adaptive-unswap' in argv:
    _i = argv.index('--adaptive-unswap')
    ADAPT = tuple((float(v) for v in argv[_i + 1].split(',')))
    del argv[_i:_i + 2]
    if len(ADAPT) != 3 or ADAPT[0] <= 1.0 or (not 0.0 < ADAPT[1] <= 1.0) or (ADAPT[2] < 0):
        raise SystemExit('--adaptive-unswap needs G,F,N with G > 1, 0 < F <= 1, N >= 0 (e.g. 4,1.0,4000000)')
if '--stop-at-block' in argv:
    _i = argv.index('--stop-at-block')
    STOP_AT_BLOCK = int(argv[_i + 1])
    del argv[_i:_i + 2]
TAIL_STOP = None
if '--tail-stop' in argv:
    _i = argv.index('--tail-stop')
    TAIL_STOP = tuple((int(v) for v in argv[_i + 1].split(',')))
    del argv[_i:_i + 2]
    if len(TAIL_STOP) != 2 or min(TAIL_STOP) <= 0:
        raise SystemExit('--tail-stop wants B,R (max bond, remaining work ops), both positive')
OFFLOAD_ELEMS = 0
RECORD_FRAMES = '--record-frames' in argv
if RECORD_FRAMES:
    argv.remove('--record-frames')
if '--absorb-offload-elems' in argv:
    _i = argv.index('--absorb-offload-elems')
    OFFLOAD_ELEMS = int(argv[_i + 1])
    del argv[_i:_i + 2]
SIDE_LOOK = 0
if '--side-lookahead' in argv:
    _i = argv.index('--side-lookahead')
    SIDE_LOOK = int(argv[_i + 1])
    del argv[_i:_i + 2]
LOG_SPECTRUM = None
if '--log-spectrum' in argv:
    _i = argv.index('--log-spectrum')
    _parts = argv[_i + 1].split(',')
    del argv[_i:_i + 2]
    LOG_SPECTRUM = (_parts[0], int(_parts[1]) if len(_parts) > 1 else 64, int(_parts[2]) if len(_parts) > 2 else 1)
STEER_BURN = None
if '--steer-after-burn' in argv:
    _i = argv.index('--steer-after-burn')
    _r, _w = argv[_i + 1].split(',')
    STEER_BURN = (float(_r), int(_w))
    del argv[_i:_i + 2]
SIDE_OBJ = None
if '--side-objective' in argv:
    _i = argv.index('--side-objective')
    SIDE_OBJ = argv[_i + 1]
    del argv[_i:_i + 2]
    if SIDE_OBJ not in ('peak', 'flat', 'floor', 'size'):
        raise SystemExit('--side-objective takes peak (max bond), flat (max/median), floor (median bond) or size (sum log2 bond)')
APPLY_ZIPUP = False
if '--apply-zipup' in argv:
    _i = argv.index('--apply-zipup')
    APPLY_ZIPUP = True
    del argv[_i:_i + 1]
LOG_BOND = False
if '--log-bond-profile' in argv:
    _i = argv.index('--log-bond-profile')
    LOG_BOND = True
    del argv[_i:_i + 1]
RESUME_SEED = None
if '--resume-seed' in argv:
    _i = argv.index('--resume-seed')
    RESUME_SEED = int(argv[_i + 1])
    del argv[_i:_i + 2]
ROUTE_HORIZON = 0
if '--route-horizon' in argv:
    _i = argv.index('--route-horizon')
    ROUTE_HORIZON = int(argv[_i + 1])
    del argv[_i:_i + 2]
PIN_EDGE = None
if '--pin-edge' in argv:
    _i = argv.index('--pin-edge')
    PIN_EDGE = argv[_i + 1]
    del argv[_i:_i + 2]
    if PIN_EDGE not in ('left', 'right'):
        raise SystemExit('--pin-edge takes left or right')
LOCAL_BUDGET = None
GRAM_MIN_DIM = 32
STOP_ON_STARVATION = False
while argv[:1] in (['--local-budget'], ['--gram-min-dim'], ['--stop-on-starvation']):
    if argv[0] == '--stop-on-starvation':
        STOP_ON_STARVATION = True
        argv = argv[1:]
        continue
    if argv[0] == '--local-budget':
        LOCAL_BUDGET = int(argv[1])
    else:
        GRAM_MIN_DIM = int(argv[1])
    argv = argv[2:]
mode = argv[0] if argv else 'run'
argv = argv[1:]
GENERATOR_SHA256 = 'cc77d7289da819f1b7b3d51195b319d2c16665afd47fb06f35ec32d65ba02d8b'
STARVE_ANCHOR = '                    "drain_budget": starve_drain,\n                })\n'
STARVE_PATCH = STARVE_ANCHOR + '                if globals().get("_STOP_ON_STARVATION"):\n                    raise StarvationStop(state, starve_side, left_work, right_work)\n'
TAILSTOP_PATCH = STARVE_ANCHOR + '                if globals().get("_TAIL_STOP"):\n                    _tb, _tr = _TAIL_STOP\n                    _rem = counters.work_ops_total - counters.work_ops_absorbed_total\n                    _mb = int(get_tn_info(state.mpo)["max_bond"])\n                    if _rem <= _tr and _mb <= _tb:\n                        counters.termination = "early_stopped"\n                        event({"event": "early_stop", "reason": "tail_stop", "work_ops_remaining": _rem,\n                               "left_work_remaining": left_work, "right_work_remaining": right_work,\n                               "max_bond": _mb, "tail_stop": [_tb, _tr]})\n                        return state\n                    event({"event": "tail_stop_deferred", "work_ops_remaining": _rem, "max_bond": _mb, "tail_stop": [_tb, _tr]})\n'
STARVE_HELPER = '\n\nclass StarvationStop(Exception):\n    """Raised where the starvation guard would start draining (engine_run_gram --stop-on-starvation)."""\n\n    def __init__(self, state, side, left_work, right_work):\n        super().__init__(f"starvation guard would drain the {side} front ({left_work} left / {right_work} right work ops pending)")\n        self.state, self.side, self.left_work, self.right_work = state, side, left_work, right_work\n'
FAST_TAIL_ANCHOR = '        if (starve_drain == 0 and work_remaining\n                and consec_zero_work >= STARVE_CAP):\n'
FAST_TAIL_PATCH = '        if (starve_drain == 0 and work_remaining\n                and (consec_zero_work >= STARVE_CAP or _fast_tail_ready(state, consec_zero_work))):\n'
FAST_TAIL_HELPER = '\n\ndef _fast_tail_ready(state, consec_zero_work):\n    k = int(globals().get("_FAST_TAIL_K") or 0)\n    cap = int(globals().get("_TAIL_CAP") or 0)   # --tail-starve-cap: trip after cap zero-work absorbs, never sticky\n    if k <= 0:\n        return False\n    if cap > 0:\n        if consec_zero_work < cap:\n            return False\n    elif not globals().get("_FAST_TAIL_ON") and consec_zero_work < 2:\n        return False\n    def capped(layers, start):   # work ops left on one side, counted only up to k + 1 (the check runs on most zero-work absorbs)\n        n = 0\n        for i in range(start, len(layers)):\n            n += _count_work_ops(layers[i])[0]\n            if n > k:\n                break\n        return n\n    lw = capped(state.layers_left, state.ii_left)\n    rw = capped(state.layers_right, state.ii_right)\n    on = min(lw, rw) == 0 and 0 < max(lw, rw) <= k\n    if on and not globals().get("_FAST_TAIL_ON"):\n        globals()["_FAST_TAIL_TRIGGERS"] = int(globals().get("_FAST_TAIL_TRIGGERS") or 0) + 1\n    globals()["_FAST_TAIL_ON"] = on and cap <= 0\n    return on\n'
OFFLOAD_ANCHOR_A = '            counts_left = elem_counts(mpo_left)\n'
OFFLOAD_PATCH_A = '            counts_left = elem_counts(mpo_left)\n            if _offload_on(mpo_left, state.mpo, counts_left):\n                mpo_left = _mpo_to(mpo_left, "cpu", keep=state.mpo)\n'
OFFLOAD_ANCHOR_B = '            if do_left:\n                state.mpo = mpo_left\n'
OFFLOAD_PATCH_B = '            if do_left:\n                if globals().get("_OFFLOAD_ELEMS"):\n                    mpo_right = None\n                    mpo_left = _mpo_to(mpo_left, _mpo_device(state.mpo), keep=state.mpo)\n                state.mpo = mpo_left\n'
OFFLOAD_HELPER = '\n\ndef _mpo_device(mpo):\n    for t in mpo.tensors:\n        dev = getattr(t.data, "device", None)\n        if dev is not None:\n            return dev\n    return None\n\n\ndef _mpo_to(mpo, device, keep=None):\n    if mpo is None or device is None:\n        return mpo\n    shared = {id(t) for t in keep.tensors} if keep is not None else set()\n    for t in mpo.tensors:\n        if id(t) in shared or not hasattr(t.data, "to"):\n            continue\n        if str(getattr(t.data, "device", "")) != str(device):\n            t.modify(data=t.data.to(device))\n    return mpo\n\n\ndef _offload_on(candidate, current, counts):\n    n = int(globals().get("_OFFLOAD_ELEMS") or 0)\n    return n > 0 and candidate is not None and candidate is not current and counts > n and _mpo_device(candidate) is not None\n'
ADAPT_ANCHOR_A = '            or chosen_counts < unswap_threshold\n'
ADAPT_PATCH_A = '            or (chosen_counts < unswap_threshold and not (unswap_threshold < 1e8 and _adaptive_unswap_now(chosen_counts, mpo_left if do_left else mpo_right, config)))\n'
ADAPT_ANCHOR_B = '            state.force_absorb = cycle.swaps_applied == 0\n'
ADAPT_PATCH_B = '            _adaptive_note_cycle(elem_counts(state.mpo))\n            state.force_absorb = cycle.swaps_applied == 0\n'
ADAPT_HELPER = '\n\ndef _adaptive_unswap_now(counts, candidate, config):\n    growth = float(globals().get("_ADAPT_GROWTH") or 0.0)\n    if growth <= 1.0 or candidate is None or counts < float(globals().get("_ADAPT_FLOOR") or 0.0):\n        return False\n    frac = float(globals().get("_ADAPT_BOND_FRAC") or 0.0)\n    if frac > 0.0 and candidate.max_bond() >= frac * float(config["max_bond"]):\n        return True\n    base = globals().get("_ADAPT_BASE")\n    return base is not None and counts >= growth * float(base)\n\n\ndef _adaptive_note_cycle(counts):\n    globals()["_ADAPT_BASE"] = float(counts)\n'
FREEZE_ANCHOR = '        chosen_counts = counts_left if do_left else counts_right'
FREEZE_PATCH = '        _fz = globals().get("_FREEZE_SIDE")\n        if _fz is not None:\n            _want_left = _fz == "right"\n            if _want_left and state.ii_left < len(state.layers_left):\n                do_left = True\n            elif (not _want_left) and state.ii_right < len(state.layers_right):\n                do_left = False\n        chosen_counts = counts_left if do_left else counts_right'
BLOCKSTOP_ANCHOR = '                    **get_tn_info(state.mpo),\n                }\n            )\n        else:\n            counters.unswap_cycles += 1\n'
STARVE_CAP_ANCHOR = '    STARVE_CAP = 24'
BLOCKSTOP_PATCH = '                    **get_tn_info(state.mpo),\n                }\n            )\n            if globals().get("_STOP_AT_BLOCK") and counters.work_ops_absorbed_total >= _STOP_AT_BLOCK:\n                raise BlockStop(state, counters.work_ops_absorbed_total)\n        else:\n            counters.unswap_cycles += 1\n'
BLOCKSTOP_HELPER = '\n\nclass BlockStop(Exception):\n    """Raised once the absorbed work-op count reaches engine_run_gram --stop-at-block."""\n\n    def __init__(self, state, block):\n        super().__init__(f"absorbed work ops reached {block}")\n        self.state, self.block = state, block\n'
PIN_ANCHOR = '    circuit_left = circuit_left.inverse()\n    circuit_left.measure_all()\n    circuit_right.measure_all()\n\n    layers_left = rewire_layers(\n        list(iter_layers(circuit_left)),\n        list(range(num_qubits)),\n        seed=seed,\n        sabre_trials=config["sabre_trials"],\n    )\n    init_meas = layers_left[-2:]\n    layers_left = layers_left[:-2]\n    layers_right = rewire_layers(\n        list(iter_layers(circuit_right)),\n        list(range(num_qubits)),\n        seed=seed,\n        sabre_trials=config["sabre_trials"],\n    )\n    final_meas = layers_right[-2:]\n    layers_right = layers_right[:-2]\n'
PIN_PATCH = '    if globals().get("_PIN_EDGE") in ("left", "right"):\n        if layout is not None:\n            raise GeneratorError("--pin-edge does not combine with initial_layout")\n        layers_left, init_meas, layers_right, final_meas = _pinned_edge_routing(\n            circuit_left, circuit_right, num_qubits, globals()["_PIN_EDGE"], seed, config["sabre_trials"]\n        )\n    else:\n        circuit_left = circuit_left.inverse()\n        circuit_left.measure_all()\n        circuit_right.measure_all()\n\n        layers_left = rewire_layers(\n            list(iter_layers(circuit_left)),\n            list(range(num_qubits)),\n            seed=seed,\n            sabre_trials=config["sabre_trials"],\n        )\n        init_meas = layers_left[-2:]\n        layers_left = layers_left[:-2]\n        layers_right = rewire_layers(\n            list(iter_layers(circuit_right)),\n            list(range(num_qubits)),\n            seed=seed,\n            sabre_trials=config["sabre_trials"],\n        )\n        final_meas = layers_right[-2:]\n        layers_right = layers_right[:-2]\n'
PIN_ANCHOR_B = '            elif state.ii_left < len(state.layers_left):\n                rewired = rewire_layers(\n                    state.layers_left[state.ii_left :] + state.init_meas,\n                    cycle.perm_left,\n'
PIN_PATCH_B = '            elif state.ii_left < len(state.layers_left) and globals().get("_PIN_EDGE") == "left":\n                # pinned left edge: never re-route the pending left layers (a re-route elides their SWAPs and lets SABRE\n                # move the edge frame); the cycle cannot have pulled swaps through this edge (engine.hows = ["right"])\n                if list(cycle.perm_left) != list(range(len(cycle.perm_left))):\n                    if _PIN_STRICT: raise GeneratorError(\'pinned left edge received a non-identity unswap permutation; set engine.hows = ["right"]\')\n                    event({"event": "pinned_edge_perm_ignored", "boundary": "left", "perm": list(cycle.perm_left)})\n                # the engine replaces each side\'s list by its PENDING layers and resets ii_left to 0 below; do the same trim here\n                state.layers_left = state.layers_left[state.ii_left :]\n            elif state.ii_left < len(state.layers_left):\n                rewired = rewire_layers(\n                    state.layers_left[state.ii_left :] + state.init_meas,\n                    cycle.perm_left,\n'
PIN_ANCHOR_C = '            if not as_perm and state.ii_right < len(state.layers_right):\n                rewired = rewire_layers(\n                    state.layers_right[state.ii_right :] + state.final_meas,\n                    cycle.perm_right,\n'
PIN_PATCH_C = '            if not as_perm and state.ii_right < len(state.layers_right) and globals().get("_PIN_EDGE") == "right":\n                if list(cycle.perm_right) != list(range(len(cycle.perm_right))):\n                    if _PIN_STRICT: raise GeneratorError(\'pinned right edge received a non-identity unswap permutation; set engine.hows = ["left"]\')\n                    event({"event": "pinned_edge_perm_ignored", "boundary": "right", "perm": list(cycle.perm_right)})\n                # the engine replaces each side\'s list by its PENDING layers and resets ii_right to 0 below; do the same trim here\n                state.layers_right = state.layers_right[state.ii_right :]\n            elif not as_perm and state.ii_right < len(state.layers_right):\n                rewired = rewire_layers(\n                    state.layers_right[state.ii_right :] + state.final_meas,\n                    cycle.perm_right,\n'
PIN_ANCHOR_D = '    return AbsorptionState(\n        mpo=mpo,\n        layers_left=layers_left,\n        layers_right=layers_right,\n        init_meas=init_meas,\n        final_meas=final_meas,\n        ii_left=0,\n        ii_right=0,\n        frame_left=FrameState(measurement_permutation(init_meas)),\n        frame_right=FrameState(measurement_permutation(final_meas)),\n        counters=counters,\n        perm_left_route=list(range(num_qubits)),\n        perm_right_route=list(range(num_qubits)),\n    )\n'
PIN_PATCH_D = '    if globals().get("_PIN_EDGE") in ("left", "right"):\n        mpo = _pin_reindex_initial_mpo(mpo, num_qubits)\n    return AbsorptionState(\n        mpo=mpo,\n        layers_left=layers_left,\n        layers_right=layers_right,\n        init_meas=init_meas,\n        final_meas=final_meas,\n        ii_left=0,\n        ii_right=0,\n        frame_left=FrameState(measurement_permutation(init_meas)),\n        frame_right=FrameState(measurement_permutation(final_meas)),\n        counters=counters,\n        perm_left_route=list(range(num_qubits)),\n        perm_right_route=list(range(num_qubits)),\n    )\n'
ROUTE_ANCHOR_L = PIN_ANCHOR_B
ROUTE_PATCH_L = '            elif state.ii_left < len(state.layers_left):\n                _route_record(state, "left", cycle.perm_left)\n                rewired = rewire_layers(\n                    state.layers_left[state.ii_left :] + state.init_meas,\n                    cycle.perm_left,\n'
ROUTE_ANCHOR_R = PIN_ANCHOR_C
ROUTE_PATCH_R = '            if not as_perm and state.ii_right < len(state.layers_right):\n                _route_record(state, "right", cycle.perm_right)\n                rewired = rewire_layers(\n                    state.layers_right[state.ii_right :] + state.final_meas,\n                    cycle.perm_right,\n'
ROUTE_ANCHOR_P = PIN_ANCHOR
ROUTE_PATCH_P = PIN_ANCHOR.replace('    circuit_right.measure_all()\n\n', '    circuit_right.measure_all()\n    _route_logical(list(iter_layers(circuit_left)), list(iter_layers(circuit_right)))\n\n')
ROUTE_ANCHOR_D = PIN_ANCHOR_D
ROUTE_PATCH_D = '    _route_initial(counters, layers_left, layers_right)\n' + PIN_ANCHOR_D
ROUTE_HELPER = '\n\ndef _route_record(state, side, perm):\n    """Persist the live routing frame for one side at one unswap cycle."""\n    import os as _os2, sys as _sys2, json as _j2\n    try:\n        _d = _os2.environ.get("WAIST_KEEP") or "."\n        with open(_os2.path.join(_d, "perm_history.jsonl"), "a") as _fh:\n            _fh.write(_j2.dumps({"side": side, "perm": [int(v) for v in perm]}) + chr(10))\n    except Exception as _e:\n        print("[engine_run_gram] route_record file FAILED: %r" % (_e,), file=_sys2.stderr)\n    try:\n        p = [int(v) for v in perm]\n        attr = "perm_left_route" if side == "left" else "perm_right_route"\n        cur = getattr(state, attr, None)\n        cur = list(cur) if cur else list(range(len(p)))\n        setattr(state, attr, [cur[i] for i in p])\n        key = "perm_" + side + "_history"\n        hist = state.counters.get(key)\n        if hist is None:\n            hist = []\n            state.counters[key] = hist\n        hist.append(p)\n    except Exception as exc:   # telemetry must never take a run down\n        try:\n            event({"event": "route_record_failed", "side": side, "error": repr(exc)})\n        except Exception:\n            pass\n\n\ndef _route_logical(layers_left, layers_right):\n    # Record the logical 2q gate sequence in LAYER order -- the order the router consumes --\n    # not circuit.data order, which DAG layering does not preserve, and after the left half has\n    # been inverted. L230 showed data order gives 46 contradictions despite matching counts.\n    # Paired with initial_wires.json (same gates, after routing) this pins wire -> qubit exactly.\n    import os as _os3, sys as _sys3, json as _j3\n    try:\n        out = {}\n        for k, ls in (("left", layers_left), ("right", layers_right)):\n            rec = []\n            for lay in ls:\n                for inst in lay.data:\n                    qs = [lay.find_bit(q).index for q in inst.qubits]\n                    if len(qs) == 2:\n                        rec.append([int(qs[0]), int(qs[1])])\n            out[k] = rec\n        d = _os3.environ.get("WAIST_KEEP") or "."\n        with open(_os3.path.join(d, "logical_gates.json"), "w") as fh:\n            _j3.dump(out, fh)\n        print("[engine_run_gram] record-frames: logical_gates.json left %d right %d"\n              % (len(out["left"]), len(out["right"])), file=_sys3.stderr)\n    except Exception as e:\n        print("[engine_run_gram] route_logical FAILED: %r" % (e,), file=_sys3.stderr)\n\n\ndef _route_initial(counters, layers_left, layers_right):\n    """Record the INITIAL routed wiring once, tying each engine gate to its circuit block."""\n    import os as _os2, sys as _sys2, json as _j2\n    try:\n        _out = {}\n        for _k, _ls in (("left", layers_left), ("right", layers_right)):\n            _rec = []\n            for _li, _lay in enumerate(_ls):\n                for _inst in _lay.data:\n                    _qs = [_lay.find_bit(_q).index for _q in _inst.qubits]\n                    if len(_qs) != 2:\n                        continue\n                    _rec.append([_li, int(_qs[0]), int(_qs[1]), 1 if _inst.operation.name == "swap" else 0])\n            _out[_k] = _rec\n        _d = _os2.environ.get("WAIST_KEEP") or "."\n        with open(_os2.path.join(_d, "initial_wires.json"), "w") as _fh:\n            _j2.dump(_out, _fh)\n        print("[engine_run_gram] record-frames: initial_wires.json left %d right %d"\n              % (len(_out["left"]), len(_out["right"])), file=_sys2.stderr)\n    except Exception as _e:\n        print("[engine_run_gram] route_initial file FAILED: %r" % (_e,), file=_sys2.stderr)\n    try:\n        for key, layers in (("initial_wires_left", layers_left),\n                            ("initial_wires_right", layers_right)):\n            rec = []\n            for li, lay in enumerate(layers):\n                for inst in lay.data:\n                    qs = [lay.find_bit(q).index for q in inst.qubits]\n                    if len(qs) != 2:\n                        continue\n                    rec.append((li, int(qs[0]), int(qs[1]),\n                                1 if inst.operation.name == "swap" else 0))\n            counters[key] = rec\n    except Exception:\n        pass\n'
PIN_ANCHOR_E = '                "event": "ordering_start",\n'
PIN_PATCH_E = '                "event": "ordering_start",\n                "pin_edge": globals().get("_PIN_EDGE"),\n                "pin_centre_site_of": globals().get("_PIN_CENTRE_SITE_OF"),\n'
PIN_HELPER = '\n\ndef _pinned_edge_routing(circuit_left, circuit_right, num_qubits, side, seed, sabre_trials):\n    """engine_run_gram --pin-edge: route the half that touches the pinned outer edge FROM that edge with the identity\n    frame (logical q at site q), then route the other half from the centre starting at the mapping the first half reached\n    there. Returns (layers_left, init_meas, layers_right, final_meas) in the exact form the unpinned code produces: layers\n    ordered from the centre outward, left layers as the routed inverted left circuit, measurement layers with clbit q = logical q."""\n    from qiskit import ClassicalRegister, QuantumCircuit\n\n    def route(circ):\n        routed = rewire_layers(list(iter_layers(circ)), list(range(num_qubits)), seed=seed, sabre_trials=sabre_trials)\n        return routed[:-2], routed[-2:]\n\n    def from_edge(forward_from_edge):\n        c = forward_from_edge.copy(); c.measure_all()\n        layers, meas = route(c)\n        centre_site_of = measurement_permutation(meas)          # logical q -> site at the centre\n        back = [layer.inverse() for layer in reversed(layers)]  # same routed gates, now ordered from the centre to the edge\n        return back, centre_site_of\n\n    def from_centre(half_from_centre, centre_site_of):\n        # engine convention (its initial_layout path): relabel so that WIRE w == centre site w, measure_all (clbit = wire),\n        # route from the trivial start. Frames the engine reports are then in relabelled-wire terms: true_frame[q] = frame[E[q]].\n        c = relabel_circuit(half_from_centre, centre_site_of); c.measure_all()\n        return route(c)\n\n    def edge_meas(centre_site_of):\n        # the pinned edge in relabelled-wire terms: wire w = E[q] sits at site q there, so clbit E[s] <- site s\n        c = QuantumCircuit(num_qubits); c.add_register(ClassicalRegister(num_qubits, "meas")); c.barrier()\n        for s_ in range(num_qubits):\n            c.measure(s_, centre_site_of[s_])\n        return list(iter_layers(c))[-2:]\n\n    if side == "left":\n        layers_left, centre_site_of = from_edge(circuit_left)\n        init_meas = edge_meas(centre_site_of)\n        layers_right, final_meas = from_centre(circuit_right, centre_site_of)\n    else:\n        layers_right, centre_site_of = from_edge(circuit_right.inverse())\n        final_meas = edge_meas(centre_site_of)\n        layers_left, init_meas = from_centre(circuit_left.inverse(), centre_site_of)\n    globals()["_PIN_CENTRE_SITE_OF"] = list(centre_site_of)\n    return layers_left, init_meas, layers_right, final_meas\n\n\ndef _pin_reindex_initial_mpo(mpo, num_qubits):\n    """optional: start the identity MPO with leg labels matching the centre mapping (logical q at site centre_site_of[q])"""\n    return mpo\n    import numpy as _np\n    E = globals().get("_PIN_CENTRE_SITE_OF") or list(range(num_qubits))\n    Einv = [int(v) for v in _np.argsort(E)]\n    ren = {}\n    for s in range(num_qubits):\n        ren[f"k{s}"] = f"k{Einv[s]}"; ren[f"b{s}"] = f"b{Einv[s]}"\n    return mpo.reindex(ren)\n'
BONDPROF_ANCHOR = '                    "work_ops_total": counters.work_ops_total,\n                    "retained_local_frobenius_log10": telemetry.retained_local_frobenius_log10(),\n'
BONDPROF_PATCH = '                    "work_ops_total": counters.work_ops_total,\n                    "retained_local_frobenius_log10": telemetry.retained_local_frobenius_log10(),\n                    **_event_extra(state, layer, telemetry, counters),\n'
BONDPROF_HELPER = '\n\ndef _event_extra(state, layer, telemetry, counters):\n    """engine_run_gram: the one hook at the layer_absorbed event site. Adds the bond profile when\n    --log-bond-profile is set, and records the retention history that --steer-after-burn reads.\n    Returns {} when neither is on, so an unflagged run is byte-identical to before."""\n    _burn_note(telemetry, counters)\n    return _bond_profile_info(state.mpo, layer)\n\n\ndef _burn_note(telemetry, counters):\n    """append (blocks absorbed, retention) to the history --steer-after-burn scores. The pair\n    matters: this fires once per absorbed LAYER and undo0.10 averages 2.3 layers per block, so a\n    window counted in entries would be 2.3x shorter than the window the threshold was calibrated\n    on. Store the block count and measure the window in blocks."""\n    if globals().get("_STEER_AFTER_BURN") is None:\n        return\n    try:\n        globals()["_BURN_HIST"].append((int(counters.work_ops_absorbed_total),\n                                        float(telemetry.retained_local_frobenius_log10())))\n    except Exception:\n        pass\n\n\ndef _steer_armed():\n    """True once the trailing-WINDOW retention burn has exceeded RATE decades/block. Latches: a run\n    that has started bleeding does not get to un-arm on a quiet stretch."""\n    if globals().get("_STEER_ARMED"):\n        return True\n    spec = globals().get("_STEER_AFTER_BURN")\n    hist = globals().get("_BURN_HIST") or []\n    if spec is None:\n        return True\n    rate, window = spec\n    if not hist:\n        return False\n    nb, nr = hist[-1]\n    j = None\n    for k in range(len(hist) - 1, -1, -1):          # oldest entry still inside the block window\n        if nb - hist[k][0] >= window:\n            j = k\n            break\n    if j is None:\n        return False\n    span = nb - hist[j][0]\n    burn = (hist[j][1] - nr) / span\n    if burn > rate:\n        globals()["_STEER_ARMED"] = True\n        print(f"[engine_run_gram] steering armed at block {nb}: "\n              f"trailing-{span}-block burn {burn:.5f} > {rate} decades/block", flush=True)\n        return True\n    return False\n\n\ndef _bond_profile_info(tn, layer):\n    """engine_run_gram --log-bond-profile: the MPO\'s per-wire bond dimensions in site order, plus\n    the qubits the incoming layer\'s two-qubit gates act on. Purely diagnostic -- it reads the\n    tensors the event already reports on and returns {} unless the flag is set, so an unflagged\n    run is byte-identical to before."""\n    if not globals().get("_LOG_BOND_PROFILE"):\n        return {}\n    try:\n        ts = list(tn.tensors)\n        prof = []\n        for a, b in zip(ts, ts[1:]):\n            shared = set(a.inds) & set(b.inds)\n            prof.append(int(max((a.ind_size(ix) for ix in shared), default=1)))\n        qs = sorted({layer.find_bit(q).index for ci in layer.data if len(ci.qubits) == 2 for q in ci.qubits})\n        # EDGES, not just qubits: whether an incoming block can cancel against material already\n        # absorbed on the other side is a question about the qubit PAIR it acts on. Logging the\n        # flattened qubit set throws that away. Names are kept so the operation is identifiable --\n        # a routing SWAP layer and a work layer touch the same wires but cost differently.\n        edges = []\n        for ci in layer.data:\n            if len(ci.qubits) != 2:\n                continue\n            a, b = (layer.find_bit(q).index for q in ci.qubits)\n            edges.append([min(a, b), max(a, b), ci.operation.name])\n        return {"bond_profile": prof, "layer_qubits": qs, "layer_edges": edges}\n    except Exception as exc:\n        return {"bond_profile_error": str(exc)[:120]}\n'
if STEER_BURN is not None and SIDE_OBJ is None and (SIDE_LOOK <= 0):
    raise SystemExit('--steer-after-burn gates --side-lookahead / --side-objective; pass one of them too')
SIDEOBJ_ANCHOR = '        chosen_counts = counts_left if do_left else counts_right'
SIDEOBJ_PATCH = '        _so = globals().get("_SIDE_OBJECTIVE")\n        _lk = globals().get("_SIDE_LOOKAHEAD") or 0\n        if globals().get("_STEER_AFTER_BURN") is not None and not _steer_armed():\n            _so, _lk = None, 0\n        if (_so is not None or _lk > 0) and mpo_left is not None and mpo_right is not None:\n            _decided = False\n            if _lk > 0:\n                _pl = _wire_bonds(state.mpo)\n                _cl = _upcoming_cost(state.layers_left, state.ii_left, _lk, _pl)\n                _cr = _upcoming_cost(state.layers_right, state.ii_right, _lk, _pl)\n                if _cl is not None and _cr is not None and _cl != _cr:\n                    do_left = _cl < _cr\n                    _decided = True\n            if not _decided and _so is not None:\n                _sl, _sr = _shape_score(mpo_left, _so), _shape_score(mpo_right, _so)\n                if _sl != _sr:\n                    do_left = _sl < _sr\n        chosen_counts = counts_left if do_left else counts_right'
SIDEOBJ_HELPER = '\n\ndef _wire_bonds(tn):\n    """per-wire bond dimensions of the current operator, in site order"""\n    ts = list(tn.tensors); prof = []\n    for a, b in zip(ts, ts[1:]):\n        shared = set(a.inds) & set(b.inds)\n        prof.append(int(max((a.ind_size(ix) for ix in shared), default=1)))\n    return prof\n\n\ndef _upcoming_cost(layers, ii, k, prof):\n    """engine_run_gram --side-lookahead: what this front\'s next k WORK layers will cost, given\n    where the bond sits right now. Absorbing onto a thin wire is cheap; onto a thick one is not,\n    and the order is the only thing we control since every layer must be absorbed eventually.\n    Returns the mean bond on the wires those layers touch, or None if the side has no work left.\n    Costs nothing beyond reading layers the engine already holds -- no candidate is built."""\n    if not prof:\n        return None\n    tot = n = seen = 0\n    for j in range(ii, min(len(layers), ii + 4 * max(1, k))):\n        layer = layers[j]\n        hit = False\n        for ci in layer.data:\n            if len(ci.qubits) != 2:\n                continue\n            hit = True\n            for q in ci.qubits:\n                w = layer.find_bit(q).index\n                for e in (w - 1, w):\n                    if 0 <= e < len(prof):\n                        tot += prof[e]; n += 1\n        if hit:\n            seen += 1\n            if seen >= k:\n                break\n    return (tot / n) if n else None\n\n\ndef _shape_score(tn, mode):\n    """engine_run_gram --side-objective: score a candidate operator by the SHAPE of its bond\n    profile rather than its total size. The default scheduler picks whichever side yields fewer\n    elements, which is myopic: the cheaper absorption now is often the one that thickens an\n    already-thick wire. "peak" minimises the largest bond; "flat" minimises max/median, the\n    inequality measure that separates surviving runs from stalling ones 4/4 on both circuits.\n    Returns +inf on any failure so a scoring error can never redirect the schedule."""\n    try:\n        ts = list(tn.tensors)\n        prof = []\n        for a, b in zip(ts, ts[1:]):\n            shared = set(a.inds) & set(b.inds)\n            prof.append(int(max((a.ind_size(ix) for ix in shared), default=1)))\n        if not prof:\n            return float("inf")\n        mx = max(prof)\n        if mode == "peak":\n            return float(mx)\n        med = sorted(prof)[len(prof) // 2] or 1\n        if mode == "floor":\n            return float(med)\n        if mode == "size":\n            import math as _m\n            return float(sum(_m.log2(max(v, 1)) for v in prof))\n        return float(mx) / float(med)\n    except Exception:\n        return float("inf")\n'

def _patched_generator():
    import hashlib, importlib.util, types
    import enigma_peaked.engine as engine_pkg
    spec = importlib.util.find_spec('enigma_peaked.engine.generator')
    source = open(spec.origin, encoding='utf-8').read()
    digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
    if digest != GENERATOR_SHA256:
        raise SystemExit(f'generator.py sha256 {digest[:16]} is not the pinned engine source ({GENERATOR_SHA256[:16]}); refusing to patch')
    helpers = ''
    if STOP_ON_STARVATION:
        if source.count(STARVE_ANCHOR) != 1:
            raise SystemExit(f'starvation-guard anchor found {source.count(STARVE_ANCHOR)} times, expected 1')
        source = source.replace(STARVE_ANCHOR, STARVE_PATCH)
        helpers += STARVE_HELPER
    if TAIL_STOP is not None:
        if STOP_ON_STARVATION:
            raise SystemExit('--tail-stop does not combine with --stop-on-starvation (same anchor)')
        if source.count(STARVE_ANCHOR) != 1:
            raise SystemExit(f'tail-stop anchor found {source.count(STARVE_ANCHOR)} times, expected 1')
        source = source.replace(STARVE_ANCHOR, TAILSTOP_PATCH)
    if FAST_TAIL_K > 0:
        if source.count(FAST_TAIL_ANCHOR) != 1:
            raise SystemExit(f'fast-tail anchor found {source.count(FAST_TAIL_ANCHOR)} times, expected 1')
        source = source.replace(FAST_TAIL_ANCHOR, FAST_TAIL_PATCH)
        helpers += FAST_TAIL_HELPER
    if OFFLOAD_ELEMS > 0:
        for anchor, patch in ((OFFLOAD_ANCHOR_A, OFFLOAD_PATCH_A), (OFFLOAD_ANCHOR_B, OFFLOAD_PATCH_B)):
            if source.count(anchor) != 1:
                raise SystemExit(f'absorb-offload anchor found {source.count(anchor)} times, expected 1')
            source = source.replace(anchor, patch)
        helpers += OFFLOAD_HELPER
    if ADAPT is not None:
        for anchor, patch in ((ADAPT_ANCHOR_A, ADAPT_PATCH_A), (ADAPT_ANCHOR_B, ADAPT_PATCH_B)):
            if source.count(anchor) != 1:
                raise SystemExit(f'adaptive-unswap anchor found {source.count(anchor)} times, expected 1')
            source = source.replace(anchor, patch)
        helpers += ADAPT_HELPER
    if FREEZE_SIDE is not None:
        if source.count(FREEZE_ANCHOR) != 1:
            raise SystemExit(f'freeze-side anchor found {source.count(FREEZE_ANCHOR)} times, expected 1')
        source = source.replace(FREEZE_ANCHOR, FREEZE_PATCH)
    if SIDE_OBJ is not None or SIDE_LOOK > 0:
        if FREEZE_SIDE is not None:
            raise SystemExit('--side-objective does not combine with --freeze-side')
        if source.count(SIDEOBJ_ANCHOR) != 1:
            raise SystemExit(f'side-objective anchor found {source.count(SIDEOBJ_ANCHOR)} times, expected 1')
        source = source.replace(SIDEOBJ_ANCHOR, SIDEOBJ_PATCH)
        helpers += SIDEOBJ_HELPER
    if LOG_BOND or STEER_BURN is not None:
        if source.count(BONDPROF_ANCHOR) != 1:
            raise SystemExit(f'bond-profile anchor found {source.count(BONDPROF_ANCHOR)} times, expected 1')
        source = source.replace(BONDPROF_ANCHOR, BONDPROF_PATCH)
        helpers += BONDPROF_HELPER
    if STOP_AT_BLOCK > 0:
        if source.count(BLOCKSTOP_ANCHOR) != 1:
            raise SystemExit(f'block-stop anchor found {source.count(BLOCKSTOP_ANCHOR)} times, expected 1')
        source = source.replace(BLOCKSTOP_ANCHOR, BLOCKSTOP_PATCH)
        helpers += BLOCKSTOP_HELPER
    if STARVE_CAP_N:
        if source.count(STARVE_CAP_ANCHOR) != 1:
            raise SystemExit(f'starve-cap anchor found {source.count(STARVE_CAP_ANCHOR)} times, expected 1')
        source = source.replace(STARVE_CAP_ANCHOR, f'    STARVE_CAP = {STARVE_CAP_N}')
    if RECORD_FRAMES:
        if PIN_EDGE is not None:
            raise SystemExit('--record-frames does not combine with --pin-edge (both patch the rewire sites)')
        for anchor, patch in ((ROUTE_ANCHOR_L, ROUTE_PATCH_L), (ROUTE_ANCHOR_R, ROUTE_PATCH_R), (ROUTE_ANCHOR_D, ROUTE_PATCH_D), (ROUTE_ANCHOR_P, ROUTE_PATCH_P)):
            if source.count(anchor) != 1:
                raise SystemExit(f'record-frames anchor found {source.count(anchor)} times, expected 1')
            source = source.replace(anchor, patch)
        helpers += ROUTE_HELPER
    if PIN_EDGE is not None:
        for anchor, patch in ((PIN_ANCHOR, PIN_PATCH), (PIN_ANCHOR_B, PIN_PATCH_B), (PIN_ANCHOR_C, PIN_PATCH_C), (PIN_ANCHOR_D, PIN_PATCH_D), (PIN_ANCHOR_E, PIN_PATCH_E)):
            if source.count(anchor) != 1:
                raise SystemExit(f'pin-edge anchor found {source.count(anchor)} times, expected 1')
            source = source.replace(anchor, patch)
        helpers += PIN_HELPER
    module = types.ModuleType(spec.name)
    module.__file__, module.__package__, module.__spec__ = (spec.origin, 'enigma_peaked.engine', spec)
    sys.modules[spec.name] = module
    exec(compile(source + helpers, spec.origin, 'exec'), module.__dict__)
    module._STOP_ON_STARVATION = bool(STOP_ON_STARVATION)
    module._FAST_TAIL_K = int(FAST_TAIL_K)
    module._TAIL_CAP = int(TAIL_CAP)
    module._OFFLOAD_ELEMS = int(OFFLOAD_ELEMS)
    module._STOP_AT_BLOCK = int(STOP_AT_BLOCK)
    module._TAIL_STOP = TAIL_STOP
    module._SIDE_OBJECTIVE = SIDE_OBJ
    module._SIDE_LOOKAHEAD = int(SIDE_LOOK)
    module._STEER_AFTER_BURN = STEER_BURN
    module._STEER_ARMED = False
    module._BURN_HIST = []
    module._LOG_BOND_PROFILE = bool(LOG_BOND)
    module._FREEZE_SIDE = FREEZE_SIDE
    module._PIN_EDGE = PIN_EDGE
    module._PIN_STRICT = os.environ.get('PIN_STRICT', '1') == '1'
    module._PIN_PERM_INV = os.environ.get('PIN_PERM_INV', '1') == '1'
    module._PIN_REINDEX = os.environ.get('PIN_REINDEX', '0') == '1'
    if ADAPT is not None:
        module._ADAPT_GROWTH, module._ADAPT_BOND_FRAC, module._ADAPT_FLOOR, module._ADAPT_BASE = (ADAPT[0], ADAPT[1], ADAPT[2], None)
    engine_pkg.generator = module
    if STOP_AT_BLOCK > 0:
        inner_absorb = module.absorb

        def absorb_or_block_stop(state, config, **kw):
            try:
                return inner_absorb(state, config, **kw)
            except module.BlockStop as ex:
                ev, ck_fn = (kw.get('event'), kw.get('checkpoint_fn'))
                if ev is not None:
                    ev({'event': 'block_stop', 'absorbed_total': ex.block, 'ii_left': ex.state.ii_left, 'ii_right': ex.state.ii_right})
                if ck_fn is not None:
                    ck_fn(ex.state, 'block-stop')
                print(f'[engine_run_gram] block stop: {ex}; checkpoint seed-block-stop saved; exiting', file=sys.stderr)
                raise SystemExit(0)
        module.absorb = absorb_or_block_stop
    if not STOP_ON_STARVATION:
        print(f"[engine_run_gram] generator patched: fast-tail-drain {FAST_TAIL_K}, tail starve cap {TAIL_CAP or 'off (sticky)'}, starve cap {STARVE_CAP_N or 24}, absorb offload above {OFFLOAD_ELEMS or 'off'} elements, stop at block {STOP_AT_BLOCK or 'off'}, adaptive unswap {(','.join((f'{v:g}' for v in ADAPT)) if ADAPT else 'off')}, tail stop {TAIL_STOP or 'off'} (sha {digest[:12]})", file=sys.stderr)
        return module
    orig_absorb = module.absorb

    def absorb_or_stop(state, config, **kw):
        try:
            return orig_absorb(state, config, **kw)
        except module.StarvationStop as ex:
            ev, ck_fn = (kw.get('event'), kw.get('checkpoint_fn'))
            if ev is not None:
                ev({'event': 'starvation_stop', 'side': ex.side, 'left_work_remaining': ex.left_work, 'right_work_remaining': ex.right_work, 'ii_left': ex.state.ii_left, 'ii_right': ex.state.ii_right})
            if ck_fn is not None:
                ck_fn(ex.state, 'starvation-stop')
            print(f'[engine_run_gram] starvation stop: {ex}; checkpoint seed-starvation-stop saved; exiting', file=sys.stderr)
            raise SystemExit(0)
    module.absorb = absorb_or_stop
    print(f'[engine_run_gram] generator patched for --stop-on-starvation (sha {digest[:12]})', file=sys.stderr)
    return module
if STOP_ON_STARVATION or TAIL_STOP is not None or FAST_TAIL_K > 0 or (OFFLOAD_ELEMS > 0) or (STOP_AT_BLOCK > 0) or (ADAPT is not None) or (PIN_EDGE is not None) or (FREEZE_SIDE is not None) or LOG_BOND or (SIDE_OBJ is not None) or (SIDE_LOOK > 0):
    gen = _patched_generator()
    if mode == 'check-patch':
        print('patched ok')
        sys.exit(0)
else:
    import enigma_peaked.engine.generator as gen
if APPLY_ZIPUP:
    import enigma_peaked.engine.mpo as _mpomod
    _orig_apply_mpo = _mpomod.apply_mpo

    def _apply_mpo_zipup(mpo1, mpo2, side, max_bond=None, cutoff=0.0, compress=True, compress_method='zipup', equalize_norms=False):
        if not compress or max_bond is None:
            return _orig_apply_mpo(mpo1, mpo2, side, max_bond, cutoff, compress, compress_method, equalize_norms)
        a, b = (mpo1, mpo2) if side == 'right' else (mpo2, mpo1)
        try:
            return a.apply(b, compress=True, contract=True, max_bond=max_bond, cutoff=cutoff)
        except Exception as exc:
            print(f'[engine_run_gram] --apply-zipup: compressed apply failed ({type(exc).__name__}: {str(exc)[:90]}); falling back to the original path', file=sys.stderr)
            return _orig_apply_mpo(mpo1, mpo2, side, max_bond, cutoff, compress, compress_method, equalize_norms)
    _mpomod.apply_mpo = _apply_mpo_zipup
    print('[engine_run_gram] apply_mpo patched: product compressed during contraction (--apply-zipup)', file=sys.stderr)
from src.peaked import gram_svd
import atexit as _atexit

@_atexit.register
def _report_svd_stats():
    st = gram_svd.STATS
    print(f"[engine_run_gram] svd stats: gram_calls {st.get('gram_calls', 0)} fallbacks {st.get('fallbacks', 0)}", flush=True)
if ROUTE_HORIZON > 0:
    from enigma_peaked.engine.layers import iter_layers as _iter_layers, merge_layers as _merge_layers
    from src.peaked.horizon_route import make_rewire_layers
    gen.rewire_layers = make_rewire_layers(ROUTE_HORIZON, _iter_layers, _merge_layers)
    print(f'[engine_run_gram] routing: best SABRE trial over the first {ROUTE_HORIZON} work gates of each re-route', file=sys.stderr)
if BOND_ROUTE is not None:
    from src.peaked.bond_route import BondProfile, make_bond_rewire
    _bond_profile = BondProfile()
    _orig_unswap = gen.unswap

    def _recording_unswap(mpo, config, **kw):
        out = _orig_unswap(mpo, config, **kw)
        _bond_profile.record(out[0])
        return out
    gen.unswap = _recording_unswap
    gen.rewire_layers = make_bond_rewire(gen.rewire_layers, _bond_profile, BOND_ROUTE[0], BOND_ROUTE[1])
    print(f'[engine_run_gram] routing: cheapest of {BOND_ROUTE[0]} candidates by bond sizes at the first {BOND_ROUTE[1]} two-qubit operations of each re-route', file=sys.stderr)
_orig_setup = gen._setup_backend

def _setup_with_gram(plan):
    backend = _orig_setup(plan)
    if backend.name == 'torch':
        gram_svd.install(min_dim=GRAM_MIN_DIM)
        if LOG_SPECTRUM is not None:
            path, mindim, every = LOG_SPECTRUM
            gram_svd.SPECTRUM.update(path=path, min_dim=mindim, every=every)
            print(f'[engine_run_gram] spectrum log -> {path} (min_dim {mindim}, every {every})', flush=True)
    return backend
gen._setup_backend = _setup_with_gram
from src.peaked.engine_local import make_local_apply
if LOCAL_BUDGET is not None:
    gen.apply_qiskit_circuit_strict_chain = make_local_apply(LOCAL_BUDGET)
import os as _os
if _os.environ.get('WAIST_KEEP'):
    import json as _wj, shutil as _wsh, enigma_peaked.engine.checkpoint as _wck
    _wdir = _os.environ['WAIST_KEEP']
    _wmax = int(_os.environ.get('WAIST_KEEP_MAXBOND', '512'))
    _os.makedirs(_wdir, exist_ok=True)
    _worig = _wck.save_checkpoint
    import time as _wtime
    _wseq = [0]

    def _save_and_keep(path, state):
        rc = _worig(path, state)
        try:
            co = state.get('counters')
            cyc = getattr(co, 'unswap_cycles', None) if co is not None else None
            blk = None
            for _k in ('absorbed_total', 'absorbed', 'work_absorbed', 'blocks_absorbed', 'n_absorbed'):
                if co is not None and getattr(co, _k, None) is not None:
                    blk = getattr(co, _k)
                    break
            if blk is None:
                for _k in ('absorbed_total', 'absorbed', 'blocks_absorbed', 'n_absorbed'):
                    if state.get(_k) is not None:
                        blk = state.get(_k)
                        break
            if blk is None and cyc is not None:
                blk = f'c{cyc}'
            m = state.get('mpo')
            mb = 0
            try:
                arrs = m['arrays'] if isinstance(m, dict) else getattr(m, 'arrays', None)
                if arrs is None:
                    arrs = getattr(m, 'sites', None)
                mb = max((int(a.shape[0]) for a in arrs)) if arrs else 0
            except Exception:
                mb = 0
            try:
                sz = _os.path.getsize(path)
            except Exception:
                sz = 0
            _wseq[0] += 1
            small = mb <= _wmax if mb else sz <= 40000000
            if small:
                dst = _os.path.join(_wdir, f'seq{_wseq[0]:05d}-cyc{cyc}-blk{blk}-bond{mb}.ckpt')
                _wsh.copyfile(path, dst)
                with open(_os.path.join(_wdir, 'index.tsv'), 'a') as _f:
                    _f.write(f'{_wseq[0]}\t{_wtime.time():.3f}\t{cyc}\t{blk}\t{mb}\t{_os.path.getsize(dst)}\t{dst}\n')
            else:
                with open(_os.path.join(_wdir, 'skipped.tsv'), 'a') as _f:
                    _f.write(f'{_wseq[0]}\t{_wtime.time():.3f}\t{cyc}\t{blk}\t{mb}\t{sz}\n')
        except Exception as _e:
            try:
                with open(_os.path.join(_wdir, 'index.tsv'), 'a') as _f:
                    _f.write(f'ERR\t-\t-\t-\t{type(_e).__name__}\n')
            except Exception:
                pass
        return rc
    _wck.save_checkpoint = _save_and_keep
    import sys as _wsys
    for _mname, _mod in list(_wsys.modules.items()):
        if not _mname.startswith('enigma_peaked'):
            continue
        if getattr(_mod, 'save_checkpoint', None) is _worig:
            setattr(_mod, 'save_checkpoint', _save_and_keep)
            print(f'[engine_run_gram] WAIST_KEEP: hooked save_checkpoint in {_mname}', file=sys.stderr)
    print(f'[engine_run_gram] WAIST_KEEP active -> {_wdir} (maxbond {_wmax})', file=sys.stderr)
if _os.environ.get('FRAME_DUMP'):
    import json as _json, enigma_peaked.engine.checkpoint as _ck
    _dump = _os.environ['FRAME_DUMP']
    _orig_save = _ck.save_checkpoint

    def _save_and_dump(path, state):
        try:
            fl = list(state.get('frame_left') or [])
            fr = list(state.get('frame_right') or [])
            co = state.get('counters')
            cyc = getattr(co, 'unswap_cycles', None) if co is not None else None
            mb = 0
            m = state.get('mpo')
            try:
                mb = max((int(a.shape[0]) for a in m['arrays'])) if isinstance(m, dict) else 0
            except Exception:
                mb = 0
            open(_dump, 'a').write(_json.dumps({'cycle': cyc, 'frame_left': [int(x) for x in fl], 'frame_right': [int(x) for x in fr], 'max_bond': mb}) + '\n')
        except Exception as _e:
            open(_dump, 'a').write(_json.dumps({'err': type(_e).__name__}) + '\n')
        return _orig_save(path, state)
    _ck.save_checkpoint = _save_and_dump
    import sys as _fsys
    for _fname, _fmod in list(_fsys.modules.items()):
        if not _fname.startswith('enigma_peaked'):
            continue
        if getattr(_fmod, 'save_checkpoint', None) is _orig_save:
            setattr(_fmod, 'save_checkpoint', _save_and_dump)
            print(f'[engine_run_gram] FRAME_DUMP: hooked save_checkpoint in {_fname}', file=sys.stderr)
from enigma_peaked.cli import main
if mode == 'resume':
    import enigma_peaked.engine.checkpoint as ck
    relaxed = {'max_bond', 'unswap_threshold', 'max_its', 'probe_cutoff', 'gate_mpo_mode', 'hows', 'apply_cutoff'}
    if argv[:1] == ['--relax-cutoff']:
        relaxed |= {'cutoff', 'final_cutoff'}
        argv = argv[1:]
    ck._ENGINE_CRITICAL_KEYS = tuple((k for k in ck._ENGINE_CRITICAL_KEYS if k not in relaxed))
    if RESUME_SEED is not None:
        _orig_load = gen.load_checkpoint

        def _load_with_seed(*a, **k):
            payload = _orig_load(*a, **k)
            print(f"[engine_run_gram] resume seed {payload['seed']} -> {RESUME_SEED}: routing after the checkpoint uses the new seed", file=sys.stderr)
            payload['seed'] = RESUME_SEED
            return payload
        gen.load_checkpoint = _load_with_seed
    from enigma_peaked.cli import main
    sys.exit(main(['resume-generator', *argv]))
sys.exit(main(['run-generator', *argv]))
