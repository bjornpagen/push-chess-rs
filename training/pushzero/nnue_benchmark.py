"""Bounded NNUE backend comparison; no corpus access or saved training.

Calls alternate order within each repetition and use identical rotating inputs.
Inference includes input preparation, submission, synchronization and readback;
resident tensor rows separately bound the cost when inputs already live there.
Training measures the existing complete Learner update, not just matrix multiply.
"""
import importlib.metadata
import platform
import time

import numpy as np
from tinygrad import Context, Device, Tensor, TinyJit
from tinygrad.helpers import NUM_CPU_THREADS

from ._native import NNUE_CONTROL, Nnue, State, nnue_inputs
from .nnue import Learner, Model, dense_features


def _summary(values):
    return {"median_us": float(np.median(values)), "p10_us": float(np.percentile(values,10)),
            "p90_us": float(np.percentile(values,90))}


def _measure(arms, inputs, repeats):
    # Each path warms through JIT capture; the timed phase excludes compilation.
    for _ in range(4):
        for run in arms.values(): run(inputs[0])
    timings = {key: [] for key in arms}
    keys = list(arms)
    for rep in range(repeats):
        for key in keys if rep % 2 == 0 else keys[::-1]:
            begin = time.perf_counter_ns()
            arms[key](inputs[rep % len(inputs)])
            timings[key].append((time.perf_counter_ns()-begin)/1000)
    return {key: _summary(values) for key, values in timings.items()}


def _tensor_arm(model, banks):
    device = model.features.device
    compiled = TinyJit(lambda features, baseline, side: model(features,baseline,side).realize())
    def tensors(row):
        ids, baseline, side = row
        return tuple(Tensor(x,device=device).realize() for x in
                     (dense_features(ids),baseline.astype(np.float32),side.astype(np.float32)))
    resident = [tensors(row) for row in banks]
    def run(index):
        with Context(TRAINING=0, TC=0): return compiled(*tensors(banks[index])).numpy()
    def ready(index):
        with Context(TRAINING=0, TC=0): return compiled(*resident[index]).numpy()
    return run, ready


def compare(batches=(1,32,256,1024), repeats=15):
    batches = sorted(set(batches))
    if not batches or min(batches) < 1 or max(batches) > 4096 or not 5 <= repeats <= 101:
        raise ValueError("batch sizes must be 1..4096 and repeats 5..101")
    rng = np.random.default_rng(419)
    states, state = [], State()
    for _ in range(max(batches) + 3):
        if state.outcome() is not None: state = State()
        states.append(state.copy())
        state.play(int(rng.choice(state.legal_ids())))
    ids, baseline = nnue_inputs(states[:max(batches)])
    sides = np.array([s.turn() for s in states[:max(batches)]],np.uint8)
    native = Nnue(NNUE_CONTROL)
    result = {"machine": platform.machine(), "macos": platform.mac_ver()[0],
              "tinygrad": importlib.metadata.version("tinygrad"), "cpu_tensor_threads": NUM_CPU_THREADS.value,
              "repeats": repeats, "compiled_warmup": 4, "rows": [],
              "regime": "current machine load; remeasure idle before small performance claims",
              "inference": "CPU sparse accumulator versus identical tinygrad forward on CPU/Metal; all return host scores",
              "training": "complete QAT forward/backward/AdamW, transfers and synchronized metrics; temporary weights only"}
    for size in batches:
        banks = []
        for shift in range(4):
            take = (np.arange(size)+shift) % len(ids)
            banks.append(tuple(np.ascontiguousarray(x[take]) for x in (ids,baseline,sides)))
        arms = {"native_cpu": lambda i: native.scores(*banks[i])}
        for device in ("CPU", "METAL"):
            run, ready = _tensor_arm(Model(device=device), banks)
            arms[f"tinygrad_{device.lower()}"] = run
            arms[f"tinygrad_{device.lower()}_resident"] = ready
        for i in range(len(banks)):
            expected = native.scores(*banks[i])
            for run in arms.values(): np.testing.assert_array_equal(run(i),expected)
        inference = _measure(arms, list(range(len(banks))), repeats)
        # Same initial weights and same evolving optimizer steps in both arms.
        training_banks = [(dense_features(a),b.astype(np.float32),c.astype(np.float32),
                           rng.integers(0,3,size=size).astype(np.float32)/2) for a,b,c in banks]
        learners = {device: Learner(Model(device=device)) for device in ("CPU","METAL")}
        training = _measure({key: learner.train for key,learner in learners.items()},training_banks,repeats)
        row = {"batch":size,"inference":inference,"training":training}
        result["rows"].append(row)
        print({"nnue_benchmark_batch_complete":size},flush=True)
    # Ensure every queued device operation has completed before returning.
    for device in ("CPU","METAL"): Device[device].synchronize()
    return result
