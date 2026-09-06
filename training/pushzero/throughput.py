"""Frozen-weight rolling self-play sweep. No optimizer or training updates."""
from dataclasses import asdict
import json
import os
import platform
import plistlib
import subprocess
import threading
import time
import numpy as np
from .learning import load_checkpoint, resolve_checkpoint
from .inference import Predictor
from .selfplay import RollingCollector


class DeviceSampler:
    """Whole-machine GPU activity and this process's RSS, not shader occupancy."""
    def __init__(self):
        self.stop = threading.Event()
        self.rows, self.error = [], None
        self.thread = threading.Thread(target=self.sample, name="device-sampler", daemon=True)

    def sample(self):
        try:
            while not self.stop.is_set():
                registry = plistlib.loads(subprocess.check_output(["/usr/sbin/ioreg", "-a", "-r", "-c", "AGXAccelerator"]))
                gpu = registry[0]["PerformanceStatistics"]["Device Utilization %"]
                rss = int(subprocess.check_output(["/bin/ps", "-p", str(os.getpid()), "-o", "rss="])) * 1024
                self.rows.append({"time": time.monotonic(), "gpu_percent": gpu, "rss_bytes": rss})
                self.stop.wait(1)
        except Exception as error:
            self.error = error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        if self.error is not None: raise RuntimeError("GPU measurement failed") from self.error


def sweep(checkpoint, settings, *, repeats=2, seconds=15., warmup=10., workers=2, group_size=16, progress=print):
    if repeats < 1 or min(seconds, warmup) <= 0 or not settings:
        raise ValueError("invalid sweep duration/repetitions")
    checkpoint = resolve_checkpoint(checkpoint)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    model, info = load_checkpoint(checkpoint)
    configs = [(f"actors-{a}-batch-{b}", a, b) for a,b in settings]
    configs.append(("identity-" + configs[0][0], *settings[0]))
    rng, measurements = np.random.default_rng(39281), []
    for repeat in range(repeats):
        for index in rng.permutation(len(configs)):
            name, actors, batch = configs[index]
            if progress: progress(json.dumps({"event": "probe_started", "arm": name, "repeat": repeat}))
            with Predictor(model, batch, workers=workers, group_size=group_size, cache_bytes=64 << 20) as p:
                c = RollingCollector(p, np.random.default_rng(74123 + repeat), actors=actors,
                    simulations=64, fast_simulations=16, full_fraction=.25, max_plies=64,
                    curriculum=.25, fast_explore=False)
                begin = time.monotonic()
                c.collect(10**9, deadline=begin + warmup)
                warm_seconds = time.monotonic() - begin
                before, before_moves, before_games = p.statistics(), c.moves, c.completed
                with DeviceSampler() as device:
                    begin = time.monotonic()
                    samples, games = c.collect(10**9, deadline=begin + seconds)
                    elapsed = time.monotonic() - begin
                after = p.statistics()
                delta = {k: after[k] - before[k] for k in asdict(p.metrics)}
                row = {"arm": name, "repeat": repeat, "actors": actors, "inference_batch_size": batch,
                    "warmup_seconds": warm_seconds, "seconds": elapsed, "counters": delta,
                    "moves": c.moves - before_moves, "games": c.completed - before_games, "samples": len(samples),
                    "requested_positions_per_second": delta["requested_rows"] / elapsed,
                    "evaluated_positions_per_second": delta["evaluated_rows"] / elapsed,
                    "moves_per_second": (c.moves - before_moves) / elapsed,
                    "samples_per_second": len(samples) / elapsed,
                    "mean_useful_device_batch": delta["evaluated_rows"] / max(1, delta["device_batches"]),
                    "whole_gpu_mean_percent": float(np.mean([r["gpu_percent"] for r in device.rows])),
                    "rss_peak_bytes": max(r["rss_bytes"] for r in device.rows),
                    "device_samples": device.rows, "native": p.last_search_metrics,
                    "selfplay": c.statistics(), "terminal_games": sum(g["white_outcome"] is not None for g in games)}
                measurements.append(row)
                if progress: progress(json.dumps({"event": "probe_completed", **{k:v for k,v in row.items() if k not in ("device_samples", "counters")}}))
    return {"format": 1, "checkpoint": str(checkpoint), "steps": info["steps"], "model": info["model"],
        "revision": revision, "dirty": dirty,
        "machine": platform.platform(), "workers": workers, "search_group_size": group_size,
        "measurements": measurements, "limitations": "Frozen network, no learning. Seeded mixed starts; scheduling changes RNG order. "
        "Same search budgets; 64-ply benchmark cap. Cold warm-up reported separately, later new-shape compilations remain in counters. "
        "GPU counters are system-wide, not training-only FLOPS. RSS includes retained runtime/allocator memory from earlier arms. "
        "Confirm selected settings in isolated processes and in real training; no playing-strength claim."}
