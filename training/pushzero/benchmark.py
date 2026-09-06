"""Opt-in inference/search probes. Importing this file runs no probes."""
from dataclasses import asdict
import importlib.metadata
import platform
import time
import numpy as np
from ._native import State
from .model import ModelConfig, Network, Predictor
from .search import search


# Deliberately different rule regimes; plus reachable seeded trajectories.
FENS = (
    None,
    "7k/8/8/4R3/4N3/8/8/K7 w - - 0 1",
    "7k/P7/R7/8/8/8/8/K7 w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "7k/8/8/3pP3/8/8/8/K7 w - d6 0 1",
    "7k/8/5KQ1/8/8/8/8/8 w - - 0 1",
)


def corpus(seed=53719, plies=96):
    states = [State(fen) for fen in FENS]
    rng, state = np.random.default_rng(seed), State()
    for ply in range(plies):
        if state.outcome() is not None: state = State()
        state.play(int(rng.choice(state.legal_ids())))
        if state.outcome() is None and ply % 4 == 0: states.append(state.copy())
    return states


def probe(config=ModelConfig(), actors=32, repeats=10, workers=1, cache_mb=0, search_simulations=0):
    if actors < 1 or repeats < 1: raise ValueError("invalid probe size")
    from tinygrad import Device
    states = corpus()
    roots = [states[i % len(states)] for i in range(actors)]
    results = []
    model = Network(config)
    # A=fixed rows, B=tail buckets, identity=A. Each owns independent cache/JIT.
    with Predictor(model, actors, workers=workers, cache_bytes=cache_mb << 20, tail_buckets=False) as a, \
         Predictor(model, actors, workers=workers, cache_bytes=cache_mb << 20) as b, \
         Predictor(model, actors, workers=workers, cache_bytes=cache_mb << 20, tail_buckets=False) as control:
        predictors = {"A-fixed": a, "B-tail": b, "A-identity": control}
        batches = [[s.observation_with_effects() if a.with_effects else s.observation() for s in roots[:n]]
                   for n in sorted({1, max(1, actors // 3), actors})]
        cold = {}
        for name, p in predictors.items():
            begin = time.monotonic()
            for rows in batches:
                for _ in range(3): p(rows)
            cold[name] = time.monotonic() - begin
            p.cache.clear()
        rng = np.random.default_rng(62831)
        for repeat in range(repeats):
            for name in rng.permutation(list(predictors)):
                p = predictors[name]
                begin = time.monotonic()
                for rows in batches: p(rows)
                results.append({"arm": str(name), "repeat": repeat, "seconds": time.monotonic() - begin,
                                "requested_rows": sum(len(rows) for rows in batches)})
        search_metrics = None
        if search_simulations:
            begin = time.monotonic()
            search(roots, b, rng, simulations=search_simulations, explore=False)
            search_metrics = {"seconds": time.monotonic() - begin, **b.last_search_metrics}
        return {"format": 1, "model": asdict(config), "parameters": model.parameter_count,
                "device": Device.DEFAULT, "tinygrad": importlib.metadata.version("tinygrad"),
                "machine": platform.machine(), "os": platform.platform(), "actors": actors, "workers": workers,
                "cache_mb": cache_mb, "cold_warmup_seconds": cold, "samples": results,
                "counters": {n: p.statistics() for n, p in predictors.items()}, "search": search_metrics,
                "limitations": "Host wall timers, not GPU kernel attribution. Cache-on reuses this corpus intentionally. "
                                "Run cache-off too; compare absolute timings in isolation. No strength inference."}
