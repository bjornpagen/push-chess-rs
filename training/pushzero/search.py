"""The Python search loop is just a batched neural-evaluation service.

Rust owns scheduling, legal actions, tree traversal, terminal values, backup,
and completed-Q policy targets. No per-tree/per-action hot-loop FFI calls.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import time
import numpy as np
from ._native import SearchBatch, SearchRuntime, observations
from .model import action_bucket


@dataclass
class SearchResult:
    move: int
    observation: tuple
    policy: np.ndarray
    visits: np.ndarray
    nodes: int
    root_value: float = 0.0
    selected_value: float = 0.0


def search(states, predictor, rng, simulations=64, candidates=16, explore=True, *,
           deadline=float("inf"), stop=lambda: False, max_nodes=16384):
    # One coordinator per predictor/model. The inference lock is reentrant, so
    # its packed calls remain safe inside a whole-search lease.
    with getattr(predictor, "_lock", nullcontext()):
        return _search(states, predictor, rng, simulations, candidates, explore, deadline=deadline, stop=stop, max_nodes=max_nodes)


def _search(states, predictor, rng, simulations, candidates, explore, *, deadline, stop, max_nodes):
    if simulations < 1 or candidates < 1:
        raise ValueError("simulations and candidates must be positive")
    if not states:
        return []
    effects = getattr(predictor, "with_effects", False)
    roots = [tuple(o) if effects else tuple(o[:3]) for o in observations(states, effects)]
    width = action_bucket(max(len(o[1]) for o in roots))
    noise = (rng.gumbel(size=(len(states), width)).astype(np.float32) if explore
             else np.zeros((len(states), width), np.float32))
    pooled = getattr(predictor, "workers", 1) != 1
    try:
        if pooled:
            if predictor._runtime is None:
                predictor._runtime = SearchRuntime(predictor.workers)
            batch = predictor._runtime
            batch.start(states, noise, simulations, candidates, effects=effects, max_nodes=max_nodes)
        else:
            batch = SearchBatch(states, noise, simulations, candidates, effects=effects, max_nodes=max_nodes)
        request = batch.advance(stop=stop() or time.monotonic() >= deadline)
        while request is not None:
            request_id, boards, actions, lengths, tokens = request
            if hasattr(predictor, "with_effects"):
                logits, values = predictor.packed(boards, actions, lengths=lengths, effects=tokens)
            else:  # simple framework-independent predictors remain valid
                logits, values = predictor.packed(boards, actions)
            request = batch.advance(request_id, logits, values,
                                    stop=stop() or time.monotonic() >= deadline)
        predictor.native_seconds += batch.native_seconds
        predictor.search_calls += batch.ffi_calls
        predictor.last_search_metrics = batch.metrics()
        return [SearchResult(move, root, policy, visits, nodes, raw, chosen)
                for root, (move, policy, visits, nodes, raw, chosen)
                in zip(roots, batch.finish(), strict=True)]
    except BaseException:
        if pooled and predictor._runtime is not None:
            predictor._runtime.close()
            predictor._runtime = None
        raise
