"""One native search scheduler, shared by rolling collection and finite queries.

Search groups own trees. Inference leases own reply routing. The device owner
serves one lease while independent native jobs prepare subsequent work.
"""
from dataclasses import dataclass
import time
import numpy as np
from ._native import SearchRuntime, observations
from .protocol import action_bucket


@dataclass
class SearchResult:
    move: int
    observation: tuple
    policy: np.ndarray
    visits: np.ndarray
    nodes: int
    root_value: float = 0.0
    selected_value: float = 0.0


class SearchDriver:
    def __init__(self, predictor, actors):
        self.predictor = predictor
        self.group_size = min(predictor.group_size, predictor.batch_size)
        lanes = (actors + self.group_size - 1) // self.group_size
        if lanes < 1:
            raise ValueError("search requires actors")
        runtime = predictor._runtime
        if runtime is not None and not runtime.idle:
            raise RuntimeError("predictor already owns active searches")
        if runtime is None or runtime.lane_count < lanes:
            if runtime is not None: runtime.close()
            runtime = predictor._runtime = SearchRuntime(predictor.workers, lanes, predictor.batch_size)
        self.runtime, self.roots = runtime, {}
        self.native_begin, self.calls_begin = runtime.native_seconds, runtime.ffi_calls

    def start(self, lane, states, rng, simulations, candidates=16, explore=True, max_nodes=16384):
        roots = [tuple(o) if self.predictor.with_effects else tuple(o[:3])
                 for o in observations(states, self.predictor.with_effects)]
        width = action_bucket(max(len(o[1]) for o in roots))
        noise = (rng.gumbel(size=(len(states), width)).astype(np.float32) if explore
                 else np.zeros((len(states), width), np.float32))
        self.runtime.start(lane, states, noise, simulations, candidates,
                           effects=self.predictor.with_effects, max_nodes=max_nodes)
        self.roots[lane] = roots

    def poll(self):
        request, completed = self.runtime.poll(1000)
        # Returned replies start native CPU work before the next GPU call.
        if request is not None:
            request_id, boards, actions, lengths, tokens = request
            logits, values = self.predictor.packed(boards, actions, lengths=lengths, effects=tokens)
            self.runtime.submit(request_id, logits, values)
        return [(lane, [SearchResult(move, root, policy, visits, nodes, raw, chosen)
                        for root, (move, policy, visits, nodes, raw, chosen)
                        in zip(self.roots.pop(lane), rows, strict=True)]) for lane, rows in completed]

    def finish(self):
        if self.roots or not self.runtime.idle:
            raise RuntimeError("cannot release unfinished searches")
        p = self.predictor
        p.native_seconds += self.runtime.native_seconds - self.native_begin
        p.search_calls += self.runtime.ffi_calls - self.calls_begin
        p.last_search_metrics = self.runtime.metrics()

    def abort(self):
        self.runtime.close()
        self.predictor._runtime = None


def search(states, predictor, rng, simulations=64, candidates=16, explore=True, *,
           deadline=float("inf"), stop=lambda: False, max_nodes=16384):
    if simulations < 1 or candidates < 1:
        raise ValueError("simulations and candidates must be positive")
    if not states:
        return []
    driver = SearchDriver(predictor, len(states))
    outputs = [None] * len(states)
    try:
        for lane, start in enumerate(range(0, len(states), driver.group_size)):
            driver.start(lane, states[start:start + driver.group_size], rng, simulations, candidates, explore, max_nodes)
        while driver.roots:
            if stop() or time.monotonic() >= deadline: driver.runtime.stop()
            for lane, results in driver.poll():
                start = lane * driver.group_size
                outputs[start:start + len(results)] = results
        driver.finish()
        return outputs
    except BaseException:
        driver.abort()
        raise
