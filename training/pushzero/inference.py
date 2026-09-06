"""Synchronous, bounded inference ownership and exact-input caching.

Staging never escapes its locked call. A completed `.numpy()` readback is the
lease boundary: only then may host arrays be refilled. Returned arrays own
their memory. No NumPy pointer is misrepresented as a Metal buffer handle.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import threading
import time

import numpy as np
from tinygrad import Context, Device, Tensor, TinyJit
from .protocol import ACTION_LIMITS, BOARD_SHAPE, action_bucket, bucket, pack_observations


@dataclass
class InferenceMetrics:
    requests: int = 0
    requested_rows: int = 0
    evaluated_rows: int = 0
    submitted_rows: int = 0
    legal_actions: int = 0
    submitted_actions: int = 0
    cache_hits: int = 0
    duplicate_rows: int = 0
    graph_misses: int = 0
    host_input_bytes: int = 0
    host_output_bytes: int = 0
    validation_cache_seconds: float = 0.0
    staging_seconds: float = 0.0
    transfer_dispatch_seconds: float = 0.0
    forward_readback_seconds: float = 0.0


class ExactCache:
    """LRU bounded by retained key/output payload bytes (not Python overhead)."""
    def __init__(self, capacity_bytes=0):
        if capacity_bytes < 0:
            raise ValueError("negative cache budget")
        self.capacity, self.bytes, self.items = int(capacity_bytes), 0, OrderedDict()

    def clear(self):
        self.items.clear()
        self.bytes = 0

    def get(self, key):
        found = self.items.get(key)
        if found is not None:
            self.items.move_to_end(key)
            return found[0]
        return None

    def put(self, key, logits, value):
        size = sum(len(k) for k in key) + logits.nbytes + 4
        if size > self.capacity:
            return
        previous = self.items.pop(key, None)
        if previous is not None:
            self.bytes -= previous[1]
        while self.items and self.bytes + size > self.capacity:
            _, (_, old_size) = self.items.popitem(last=False)
            self.bytes -= old_size
        immutable = logits.copy()
        immutable.flags.writeable = False
        self.items[key] = ((immutable, value), size)
        self.bytes += size


class Predictor:
    def __init__(self, model, batch_size=32, jit=True, *, cache_bytes=0, max_graphs=32,
                 workers=1, tail_buckets=True):
        if batch_size < 1 or max_graphs < 1 or workers < 0:
            raise ValueError("invalid predictor capacity")
        self.model, self.batch_size, self.jit = model, int(batch_size), jit
        self.max_graphs, self.workers, self.tail_buckets = max_graphs, workers, tail_buckets
        self.compiled, self.staging = OrderedDict(), {}
        self.cache, self.metrics = ExactCache(cache_bytes), InferenceMetrics()
        self.positions = self.seconds = self.native_seconds = self.search_calls = 0
        self.revision = model.revision
        self._lock = threading.RLock()
        self._runtime = None

    @property
    def with_effects(self):
        return bool(self.model.config.effect_channels)

    def close(self):
        with self._lock:
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
            self.compiled.clear()
            self.staging.clear()
            self.cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def statistics(self):
        return {**asdict(self.metrics), "cache_payload_bytes": self.cache.bytes,
                "graphs": len(self.compiled), "seconds": self.seconds,
                "timing_note": "host timers; forward_readback includes device execution and synchronization"}

    def _invalidate(self):
        if self.revision != self.model.revision:
            # Inference graphs may retain transformed weights, not just inputs.
            self.compiled.clear()
            self.staging.clear()
            self.cache.clear()
            self.revision = self.model.revision

    def __call__(self, observations):
        if not observations:
            return []
        boards, actions, lengths, effects = pack_observations(observations, self.with_effects)
        logits, values = self.packed(boards, actions, lengths=lengths, effects=effects)
        return [(logits[i, :n].copy(), float(values[i])) for i, n in enumerate(lengths)]

    def _graph(self, rows, width, tokens):
        key = (rows, width, tokens)
        if key not in self.compiled:
            while len(self.compiled) >= self.max_graphs:
                evicted, _ = self.compiled.popitem(last=False)
                self.staging.pop(evicted)
            def forward(x, a, *extra):
                p, v = self.model(x, a, extra[0] if extra else None)
                wdl = v.softmax()
                # One readback/synchronization boundary, not one per output head.
                return p.cat((wdl[:, 0] - wdl[:, 2]).unsqueeze(1), dim=1).realize()
            self.compiled[key] = TinyJit(forward) if self.jit else forward
            arrays = [np.zeros((rows, *BOARD_SHAPE), np.float32),
                      np.zeros((rows, width, 6), np.int32)]
            if tokens:
                arrays.append(np.zeros((rows, tokens, 4), np.int32))
            self.staging[key] = arrays
            self.metrics.graph_misses += 1
        self.compiled.move_to_end(key)
        return self.compiled[key], self.staging[key]

    def packed(self, input_boards, input_actions, *, lengths=None, effects=None):
        with self._lock:
            return self._packed(input_boards, input_actions, lengths, effects)

    def _packed(self, boards, actions, lengths, effects):
        begin = time.monotonic()
        self._invalidate()
        boards = np.asarray(boards, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32)
        if boards.ndim != 4 or boards.shape[1:] != BOARD_SHAPE or actions.ndim != 3 or actions.shape[2] != 6 or len(boards) != len(actions):
            raise ValueError("invalid inference batch shapes")
        n, original_width = actions.shape[:2]
        lengths = np.full(n, original_width, np.int32) if lengths is None else np.asarray(lengths, dtype=np.int32)
        if lengths.shape != (n,) or (lengths < 1).any() or (lengths > original_width).any() or not np.isfinite(boards).all():
            raise ValueError("invalid board or legal lengths")
        if self.with_effects:
            if effects is None:
                raise ValueError("missing exact effects")
            effects = np.asarray(effects, dtype=np.int32)
            if effects.ndim != 3 or effects.shape[0] != n or effects.shape[2] != 4:
                raise ValueError("invalid effect shape")
        elif effects is not None:
            effects = None  # baseline function has no effect input or cache dependency
        output = np.zeros((n, original_width), np.float32)
        values = np.empty(n, np.float32)
        unique, pending, keys, token_rows = {}, [], {}, {}
        for i, count in enumerate(lengths):
            legal = actions[i, :count]
            if (legal < 0).any() or (legal >= ACTION_LIMITS).any():
                raise ValueError("out-of-range action encoding")
            token = None
            if effects is not None:
                token = effects[i, effects[i, :, 0] != 0]
                if (token < 0).any() or (token[:, 0] > count).any() or (token[:, 1] >= 64).any() or (token[:, 2:] >= 13).any():
                    raise ValueError("out-of-range effect encoding")
                token_rows[i] = token
            if not self.cache.capacity:
                pending.append(i)
                continue
            key = (boards[i].tobytes(), legal.tobytes(), b"" if token is None else token.tobytes())
            keys[i] = key
            cached = self.cache.get(key) if self.cache.capacity else None
            if cached is not None:
                output[i, :count], values[i] = cached
                self.metrics.cache_hits += 1
            elif key in unique and self.cache.capacity:
                unique[key].append(i)
                self.metrics.duplicate_rows += 1
            else:
                unique[key] = [i]
                pending.append(i)
        self.metrics.validation_cache_seconds += time.monotonic() - begin
        self.metrics.requests += 1
        self.metrics.requested_rows += n
        self.metrics.legal_actions += int(lengths.sum())
        for start in range(0, len(pending), self.batch_size):
            indices = pending[start:start + self.batch_size]
            count = len(indices)
            rows = min(self.batch_size, bucket(count)) if self.tail_buckets else self.batch_size
            width = action_bucket(max(int(lengths[i]) for i in indices))
            tokens = bucket(max(len(token_rows[i]) for i in indices), 16) if effects is not None else 0
            tick = time.monotonic()
            forward, staging = self._graph(rows, width, tokens)
            for array in staging:
                array[count:].fill(0)
            for row, i in enumerate(indices):
                staging[0][row] = boards[i]
                staging[1][row, :lengths[i]] = actions[i, :lengths[i]]
                staging[1][row, lengths[i]:].fill(0)
                if tokens:
                    staging[2][row, :len(token_rows[i])] = token_rows[i]
                    staging[2][row, len(token_rows[i]):].fill(0)
            self.metrics.staging_seconds += time.monotonic() - tick
            tick = time.monotonic()
            inputs = [Tensor(array, device=Device.DEFAULT) for array in staging]
            self.metrics.transfer_dispatch_seconds += time.monotonic() - tick
            tick = time.monotonic()
            with Context(TRAINING=0):
                result = forward(*inputs).numpy()
            self.metrics.forward_readback_seconds += time.monotonic() - tick
            if not np.isfinite(result).all():
                raise FloatingPointError("non-finite inference")
            self.metrics.evaluated_rows += count
            self.metrics.submitted_rows += rows
            self.metrics.submitted_actions += rows * width
            self.metrics.host_input_bytes += sum(a.nbytes for a in staging)
            self.metrics.host_output_bytes += result.nbytes
            for row, i in enumerate(indices):
                legal_logits, value = result[row, :lengths[i]], float(result[row, -1])
                destinations = unique[keys[i]] if self.cache.capacity else [i]
                for dest in destinations:
                    output[dest, :lengths[dest]], values[dest] = legal_logits, value
                if self.cache.capacity:
                    self.cache.put(keys[i], legal_logits, value)
        self.positions += n
        self.seconds += time.monotonic() - begin
        return output, values
