"""Versioned, pickle-free replay shards. Truncations never become draw labels."""
from dataclasses import dataclass, field
from collections import OrderedDict
import json
import os
from pathlib import Path
import uuid

import numpy as np
from ._native import State, RULES_VERSION, ENCODING_VERSION, EFFECT_ENCODING_VERSION
from .protocol import action_bucket, bucket, observation_parts


@dataclass(frozen=True)
class GameLog:
    initial_fen: str
    moves: tuple[int, ...]


class ObservationCache:
    """Exact reconstruction from full history, bounded by array payload bytes."""
    def __init__(self, capacity_bytes=64 << 20):
        if capacity_bytes < 0:
            raise ValueError("negative reconstruction cache budget")
        self.capacity, self.bytes, self.items = capacity_bytes, 0, OrderedDict()

    def get(self, game, ply, effects):
        # Identity hashing avoids re-hashing an entire game on every minibatch.
        key = (id(game), ply, effects)
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key][1]
        state = State(game.initial_fen)
        for move in game.moves[:ply]:
            state.play(move)
        row = observation_parts(state.observation_with_effects() if effects else state.observation())
        size = sum(a.nbytes for a in row if a is not None)
        for a in row:
            if a is not None:
                a.flags.writeable = False
        if size <= self.capacity:
            while self.items and self.bytes + size > self.capacity:
                _, (_, _, old_size) = self.items.popitem(last=False)
                self.bytes -= old_size
            # Retain game: its identity cannot be recycled while the key lives.
            self.items[key] = (game, row, size)
            self.bytes += size
        return row


def validate_targets(ids, policy, wdl, value_weight):
    n = len(ids)
    if n < 1 or len(np.unique(ids)) != n:
        raise ValueError("invalid replay action identity")
    if policy.shape != (n,) or not np.isfinite(policy).all() or (policy < 0).any() or not np.isclose(policy.sum(), 1, atol=1e-5):
        raise ValueError("invalid replay policy")
    if wdl.shape != (3,) or not np.isfinite(wdl).all() or (wdl < 0).any() or value_weight not in (0, 1) or not np.isclose(wdl.sum(), value_weight):
        raise ValueError("invalid replay outcome; truncations require zero targets")


@dataclass
class Sample:
    board: np.ndarray
    ids: np.ndarray
    actions: np.ndarray
    policy: np.ndarray
    wdl: np.ndarray
    value_weight: float
    initial_fen: str
    history: list[int]
    effects: np.ndarray | None = None
    game: GameLog | None = None
    ply: int | None = None
    provenance: dict = field(default_factory=dict)

    def observation(self, effects=False):
        if effects and self.effects is None:
            game, ply = game_reference(self)
            row = ObservationCache(0).get(game, ply, True)
            if not np.array_equal(row[1], self.ids):
                raise ValueError("reconstructed action identity changed")
            return row
        return self.board, self.ids, self.actions, self.effects if effects else None

    def validate(self):
        validate_targets(self.ids, self.policy, self.wdl, self.value_weight)
        n = len(self.ids)
        if self.board.shape != (32, 8, 8) or not np.isfinite(self.board).all():
            raise ValueError("invalid replay board")
        if self.actions.shape != (n, 6):
            raise ValueError("invalid replay action identity")
        if (self.actions < 0).any() or (self.actions >= np.array([64, 64, 3, 16, 7, 4])).any():
            raise ValueError("invalid replay action encoding")
        if self.game is not None and (self.ply is None or not 0 <= self.ply <= len(self.game.moves)):
            raise ValueError("invalid replay ply reference")


def game_reference(sample):
    if sample.game is not None:
        return sample.game, sample.ply
    return GameLog(sample.initial_fen, tuple(sample.history)), len(sample.history)


@dataclass
class CompactSample:
    game: GameLog
    ply: int
    ids: np.ndarray
    policy: np.ndarray
    wdl: np.ndarray
    value_weight: float
    cache: ObservationCache
    provenance: dict = field(default_factory=dict)

    @property
    def initial_fen(self): return self.game.initial_fen
    @property
    def history(self): return list(self.game.moves[:self.ply])
    @property
    def board(self): return self.observation()[0]
    @property
    def actions(self): return self.observation()[2]
    @property
    def effects(self): return self.observation(True)[3]

    def observation(self, effects=False):
        row = self.cache.get(self.game, self.ply, effects)
        if not np.array_equal(row[1], self.ids):
            raise ValueError("reconstructed action identity changed")
        return row

    def validate(self):
        if not 0 <= self.ply <= len(self.game.moves):
            raise ValueError("invalid replay ply reference")
        validate_targets(self.ids, self.policy, self.wdl, self.value_weight)


class Replay:
    def __init__(self, capacity=100_000, cache_bytes=64 << 20):
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity, self.samples = capacity, []
        self.cache = ObservationCache(cache_bytes)

    def extend(self, samples):
        for s in samples:
            s.validate()
            game, ply = game_reference(s)
            self.samples.append(CompactSample(game, ply, s.ids, s.policy, s.wdl, s.value_weight, self.cache, s.provenance))
        self.samples = self.samples[-self.capacity:]

    def batch(self, rng, count, *, effects=False):
        if not self.samples or count < 1:
            raise ValueError("cannot sample an empty replay or empty batch")
        selected = [self.samples[i] for i in rng.integers(len(self.samples), size=count)]
        width = action_bucket(max(len(s.ids) for s in selected))
        observations = [s.observation(effects) for s in selected]
        boards = np.stack([o[0] for o in observations]).astype(np.float32)
        actions = np.zeros((count, width, 6), np.int32)
        policy = np.zeros((count, width), np.float32)
        mask = np.zeros((count, width), np.float32)
        for i, sample in enumerate(selected):
            n = len(sample.ids)
            actions[i, :n], policy[i, :n], mask[i, :n] = observations[i][2], sample.policy, 1
        result = (boards, actions, policy, mask, np.stack([s.wdl for s in selected]), np.asarray([s.value_weight for s in selected], np.float32))
        if effects:
            tokens = np.zeros((count, bucket(max(len(o[3]) for o in observations), 16), 4), np.int32)
            for i, o in enumerate(observations): tokens[i, :len(o[3])] = o[3]
            return (*result, tokens)
        return result


def save_shard(directory, samples, metadata):
    if not samples:
        raise ValueError("cannot save an empty replay shard")
    for sample in samples:
        sample.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"replay-{uuid.uuid4().hex}.npz"
    games, indices, refs = [], {}, []
    for s in samples:
        game, ply = game_reference(s)
        if game not in indices:
            indices[game] = len(games)
            games.append({"fen": game.initial_fen, "moves": game.moves})
        refs.append((indices[game], ply))
    lengths = np.asarray([len(s.ids) for s in samples], np.int64)
    offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths)))
    info = {**metadata, "format": 2, "rules": RULES_VERSION, "encoding": ENCODING_VERSION,
            "effect_encoding": EFFECT_ENCODING_VERSION, "inputs": "exact-history-reconstruction"}
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, ids=np.concatenate([s.ids for s in samples]).astype(np.uint32),
            policies=np.concatenate([s.policy for s in samples]).astype(np.float32), offsets=offsets,
            wdl=np.stack([s.wdl for s in samples]), weights=np.asarray([s.value_weight for s in samples], np.float32),
            refs=np.asarray(refs, np.int64), games=np.asarray(json.dumps(games)), metadata=np.asarray(json.dumps(info)),
            provenance=np.asarray(json.dumps([s.provenance for s in samples])))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def load_shard(path):
    with np.load(path, allow_pickle=False) as d:
        info = json.loads(str(d["metadata"]))
        if info.get("format") not in (1, 2) or info.get("rules") != RULES_VERSION or info.get("encoding") != ENCODING_VERSION:
            raise ValueError("incompatible replay rules or format")
        if info["format"] == 2:
            if info.get("effect_encoding") != EFFECT_ENCODING_VERSION:
                raise ValueError("incompatible replay effect encoding")
            games = [GameLog(g["fen"], tuple(g["moves"])) for g in json.loads(str(d["games"]))]
            refs, offsets = d["refs"], d["offsets"]
            if refs.dtype.kind not in "iu" or offsets.dtype.kind not in "iu" or refs.ndim != 2 or refs.shape[1] != 2 or offsets.shape != (len(refs) + 1,) or offsets[0] != 0 or (np.diff(offsets) < 1).any() or offsets[-1] != len(d["ids"]) or d["ids"].ndim != 1 or d["policies"].shape != d["ids"].shape or d["wdl"].shape != (len(refs), 3) or d["weights"].shape != (len(refs),):
                raise ValueError("invalid compact replay layout")
            provenance = json.loads(str(d["provenance"])) if "provenance" in d else [{} for _ in refs]
            if len(provenance) != len(refs) or any(not isinstance(p, dict) for p in provenance):
                raise ValueError("invalid replay provenance")
            cache, samples = ObservationCache(), []
            for i, (g, ply) in enumerate(refs):
                if not 0 <= g < len(games): raise ValueError("invalid replay game reference")
                a, b = offsets[i:i+2]
                sample = CompactSample(games[g], int(ply), d["ids"][a:b].copy(), d["policies"][a:b].copy(),
                                       d["wdl"][i].copy(), float(d["weights"][i]), cache, provenance[i])
                sample.validate()
                samples.append(sample)
            return samples, info
        histories = json.loads(str(d["histories"]))
        samples = []
        for i, n in enumerate(d["lengths"]):
            if n > d["ids"].shape[1]:
                raise ValueError("replay legal-action length exceeds storage")
            policy = d["policies"][i, :n].copy()
            if n < 1 or not np.isfinite(policy).all() or (policy < 0).any() or not np.isclose(policy.sum(), 1, atol=1e-5):
                raise ValueError("invalid replay policy")
            sample = Sample(d["boards"][i].astype(np.float32), d["ids"][i, :n].copy(),
                d["actions"][i, :n].astype(np.int32), policy, d["wdl"][i].copy(), float(d["weights"][i]),
                histories[i]["fen"], histories[i]["moves"])
            sample.validate()
            samples.append(sample)
        return samples, info
