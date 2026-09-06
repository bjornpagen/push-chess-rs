"""Batched self-play, exact terminal rewards, and optional rules-only curriculum."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
import uuid
import numpy as np
from ._native import State, RULES_VERSION, ENCODING_VERSION
from .replay import Sample, CompactSample, ObservationCache, GameLog, game_reference, validate_targets
from .protocol import observation_parts
from .search import search, SearchDriver


def reduced_start(rng):
    """Random sparse positions, not engine demonstrations or strategic labels."""
    for _ in range(1000):
        board = [None] * 64
        squares = rng.choice(64, size=6, replace=False)
        # Diverse sparse endings: independently sampled colors/types, both kings.
        board[squares[0]], board[squares[1]] = "K", "k"
        pieces = list("QRBNqrbn")
        for sq in squares[2:2 + int(rng.integers(1, 5))]:
            board[sq] = str(rng.choice(pieces))
        ranks = []
        for rank in range(7, -1, -1):
            line, empty = "", 0
            for piece in board[rank * 8:(rank + 1) * 8]:
                if piece is None:
                    empty += 1
                else:
                    line += (str(empty) if empty else "") + piece
                    empty = 0
            ranks.append(line + (str(empty) if empty else ""))
        fen = "/".join(ranks) + (" w" if rng.random() < .5 else " b") + " - - 0 1"
        try:
            state = State(fen)
            if state.outcome() is None:
                return state
        except ValueError:
            pass
    raise RuntimeError("could not generate a valid sparse position")


@dataclass
class PendingTarget:
    ply: int
    ids: np.ndarray
    policy: np.ndarray
    turn: int
    predicted: float
    provenance: dict


@dataclass
class Trajectory:
    state: State
    curriculum: bool
    initial: str | None = None
    moves: list[int] = field(default_factory=list)
    source: str = "standard"
    examples: list[PendingTarget] = field(default_factory=list)
    start_ply: int = field(init=False)

    def __post_init__(self):
        if self.initial is None: self.initial = self.state.fen()
        self.start_ply = len(self.moves)

    def advance(self, result, full, max_plies, halted, provenance=None):
        turn = self.state.turn()
        self.state.play(result.move)
        self.moves.append(result.move)
        outcome = self.state.outcome()
        finished = outcome is not None or len(self.moves) - self.start_ply >= max_plies or halted
        # Fast moves allocate no replay sample/history unless they are the only
        # fallback for a finished game. Root buffers already have safe ownership.
        if full or (finished and not self.examples):
            origin = {**(provenance or {}), "board_density": int(result.observation[0][:12].sum())}
            self.examples.append(PendingTarget(len(self.moves) - 1, result.observation[1].copy(),
                result.policy.copy(), turn, result.root_value, origin))
        return finished

    def finish(self, archive=None):
        outcome = self.state.outcome()
        game = GameLog(self.initial, tuple(self.moves))
        errors, samples = [], []
        cache = ObservationCache(0)
        for target in self.examples:
            wdl = np.zeros(3, np.float32)
            if outcome is not None:
                relative = outcome if target.turn == 0 else -outcome
                wdl[0 if relative > 0 else 2 if relative < 0 else 1] = 1
                errors.append((relative - target.predicted) ** 2)
                if archive is not None:
                    # Observed prediction error is only a sampling priority,
                    # never a replacement value target or a claimed true regret.
                    archive.add(game, target.ply, min(2.0, abs(relative - target.predicted)))
            samples.append(CompactSample(game, target.ply, target.ids, target.policy, wdl,
                float(outcome is not None), cache, target.provenance))
        record = {"initial_fen": self.initial, "moves": self.moves, "white_outcome": outcome,
                  "truncated": outcome is None, "curriculum": self.curriculum,
                  "source": self.source, "start_ply": self.start_ply,
                  "value_mse": float(np.mean(errors)) if errors else None}
        return samples, record


class RollingCollector:
    """Fixed actor slots; collection quotas end moves, never unfinished games."""
    def __init__(self, predictor, rng, actors=32, simulations=64, fast_simulations=16,
                 full_fraction=.25, max_plies=512, curriculum=.25, *, archive=None,
                 restart_fraction=0., fast_explore=True, max_nodes=16384, restored=None):
        if min(actors, simulations, fast_simulations, max_plies) < 1 or not 0 < full_fraction <= 1:
            raise ValueError("invalid self-play capacity/search budget")
        if not 0 <= curriculum <= 1 or not 0 <= restart_fraction <= 1 - curriculum:
            raise ValueError("invalid start mixture")
        self.predictor, self.rng, self.archive = predictor, rng, archive
        self.simulations, self.fast_simulations, self.full_fraction = simulations, fast_simulations, full_fraction
        self.max_plies, self.curriculum, self.restart_fraction = max_plies, curriculum, restart_fraction
        self.fast_explore, self.max_nodes = fast_explore, max_nodes
        self.capacity = actors
        self.moves = self.completed = self.started = 0
        self.active = False
        self.slots = list(restored) if restored is not None else []
        if self.slots and len(self.slots) != actors:
            raise ValueError("restored actor population differs; cannot silently discard games")

    def new_game(self):
        choice = self.rng.random()
        restart = self.archive.sample(self.rng) if self.archive is not None and choice < self.restart_fraction else None
        self.started += 1
        if restart is not None:
            state, game = restart
            return Trajectory(state, True, game.initial_fen, list(game.moves), "restart")
        sparse = self.restart_fraction <= choice < self.restart_fraction + self.curriculum
        return Trajectory(reduced_start(self.rng) if sparse else State(), bool(sparse), source="sparse" if sparse else "standard")

    def statistics(self):
        return {"actors": len(self.slots), "capacity": self.capacity, "moves": self.moves, "completed_total": self.completed,
                "pending_targets": sum(len(g.examples) for g in self.slots),
                "pending_target_bytes": sum(t.ids.nbytes + t.policy.nbytes for g in self.slots for t in g.examples)}

    def collect(self, games, *, deadline=float("inf"), stop=lambda: False, progress=None,
                policy_steps=0, iteration=0, move_limit=None):
        if games < 1 or (move_limit is not None and move_limit < 1):
            raise ValueError("collection quotas must be positive")
        if self.active: raise RuntimeError("collector already active")
        if stop() or time.monotonic() >= deadline: return [], []
        if not self.slots: self.slots = [self.new_game() for _ in range(self.capacity)]
        samples, records, full_search = [], [], {}
        begin_moves = self.moves
        next_progress = time.monotonic() + 15
        driver = SearchDriver(self.predictor, len(self.slots))
        self.active = True
        def admitting():
            return (len(records) < games and (move_limit is None or self.moves - begin_moves < move_limit)
                    and time.monotonic() < deadline and not stop())
        def launch(lane):
            start = lane * driver.group_size
            full = self.rng.random() < self.full_fraction
            full_search[lane] = full
            driver.start(lane, [g.state for g in self.slots[start:start + driver.group_size]], self.rng,
                         self.simulations if full else self.fast_simulations,
                         explore=full or self.fast_explore, max_nodes=self.max_nodes)
        try:
            if admitting():
                for lane in range((len(self.slots) + driver.group_size - 1) // driver.group_size): launch(lane)
            while driver.roots:
                finished = driver.poll()
                for lane, results in finished:
                    start = lane * driver.group_size
                    for slot, result in enumerate(results, start):
                        game = self.slots[slot]
                        self.moves += 1
                        if game.advance(result, full_search[lane], self.max_plies, False,
                                        {"source": "selfplay", "iteration": iteration, "policy_steps": policy_steps}):
                            examples, record = game.finish(self.archive)
                            samples.extend(examples)
                            records.append(record)
                            self.completed += 1
                            self.slots[slot] = self.new_game()
                # Decide after consuming the whole completion packet. No batch
                # tail is chased to game termination, even at a learning boundary.
                if admitting():
                    for lane, _ in finished: launch(lane)
                if progress is not None and time.monotonic() >= next_progress:
                    progress({"event": "selfplay_progress", "completed": len(records),
                              "live": len(self.slots), "positions": self.predictor.positions,
                              "samples": len(samples), **self.statistics()})
                    next_progress = time.monotonic() + 15
            driver.finish()
            return samples, records
        except BaseException:
            driver.abort()
            raise
        finally:
            self.active = False

    def save(self, directory):
        if self.active: raise RuntimeError("checkpoint requires a move boundary")
        directory = Path(directory)
        path = directory / f"actors-{uuid.uuid4().hex}.npz"
        games = [{"fen": g.initial, "moves": g.moves, "start_ply": g.start_ply,
                  "source": g.source, "curriculum": g.curriculum} for g in self.slots]
        targets = [(i, t) for i, g in enumerate(self.slots) for t in g.examples]
        offsets = np.cumsum([0] + [len(t.ids) for _, t in targets], dtype=np.int64)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, metadata=np.asarray(json.dumps({"format": 1, "rules": RULES_VERSION,
                "encoding": ENCODING_VERSION, "stats": self.statistics(), "started": self.started})),
                games=np.asarray(json.dumps(games)), offsets=offsets,
                ids=np.concatenate([t.ids for _,t in targets]) if targets else np.empty(0, np.uint32),
                policies=np.concatenate([t.policy for _,t in targets]) if targets else np.empty(0, np.float32),
                refs=np.asarray([(i,t.ply,t.turn) for i,t in targets], np.int64).reshape(-1, 3),
                predictions=np.asarray([t.predicted for _,t in targets], np.float32),
                provenance=np.asarray(json.dumps([t.provenance for _,t in targets])))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return path

    @staticmethod
    def restore(path):
        with np.load(path, allow_pickle=False) as d:
            metadata = json.loads(str(d["metadata"]))
            if (metadata.get("format"), metadata.get("rules"), metadata.get("encoding")) != (1, RULES_VERSION, ENCODING_VERSION):
                raise ValueError("incompatible actor checkpoint")
            slots = []
            for g in json.loads(str(d["games"])):
                state = State(g["fen"])
                for move in g["moves"]: state.play(move)
                if state.outcome() is not None or not 0 <= g["start_ply"] <= len(g["moves"]):
                    raise ValueError("invalid unfinished game")
                game = Trajectory(state, g["curriculum"], g["fen"], list(g["moves"]), g["source"])
                game.start_ply = g["start_ply"]
                slots.append(game)
            refs, offsets = d["refs"], d["offsets"]
            provenance = json.loads(str(d["provenance"]))
            if (refs.ndim != 2 or refs.shape[1] != 3 or refs.dtype.kind not in "iu"
                or offsets.shape != (len(refs) + 1,) or offsets.dtype.kind not in "iu" or offsets[0] != 0
                or (np.diff(offsets) < 1).any() or offsets[-1] != len(d["ids"])
                or d["ids"].ndim != 1 or d["policies"].shape != d["ids"].shape
                or d["predictions"].shape != (len(refs),) or len(provenance) != len(refs)):
                raise ValueError("invalid actor target layout")
            for i, (slot, ply, turn) in enumerate(refs):
                if not 0 <= slot < len(slots) or not slots[slot].start_ply <= ply < len(slots[slot].moves) or turn not in (0, 1):
                    raise ValueError("invalid actor target reference")
                a, b = offsets[i:i + 2]
                ids, policy, predicted = d["ids"][a:b].copy(), d["policies"][a:b].copy(), float(d["predictions"][i])
                validate_targets(ids, policy, np.zeros(3, np.float32), 0)
                if not np.isfinite(predicted) or abs(predicted) > 1.00001 or not isinstance(provenance[i], dict):
                    raise ValueError("invalid pending prediction/provenance")
                targets = slots[slot].examples
                if targets and ply <= targets[-1].ply: raise ValueError("actor targets must have increasing plies")
                targets.append(PendingTarget(int(ply), ids, policy, int(turn), predicted, provenance[i]))
            # Reconstruct each history once, proving identity and perspective at
            # every referenced position before accepting any resumed targets.
            for game in slots:
                state, targets = State(game.initial), {t.ply: t for t in game.examples}
                for ply, move in enumerate(game.moves):
                    if ply in targets:
                        target = targets[ply]
                        if state.legal_ids() != target.ids.tolist() or state.turn() != target.turn:
                            raise ValueError("actor target identity changed")
                    state.play(move)
            return slots, metadata


def collect(predictor, rng, games=64, actors=32, simulations=64, fast_simulations=16,
            full_fraction=.25, max_plies=512, curriculum=.25, deadline=float("inf"), stop=lambda: False, progress=None,
            *, archive=None, restart_fraction=0.0, fast_explore=True, max_nodes=16384):
    # Explicit one-shot convenience API. Training owns a persistent collector.
    collector = RollingCollector(predictor, rng, actors, simulations, fast_simulations,
        full_fraction, max_plies, curriculum, archive=archive, restart_fraction=restart_fraction,
        fast_explore=fast_explore, max_nodes=max_nodes)
    return collector.collect(games, deadline=deadline, stop=stop, progress=progress)


def reanalyse(samples, predictor, rng, simulations, *, deadline=float("inf"), stop=lambda: False, max_nodes=16384):
    states = []
    for sample in samples:
        game, ply = game_reference(sample)
        state = State(game.initial_fen)
        for move in game.moves[:ply]:
            state.play(move)
        if state.legal_ids() != sample.ids.tolist():
            raise ValueError("replay action identity changed")
        states.append(state)
    results = search(states, predictor, rng, simulations=simulations, explore=False, deadline=deadline, stop=stop, max_nodes=max_nodes)
    # Original outcome and its validity flag remain unchanged.
    return [Sample(r.observation[0], s.ids, r.observation[2], r.policy, s.wdl, s.value_weight, s.initial_fen, [],
                   effects=observation_parts(r.observation)[3], game=game_reference(s)[0], ply=game_reference(s)[1],
                   provenance={"source": "reanalysis", "outcome_source": s.provenance.get("outcome_source", s.provenance),
                               "board_density": int(r.observation[0][:12].sum()),
                               "policy_revision": getattr(getattr(predictor, "model", None), "revision", None)})
            for s, r in zip(samples, results, strict=True)]
