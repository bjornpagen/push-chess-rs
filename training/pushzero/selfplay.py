"""Batched self-play, exact terminal rewards, and optional rules-only curriculum."""
from dataclasses import dataclass, field
import time
import numpy as np
from ._native import State
from .replay import Sample, GameLog, game_reference
from .protocol import observation_parts
from .search import search


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
class Trajectory:
    state: State
    curriculum: bool
    initial: str | None = None
    moves: list[int] = field(default_factory=list)
    source: str = "standard"
    examples: list[tuple[Sample, int, float]] = field(default_factory=list)
    start_ply: int = field(init=False)

    def __post_init__(self):
        if self.initial is None: self.initial = self.state.fen()
        self.start_ply = len(self.moves)

    def advance(self, result, full, max_plies, halted):
        turn = self.state.turn()
        self.state.play(result.move)
        self.moves.append(result.move)
        outcome = self.state.outcome()
        finished = outcome is not None or len(self.moves) - self.start_ply >= max_plies or halted
        # Fast moves allocate no replay sample/history unless they are the only
        # fallback for a finished game. Root buffers already have safe ownership.
        if full or (finished and not self.examples):
            board, ids, actions, effects = observation_parts(result.observation)
            sample = Sample(board, ids, actions, result.policy, np.zeros(3, np.float32), 0,
                            self.initial, [], effects=effects, ply=len(self.moves) - 1)
            self.examples.append((sample, turn, result.root_value))
        return finished

    def finish(self, archive=None):
        outcome = self.state.outcome()
        game = GameLog(self.initial, tuple(self.moves))
        errors = []
        for sample, turn, predicted in self.examples:
            sample.game = game
            sample.history = HistoryPrefix(game.moves, sample.ply)
            if outcome is not None:
                relative = outcome if turn == 0 else -outcome
                sample.wdl[0 if relative > 0 else 2 if relative < 0 else 1] = 1
                sample.value_weight = 1.0
                errors.append((relative - predicted) ** 2)
                if archive is not None:
                    # Observed prediction error is only a sampling priority,
                    # never a replacement value target or a claimed true regret.
                    archive.add(game, sample.ply, min(2.0, abs(relative - predicted)))
        record = {"initial_fen": self.initial, "moves": self.moves, "white_outcome": outcome,
                  "truncated": outcome is None, "curriculum": self.curriculum,
                  "source": self.source, "start_ply": self.start_ply,
                  "value_mse": float(np.mean(errors)) if errors else None}
        return [s for s, _, _ in self.examples], record


class HistoryPrefix:
    """Immutable view; examples share a complete game, not copied prefixes."""
    def __init__(self, moves, ply): self.moves, self.ply = moves, ply
    def __len__(self): return self.ply
    def __iter__(self):
        for i in range(self.ply): yield self.moves[i]
    def __getitem__(self, index): return self.moves[:self.ply][index]


def collect(predictor, rng, games=64, actors=32, simulations=64, fast_simulations=16,
            full_fraction=.25, max_plies=512, curriculum=.25, deadline=float("inf"), stop=lambda: False, progress=None,
            *, archive=None, restart_fraction=0.0, fast_explore=True, max_nodes=16384):
    if min(games, actors, simulations, fast_simulations, max_plies) < 1 or not 0 < full_fraction <= 1 or not 0 <= curriculum <= 1:
        raise ValueError("invalid self-play configuration")
    if not 0 <= restart_fraction <= 1 or curriculum + restart_fraction > 1:
        raise ValueError("start mixture fractions must sum to at most one")
    samples, game_records, live = [], [], []
    started = completed = 0
    next_progress = time.monotonic() + 15
    while completed < games:
        while len(live) < actors and started < games and time.monotonic() < deadline and not stop():
            choice = rng.random()
            restart = archive.sample(rng) if archive is not None and choice < restart_fraction else None
            if restart is not None:
                state, game = restart
                live.append(Trajectory(state, True, game.initial_fen, list(game.moves), "restart"))
            else:
                sparse = restart_fraction <= choice < restart_fraction + curriculum
                live.append(Trajectory(reduced_start(rng) if sparse else State(), bool(sparse), source="sparse" if sparse else "standard"))
            started += 1
        if not live:
            break
        # A shared random cap per batch keeps GPU batches full. Every game still
        # gets the requested marginal full-search probability at each ply.
        full = rng.random() < full_fraction
        budget = simulations if full else fast_simulations
        results = search([g.state for g in live], predictor, rng, simulations=budget, explore=full or fast_explore,
                         deadline=deadline, stop=stop, max_nodes=max_nodes)
        remaining = []
        for game, result in zip(live, results, strict=True):
            if game.advance(result, full, max_plies, time.monotonic() >= deadline or stop()):
                examples, record = game.finish(archive)
                samples.extend(examples)
                game_records.append(record)
                completed += 1
            else:
                remaining.append(game)
        live = remaining
        if progress is not None and time.monotonic() >= next_progress:
            progress({"event": "selfplay_progress", "completed": completed, "started": started,
                      "live": len(live), "positions": predictor.positions, "samples": len(samples)})
            next_progress = time.monotonic() + 15
    return samples, game_records


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
                               "policy_revision": getattr(getattr(predictor, "model", None), "revision", None)})
            for s, r in zip(samples, results, strict=True)]
