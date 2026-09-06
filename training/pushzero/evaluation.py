"""Paired held-out matches. No evaluation result enters training replay."""
from contextlib import ExitStack
import math
import time
from pathlib import Path
import numpy as np
from ._native import State, Opponent
from .learning import load_checkpoint, resolve_checkpoint
from .model import Predictor
from .search import search


def paired_summary(records, alpha=.05):
    """Distribution-free opening-pair bounds; unresolved games stay intervals.

    Hoeffding treats each opening pair as one bounded observation. The interval
    is intentionally conservative, and assumes independently sampled openings,
    fixed contestants, and a predeclared sample size (not repeated peeking).
    """
    if not records or not 0 < alpha < 1: raise ValueError("invalid score sample")
    pairs = {}
    for r in records:
        score = None if r["result"] is None else (1 + r["result"]) / 2
        pairs.setdefault(r["pair"], []).append((0., 1.) if score is None else (score, score))
    if any(len(rows) != 2 for rows in pairs.values()): raise ValueError("incomplete opening pair")
    bounds = np.asarray([[sum(x[k] for x in rows) / 2 for k in (0, 1)] for rows in pairs.values()])
    lo, hi = bounds.mean(axis=0)
    radius = math.sqrt(math.log(2 / alpha) / (2 * len(pairs)))
    return {"score_bounds": [float(lo), float(hi)],
            "paired_confidence_bounds": [max(0., float(lo) - radius), min(1., float(hi) + radius)],
            "confidence": 1 - alpha, "confidence_method": "opening-pair Hoeffding; fixed sample size"}


def evaluate(checkpoint, opponent="cataclysm", pairs=8, simulations=64, opponent_ms=50,
             opponent_nodes=0, max_plies=512, seed=918273, opening_plies=6, progress=None,
             *, move_ms=None, weights="raw", opponent_weights="raw"):
    if pairs < 1 or simulations < 1 or max_plies < 1 or opening_plies < 0:
        raise ValueError("invalid evaluation budget")
    if move_ms is not None and (not math.isfinite(move_ms) or not 1 <= move_ms <= 3_600_000):
        raise ValueError("wall-time budget must be 1..3600000 ms")
    if move_ms is not None and opponent_nodes:
        raise ValueError("equal-time evaluation cannot also limit only the opponent's nodes")
    checkpoint = resolve_checkpoint(checkpoint)
    model, _ = load_checkpoint(checkpoint, weights=weights)
    neural_opponent = Path(opponent).is_dir() or str(opponent).endswith((".safetensors", ".json"))
    if neural_opponent: opponent = resolve_checkpoint(opponent)
    native = None if opponent == "random" or neural_opponent else Opponent(opponent)
    with ExitStack() as stack:
        predictor = stack.enter_context(Predictor(model, batch_size=1))
        other = stack.enter_context(Predictor(load_checkpoint(opponent, weights=opponent_weights)[0], 1)) if neural_opponent else None
        rng = np.random.default_rng(seed)
        # Warm the initial shape; unfamiliar later shapes may still compile.
        for p in (predictor, other):
            if p is not None:
                state = State()
                obs = state.observation_with_effects() if p.with_effects else state.observation()
                for _ in range(3): p([obs])
        records, durations = [], [[], []]
        for pair in range(pairs):
            opening, history = State(), []
            for _ in range(opening_plies):
                move = int(rng.choice(opening.legal_ids()))
                opening.play(move)
                history.append(move)
                if opening.outcome() is not None: break
            if opening.outcome() is not None: opening, history = State(), []
            for color in (0, 1):
                state, moves = opening.copy(), []
                if native is not None: native = Opponent(opponent)
                while state.outcome() is None and len(moves) < max_plies:
                    contestant = 0 if state.turn() == color else 1
                    start = time.monotonic()
                    if contestant == 0 or other is not None:
                        p = predictor if contestant == 0 else other
                        # Equal-time mode gets a high safety cap, not the small
                        # diagnostic cap that could stop one player prematurely.
                        cap = simulations if move_ms is None else 1_000_000
                        deadline = float("inf") if move_ms is None else start + move_ms / 1000
                        move = search([state], p, rng, simulations=cap, explore=False, deadline=deadline,
                                      max_nodes=16384 if move_ms is None else 1_000_001)[0].move
                    elif native is not None:
                        move = native.choose(state, time_ms=opponent_ms if move_ms is None else int(move_ms), nodes=opponent_nodes)
                    else:
                        move = int(rng.choice(state.legal_ids()))
                    durations[contestant].append(1000 * (time.monotonic() - start))
                    state.play(move)
                    moves.append(move)
                white = state.outcome()
                relative = None if white is None else white if color == 0 else -white
                records.append({"pair": pair, "model_color": color, "opening_moves": history, "moves": moves,
                                "result": relative, "truncated": relative is None, "final_fen": state.fen()})
                if progress: progress({"event": "evaluation_game", "pair": pair, "color": color, "result": relative, "plies": len(moves)})
        wins, draws, losses = [sum(r["result"] == v for r in records) for v in (1, 0, -1)]
        return {"checkpoint": str(checkpoint), "opponent": str(opponent), "seed": seed, "pairs": pairs,
                "weights": weights, "opponent_weights": opponent_weights,
                "simulations": simulations if move_ms is None else None, "move_ms": move_ms,
                "opponent_ms": opponent_ms if move_ms is None else move_ms, "opponent_nodes": opponent_nodes,
                "wins": wins, "draws": draws, "losses": losses, "truncated": len(records) - wins - draws - losses,
                **paired_summary(records),
                "mean_move_ms": [float(np.mean(d)) if d else 0. for d in durations],
                "p95_move_ms": [float(np.quantile(d, .95)) if d else 0. for d in durations],
                "deadline_overruns": [sum(t > move_ms for t in d) if move_ms is not None else 0 for d in durations],
                "budget_note": ("Same requested wall-time per move; cooperative neural-round deadlines, not hard real-time. "
                                "Root evaluation and new shape compilation may overrun; actual times reported."
                                if move_ms is not None else "Equal simulation caps for checkpoint opponents; native budgets separate. Not time-equalized."),
                "games": records}
