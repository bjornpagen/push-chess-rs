"""Typed complete-game pages from bumbledb, the only durable game source."""
import numpy as np
from ._native import CorpusReader, State, RULES_VERSION
from .replay import GameLog, CompactSample, ObservationCache, Replay


def games(path, split="train", page_size=8, *, nnue=False, runs=None):
    reader = CorpusReader(str(path))
    try:
        cursor = (0, 0)
        while True:
            batch, cursor, done = reader.page(split, cursor, page_size, nnue=nnue, runs=runs)
            yield from batch
            if done: break
    finally:
        reader.close()


def samples(game, cache=None):
    # The full policy model has no rules input. Do not mix historical policy
    # targets into a current-rules checkpoint; NNUE has a separate audited path.
    if game["rules"] != RULES_VERSION:
        raise ValueError("policy training requires current-rules games")
    cache = ObservationCache(0) if cache is None else cache
    log = GameLog(game["initial_fen"], tuple(map(int, game["moves"])))
    state, result = State(log.initial_fen), []
    for ply, (move, row) in enumerate(zip(log.moves, game["analysis"], strict=True)):
        # Native columns: side, score, analysis-present, complete, nodes, depth.
        side, score, analysed, complete, nodes, depth = map(int, row)
        if side != state.turn(): raise ValueError("corpus perspective mismatch")
        if analysed and complete:
            ids = np.asarray(state.legal_ids(), np.uint32)
            selected = np.flatnonzero(ids == move)
            if len(selected) != 1: raise ValueError("illegal teacher action")
            policy = np.zeros(len(ids), np.float32)
            policy[selected[0]] = 1
            wdl = np.zeros(3, np.float32)
            outcome = game["white_value"]
            if outcome is not None:
                relative = outcome if state.turn() == 0 else -outcome
                wdl[0 if relative > 0 else 2 if relative < 0 else 1] = 1
            teacher = game["white"] if state.turn() == 0 else game["black"]
            result.append(CompactSample(log, ply, ids, policy, wdl, float(outcome is not None), cache,
                {"source":"bumbledb", "run":game["run_id"], "game":game["game_index"], "teacher":teacher,
                 "opening":game["opening_key"], "trajectory":game["trajectory_key"],
                 "score_stm":score, "nodes":nodes, "depth":depth}))
        state.play(move)
    if state.fen() != game["final_fen"] or state.outcome() != game["white_value"]:
        raise ValueError("corpus outcome/history mismatch")
    return result


def replay(path, *, capacity=100_000, max_games=10_000, seed=1):
    if capacity < 1 or max_games < 1: raise ValueError("positive corpus limits required")
    rng, retained, seen, count = np.random.default_rng(seed), [], 0, 0
    stream, trajectories = games(path, "train"), set()
    try:
        for game in stream:
            count += 1
            if game["trajectory_key"] in trajectories:
                if count >= max_games: break
                continue
            trajectories.add(game["trajectory_key"])
            for sample in samples(game):
                seen += 1
                if len(retained) < capacity: retained.append(sample)
                else:
                    index = int(rng.integers(seen))
                    if index < capacity: retained[index] = sample
            if count >= max_games: break
    finally:
        stream.close()
    if not retained: raise ValueError("no completed teacher searches in eligible training games")
    buffer = Replay(capacity)
    buffer.extend(retained)
    return buffer, {"source":"bumbledb", "path":str(path), "split":"train", "games":count,
                    "eligible_positions":seen, "retained_positions":len(retained), "seed":seed}
