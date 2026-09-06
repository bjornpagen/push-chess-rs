"""Bounded self-generated restart archive; no engine demonstrations or labels."""
from collections import OrderedDict
from ._native import State
from .replay import GameLog


class RestartArchive:
    def __init__(self, capacity=2048, records=()):
        if capacity < 0: raise ValueError("negative restart capacity")
        self.capacity, self.items = capacity, OrderedDict()
        for r in records:
            self.add(GameLog(r["fen"], tuple(r["moves"])), r["ply"], r["priority"])

    def add(self, game, ply, priority):
        if not self.capacity: return
        if not 0 <= ply <= len(game.moves) or not 0 <= priority <= 2:
            raise ValueError("invalid restart reference or priority")
        prefix = game.moves[:ply]
        key = (game.initial_fen, prefix)
        self.items.pop(key, None)
        self.items[key] = (GameLog(game.initial_fen, prefix), float(priority))
        while len(self.items) > self.capacity: self.items.popitem(last=False)

    def sample(self, rng):
        if not self.items: return None
        values = list(self.items.values())
        weights = [0.05 + priority for _, priority in values]
        total = sum(weights)
        game, _ = values[int(rng.choice(len(values), p=[w / total for w in weights]))]
        state = State(game.initial_fen)
        for move in game.moves: state.play(move)
        if state.outcome() is not None: raise ValueError("terminal restart entry")
        return state, game

    def records(self):
        return [{"fen": g.initial_fen, "moves": g.moves, "ply": len(g.moves), "priority": p}
                for g, p in self.items.values()]

    def __len__(self): return len(self.items)
