"""Pause/resume orchestration tests with no neural compute or self-play.

These replace every network/training operation with an in-memory test double.
"""
from contextlib import contextmanager
from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
from pushzero import State
from pushzero import run
from pushzero.replay import Sample


def test_interrupted_updates_resume_before_new_games(tmp_path, monkeypatch):
    control = SimpleNamespace(stop=False, interrupt_once=True)
    calls = []

    class FakeModel:
        parameter_count = 1
        def __init__(self, config):
            self.config = config

    class FakeLearner:
        def __init__(self, model, **_kwargs):
            self.model, self.steps = model, 0
        def train(self, _batch):
            calls.append("update")
            self.steps += 1
            if control.interrupt_once:
                control.stop, control.interrupt_once = True, False
            return {"loss": 0.0}

    class FakePredictor:
        positions = seconds = native_seconds = search_calls = 0
        def __init__(self, *_args, **_kwargs): pass

    def save(path, learner, metadata):
        info = {**metadata, "steps": learner.steps, "model": asdict(learner.model.config)}
        run.write_json(path, info)
        return info

    def load(path, **_kwargs):
        info = json.loads(path.read_text())
        learner = FakeLearner(FakeModel(run.ModelConfig(**info["model"])))
        learner.steps = info["steps"]
        return learner, info

    def collect(*_args, **_kwargs):
        calls.append("collect")
        state = State()
        board, ids, actions = state.observation()
        sample = Sample(board, ids, actions, np.full(len(ids), 1/len(ids), np.float32),
                        np.array([0,1,0], np.float32), 1, state.fen(), [])
        return [sample], [{"white_outcome": 0, "truncated": False}]

    @contextmanager
    def signals():
        yield lambda: control.stop

    for name, replacement in {"Network": FakeModel, "Learner": FakeLearner,
            "Predictor": FakePredictor, "save_checkpoint": save, "load_checkpoint": load,
            "collect": collect, "stop_signals": signals,
            "Tensor": SimpleNamespace(manual_seed=lambda _seed: None),
            "Device": SimpleNamespace(DEFAULT="TEST_DOUBLE")}.items():
        monkeypatch.setattr(run, name, replacement)
    config = run.TrainConfig(channels=8,blocks=1,actors=1,games=1,simulations=1,
                             fast_simulations=1,batch_size=1,reuse=3)
    first = run.train(tmp_path, config, minutes=1, iterations=1)
    assert first["steps"] == 1 and first["pending_updates"] == 2
    assert calls == ["collect", "update"]

    calls.clear()
    control.stop = False
    resumed = run.train(tmp_path, minutes=1, iterations=1, resume=True)
    assert calls == ["update", "update", "collect", "update", "update", "update"]
    assert resumed["steps"] == 6 and resumed["pending_updates"] == 0
    assert resumed["iteration"] == 2
