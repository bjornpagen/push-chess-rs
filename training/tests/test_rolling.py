"""Population, lease, and checkpoint invariants without GPU compute."""
import json
import time
import numpy as np
import pytest
from pushzero._native import State, SearchRuntime
from pushzero.selfplay import RollingCollector, Trajectory
from test_pushzero import ZeroPredictor


def next_request(runtime):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        request, _ = runtime.poll()
        if request is not None: return request
    raise AssertionError("native request stalled")


def test_independent_leases_reverse_reply_order_and_preserve_buffers():
    runtime = SearchRuntime(2, 2, 1)
    state = State()
    try:
        for lane in range(2): runtime.start(lane, [state], np.zeros((1, 128), np.float32), 4, 4, effects=True)
        first, second = next_request(runtime), next_request(runtime)
        saved = [a.copy() for a in first[1:]]
        for req in (second, first):
            ident, boards, actions, lengths, effects = req
            logits, values = np.zeros(actions.shape[:2], np.float32), np.zeros(len(boards), np.float32)
            with pytest.raises(ValueError): runtime.submit(ident, logits, values + np.nan)
            runtime.submit(ident, logits, values)
            logits[:] = 999  # Native worker owns its reply, not this NumPy storage.
            with pytest.raises(ValueError): runtime.submit(ident, logits, values)
        deadline = time.monotonic() + 5
        done = []
        while not runtime.idle:
            assert time.monotonic() < deadline
            request, completed = runtime.poll()
            done.extend(completed)
            if request is not None:
                ident, boards, actions, _, _ = request
                runtime.submit(ident, np.zeros(actions.shape[:2], np.float32), np.zeros(len(boards), np.float32))
        assert sorted(lane for lane, _ in done) == [0, 1]
        for a, b in zip(first[1:], saved): np.testing.assert_array_equal(a, b)
    finally:
        runtime.close()


def test_lane_and_batch_capacity_are_explicit():
    runtime = SearchRuntime(1, 1, 1)
    state = State()
    try:
        with pytest.raises(ValueError): runtime.start(1, [state], np.zeros((1, 128), np.float32), 2, 2)
        with pytest.raises(ValueError): runtime.start(0, [state, state], np.zeros((2, 128), np.float32), 2, 2)
        runtime.start(0, [state], np.zeros((1, 128), np.float32), 2, 2)
        with pytest.raises(ValueError): runtime.start(0, [state], np.zeros((1, 128), np.float32), 2, 2)
    finally:
        runtime.close()


def test_one_lane_finishes_while_another_lease_is_held():
    runtime = SearchRuntime(2, 2, 1)
    state = State()
    try:
        for lane in range(2): runtime.start(lane, [state], np.zeros((1, 128), np.float32), 3, 3)
        first, held = next_request(runtime), next_request(runtime)
        request = first
        finished = []
        deadline = time.monotonic() + 5
        while not finished:
            assert time.monotonic() < deadline
            if request is not None:
                ident, boards, actions, _, _ = request
                assert ident != held[0]
                runtime.submit(ident, np.zeros(actions.shape[:2], np.float32), np.zeros(len(boards), np.float32))
            request, finished = runtime.poll()
        assert len(finished) == 1 and not runtime.idle
        # Closing does not need the deliberately unanswered second lease.
    finally:
        runtime.close()


def collector(predictor, restored=None, max_plies=32):
    predictor.group_size, predictor.batch_size = 2, 4
    return RollingCollector(predictor, np.random.default_rng(81), actors=8, simulations=2,
        fast_simulations=1, full_fraction=1, max_plies=max_plies, curriculum=0, restored=restored)


def test_quota_preserves_all_slots_and_unfinished_targets(tmp_path):
    p = ZeroPredictor()
    try:
        c = collector(p)
        samples, records = c.collect(10000, move_limit=8, policy_steps=7)
        assert not records and not samples
        assert len(c.slots) == 8 and all(g.moves for g in c.slots)
        assert c.statistics()["pending_targets"] >= 8
        original = [(g.initial, list(g.moves), list(g.examples)) for g in c.slots]
        path = c.save(tmp_path)
        restored, meta = RollingCollector.restore(path)
        assert meta["stats"] == c.statistics()
        resumed = collector(p, restored)
        for (fen, moves, targets), g in zip(original, resumed.slots):
            assert (g.initial, g.moves) == (fen, moves)
            np.testing.assert_array_equal(g.state.observation()[0], c.slots[resumed.slots.index(g)].state.observation()[0])
            for a, b in zip(targets, g.examples):
                assert a.ply == b.ply and a.turn == b.turn and a.provenance == b.provenance
                np.testing.assert_array_equal(a.policy, b.policy)
        resumed.collect(10000, move_limit=8, policy_steps=9)
        assert len(resumed.slots) == 8
        assert all(len(g.moves) > len(before[1]) for g, before in zip(resumed.slots, original))
        versions = {t.provenance["policy_steps"] for g in resumed.slots for t in g.examples}
        assert versions == {7, 9}
    finally:
        p.close()


def test_finished_games_are_replaced_without_waiting_for_long_games():
    p = ZeroPredictor()
    try:
        c = collector(p, max_plies=12)
        c.slots = [Trajectory(State(), False) for _ in range(8)]
        # One slot is one ply from a real draw. Other actors stay in long games.
        c.slots[0] = Trajectory(State("7k/8/8/8/8/8/8/K7 w - - 99 1"), False)
        untouched = c.slots[1:]
        samples, records = c.collect(1, policy_steps=13)
        assert len(records) >= 1 and records[0]["white_outcome"] == 0
        assert len(c.slots) == 8 and c.slots[1:] == untouched
        assert all(g.state.outcome() is None for g in c.slots)
        assert all(g.examples for g in c.slots[1:])
        assert all(s.value_weight == 1 and s.provenance["policy_steps"] == 13 for s in samples)
    finally:
        p.close()


def test_no_population_draws_rng_before_pending_learning_or_after_stop(tmp_path):
    p = ZeroPredictor()
    try:
        rng = np.random.default_rng(9)
        before = json.dumps(rng.bit_generator.state)
        c = RollingCollector(p, rng)
        c.save(tmp_path)
        assert c.collect(1, stop=lambda: True) == ([], [])
        assert json.dumps(rng.bit_generator.state) == before
        assert c.slots == []
    finally:
        p.close()


def test_actor_snapshot_rejects_corrupt_targets(tmp_path):
    p = ZeroPredictor()
    try:
        c = collector(p)
        c.collect(10000, move_limit=8)
        path = c.save(tmp_path)
        with np.load(path, allow_pickle=False) as d: data = dict(d)
        data["ids"] = data["ids"].copy()
        data["ids"][0] = 2**32 - 1
        bad = tmp_path / "bad.npz"
        np.savez_compressed(bad, **data)
        with pytest.raises(ValueError, match="identity"): RollingCollector.restore(bad)
        restored, _ = RollingCollector.restore(path)
        with pytest.raises(ValueError, match="population"):
            RollingCollector(p, np.random.default_rng(0), actors=9, restored=restored)
    finally:
        p.close()
