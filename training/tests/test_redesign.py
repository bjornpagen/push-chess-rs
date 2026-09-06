"""Regression specifications for the inference-first redesign."""
import json
import numpy as np
import pytest
from tinygrad import Tensor
from pushzero._native import State, SearchBatch, SearchRuntime, RULES_VERSION, ENCODING_VERSION
from pushzero.model import ModelConfig, Network, Predictor
from pushzero.protocol import pack_observations
from pushzero.learning import Learner, save_checkpoint, load_checkpoint
from pushzero.replay import Sample, Replay, GameLog, save_shard, load_shard
from pushzero.curriculum import RestartArchive
from pushzero.evaluation import paired_summary
from pushzero.selfplay import Trajectory
from pushzero.search import SearchResult
from pushzero.experiments import manifest


def example(state=None):
    state = State() if state is None else state
    board, ids, actions, effects = state.observation_with_effects()
    return Sample(board, ids, actions, np.full(len(ids), 1 / len(ids), np.float32),
                  np.array([0., 1., 0.], np.float32), 1., state.fen(), [], effects)


def test_cache_deduplicates_exact_inputs_and_revisions_invalidate():
    model = Network(ModelConfig(8, 1))
    obs = State().observation()
    with Predictor(model, 4, jit=False, cache_bytes=1 << 20) as p:
        first = p([obs, obs])
        assert p.metrics.evaluated_rows == 1 and p.metrics.duplicate_rows == 1
        second = p([obs])
        assert p.metrics.cache_hits == 1
        first[0][0][:] = 999
        np.testing.assert_array_equal(p([obs])[0][0], second[0][0])
        model.revision += 1
        p([obs])
        assert p.metrics.evaluated_rows == 2
        assert p.cache.bytes <= p.cache.capacity


def test_tail_packing_graph_budget_and_retained_results():
    obs = State().observation()
    with Predictor(Network(ModelConfig(8, 1)), 8, jit=False, max_graphs=2) as p:
        first = p([obs])[0][0]
        retained = first.copy()
        for n in (2, 3, 8): p([obs] * n)
        assert len(p.compiled) == len(p.staging) == 2
        assert p.metrics.evaluated_rows == 14
        assert p.metrics.submitted_rows == 15
        np.testing.assert_array_equal(first, retained)


def test_effect_schema_padding_and_action_permutation_equivariance():
    state = State()
    obs = state.observation_with_effects()
    boards, actions, lengths, effects = pack_observations([obs], True)
    model = Network(ModelConfig(8, 1, global_every=1, effect_channels=4))
    # Nonzero output head makes this a representation test, not uniform logits.
    model.policy.weight.assign(Tensor.ones_like(model.policy.weight)).realize()
    model.revision += 1
    with Predictor(model, 1, jit=False) as p:
        before, value = p.packed(boards, actions, lengths=lengths, effects=effects)
        n = int(lengths[0])
        shuffled = actions.copy()
        shuffled[0, :n] = actions[0, :n][::-1]
        tokens = effects.copy()
        active = tokens[..., 0] != 0
        tokens[..., 0][active] = n + 1 - tokens[..., 0][active]
        after, other = p.packed(boards, shuffled, lengths=lengths, effects=tokens)
        np.testing.assert_allclose(after[0, :n], before[0, :n][::-1], atol=1e-5)
        np.testing.assert_array_equal(value, other)
        with pytest.raises(ValueError): p.packed(boards, actions, lengths=lengths)
        bad = effects.copy()
        bad[0, 0, 0] = n + 1
        with pytest.raises(ValueError): p.packed(boards, actions, lengths=lengths, effects=bad)


def test_native_advance_owns_output_and_rejects_bad_reply_atomically():
    state = State()
    batch = SearchBatch([state], np.zeros((1, 128), np.float32), 4, 4, effects=True)
    request_id, boards, actions, lengths, effects = batch.advance()
    saved = [a.copy() for a in (boards, actions, lengths, effects)]
    logits = np.zeros(actions.shape[:2], np.float32)
    with pytest.raises(ValueError): batch.advance(request_id + 1, logits, np.zeros(1, np.float32))
    with pytest.raises(ValueError): batch.advance(request_id, logits, np.array([np.nan], np.float32))
    assert batch.advance(request_id, logits, np.zeros(1, np.float32), stop=True) is None
    assert batch.finish()[0][2].sum() == 0
    del batch
    for a, b in zip((boards, actions, lengths, effects), saved): np.testing.assert_array_equal(a, b)


def test_pool_closes_with_unanswered_batch_and_is_idempotent():
    state = State()
    runtime = SearchRuntime(2, 0)
    runtime.start([state, state], np.zeros((2, 128), np.float32), 8, 4)
    request = runtime.advance()
    assert request is not None
    runtime.close()
    runtime.close()
    with pytest.raises(ValueError): runtime.start([state], np.zeros((1, 128), np.float32), 8, 4)


def test_compact_shards_share_game_logs_and_reconstruct_actual_history(tmp_path):
    state = State("7k/8/8/8/8/8/8/K7 w - - 0 1")
    initial, moves, samples = state.fen(), [], []
    for from_sq, to_sq in [(0, 1), (63, 62), (1, 0), (62, 63), (0, 1)]:
        s = example(state)
        s.initial_fen, s.history, s.ply = initial, list(moves), len(moves)
        samples.append(s)
        move = next(int(m) for m, a in zip(*state.observation()[1:])
                    if int(a[0]) == (from_sq ^ (56 if state.turn() else 0)) and int(a[1]) == (to_sq ^ (56 if state.turn() else 0)))
        state.play(move)
        moves.append(move)
    game = GameLog(initial, tuple(moves))
    for s in samples: s.game = game
    path = save_shard(tmp_path, samples, {"model_version": "test"})
    with np.load(path, allow_pickle=False) as d:
        assert "boards" not in d and "actions" not in d
        assert len(json.loads(str(d["games"]))) == 1
    restored, info = load_shard(path)
    assert info["format"] == 2 and info["model_version"] == "test"
    for a, b in zip(samples, restored):
        np.testing.assert_array_equal(a.board, b.board)
        np.testing.assert_array_equal(a.actions, b.actions)
        np.testing.assert_array_equal(a.effects, b.effects)
        assert list(a.history) == b.history
    replay = Replay(3, cache_bytes=20000)
    replay.extend(restored)
    batch = replay.batch(np.random.default_rng(0), 4, effects=True)
    assert len(batch) == 7 and len(replay.samples) == 3
    assert replay.cache.bytes <= replay.cache.capacity


def test_legacy_shard_reader_is_retained(tmp_path):
    s = example()
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, boards=s.board[None].astype(np.float16), ids=s.ids[None], actions=s.actions[None],
        policies=s.policy[None], lengths=np.array([len(s.ids)]), wdl=s.wdl[None], weights=np.array([1.]),
        histories=np.asarray(json.dumps([{"fen": s.initial_fen, "moves": []}])),
        metadata=np.asarray(json.dumps({"format": 1, "rules": RULES_VERSION, "encoding": ENCODING_VERSION})))
    loaded, info = load_shard(path)
    assert info["format"] == 1
    np.testing.assert_array_equal(loaded[0].ids, s.ids)


def test_effect_learning_and_ema_checkpoint_are_separate(tmp_path):
    model = Network(ModelConfig(8, 1, 1, 4))
    learner = Learner(model, jit=False, ema_decay=.9)
    replay = Replay()
    replay.extend([example()])
    batch = replay.batch(np.random.default_rng(0), 2, effects=True)
    for _ in range(2): assert np.isfinite(learner.train(batch)["loss"])
    path = tmp_path / "effects.safetensors"
    save_checkpoint(path, learner, {})
    restored, _ = load_checkpoint(path, training=True, jit=False)
    averaged, _ = load_checkpoint(path, weights="ema")
    assert averaged.config == model.config and restored.ema_decay == .9
    for key in learner.ema: np.testing.assert_array_equal(learner.ema[key].numpy(), restored.ema[key].numpy())
    a, b = learner.train(batch), restored.train(batch)
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-5)
    with pytest.raises(ValueError): load_checkpoint(path, training=True, weights="ema")


def test_restarts_keep_prefix_and_truncations_supply_no_priority():
    state = State()
    initial = state.fen()
    move = state.legal_ids()[0]
    state.play(move)
    archive = RestartArchive(2)
    archive.add(GameLog(initial, (move,)), 1, .7)
    restarted, game = archive.sample(np.random.default_rng(0))
    np.testing.assert_array_equal(restarted.observation()[0], state.observation()[0])
    trajectory = Trajectory(restarted, True, game.initial_fen, list(game.moves), "restart")
    obs = restarted.observation()
    result = SearchResult(int(obs[1][0]), obs, np.full(len(obs[1]), 1 / len(obs[1]), np.float32),
                          np.zeros(len(obs[1]), np.uint32), 1)
    assert trajectory.advance(result, True, 1, False)
    samples, record = trajectory.finish(archive)
    assert record["truncated"] and record["start_ply"] == 1
    assert len(archive) == 1 and samples[0].value_weight == 0
    assert list(samples[0].history) == [move]


def test_paired_intervals_and_plan_are_explicit():
    unresolved = [{"pair": 0, "result": None}, {"pair": 0, "result": None}]
    assert paired_summary(unresolved)["score_bounds"] == [0., 1.]
    assert paired_summary(unresolved)["paired_confidence_bounds"] == [0., 1.]
    with pytest.raises(ValueError): paired_summary(unresolved[:1])
    plan = manifest()
    assert plan["status"] == "planned-not-run" and not plan["promotion"]["automatic"]
    assert len(plan["architecture_arms"]) == 5
