"""Exact shared features, signed integer arithmetic, QAT, and game balancing."""
import copy
import hashlib
import numpy as np
import pytest
from tinygrad import Context, Tensor
from pushzero._native import State, nnue_control, nnue_inputs, nnue_evaluate
from pushzero import nnue


def positions(count=32):
    result, state = [], State()
    for i in range(count):
        result.append(state.copy())
        if state.outcome() is not None: state = State()
        ids = state.legal_ids()
        state.play(int(ids[(i*13+7) % len(ids)]))
    return result


def metal_scores(model, ids, baseline, sides):
    with Context(TRAINING=0, TC=0):
        return model(Tensor(nnue.dense_features(ids)), Tensor(baseline.astype(np.float32)),
                     Tensor(sides.astype(np.float32))).numpy()


def test_control_features_integer_reference_and_metal_are_exact():
    states = positions(96)
    ids, baseline = nnue_inputs(states)
    side = np.array([state.turn() for state in states])
    assert ids.shape == (96, 2, 64) and ids.dtype == np.uint16
    assert baseline.shape == (96,) and baseline.dtype == np.int32
    # Rank reflection, relative color swap and the zero-padding identity.
    white = ids[:, 0]
    expected = np.where(white == 768, 768, ((white // 64 + 6) % 12) * 64 + ((white % 64) ^ 56))
    np.testing.assert_array_equal(ids[:, 1], expected)
    for row in ids:
        for perspective in row:
            real = perspective[perspective != 768]
            assert len(np.unique(real)) == len(real)
    model = nnue.Residual()
    assert model.export() == nnue_control()
    scores = nnue_evaluate(nnue_control(), states)
    np.testing.assert_array_equal(nnue.integer_scores(nnue_control(), ids, baseline, side), scores)
    np.testing.assert_array_equal(metal_scores(model, ids, baseline, side), scores)
    assert nnue_inputs([])[0].shape == (0, 2, 64)
    assert nnue_evaluate(nnue_control(), []).shape == (0,)
    with pytest.raises(ValueError): nnue_evaluate(b"bad", states)
    with pytest.raises(ValueError): nnue_inputs([states[0]] * 4097)


def test_full_i16_decoder_and_signed_division_agree_with_rust():
    states = positions(16)
    ids, baseline = nnue_inputs(states)
    side = np.array([state.turn() for state in states])
    rng = np.random.default_rng(33)
    data = rng.integers(-32768, 32768, (nnue.FEATURES+2)*nnue.WIDTH, dtype=np.int16).astype("<i2").tobytes()
    np.testing.assert_array_equal(nnue.integer_scores(data, ids, baseline, side), nnue_evaluate(data, states))
    with pytest.raises(ValueError, match="exact float32"): nnue.Residual(data)
    with pytest.raises(ValueError): nnue.dense_features(np.full((1,2,64), 769, np.uint16))
    with pytest.raises(ValueError): nnue.dense_features(np.zeros((1,2,63), np.uint16))


def test_half_integer_rounding_and_parameter_clipping_match_export():
    states = positions(12)
    ids, baseline = nnue_inputs(states)
    side = np.array([state.turn() for state in states])
    model, rng = nnue.Residual(), np.random.default_rng(32)
    for key, scale in nnue.SCALES.items():
        words = rng.integers(-120, 121, nnue.SHAPES[key]).astype(np.float32) + .5
        words.flat[:4] = [-40000.5, -2047.5, 2047.5, 40000.5]
        getattr(model, key).assign(Tensor(words / scale)).realize()
    data = model.export()
    np.testing.assert_array_equal(metal_scores(model, ids, baseline, side), nnue_evaluate(data, states))
    for key, scale in nnue.SCALES.items():
        limit = (-2047,2047) if key == "output" else (-32768,32767)
        expected = np.rint(np.clip(getattr(model,key).numpy()*scale, *limit))
        np.testing.assert_array_equal(nnue.decode(data)[key], expected)


@pytest.mark.parametrize("jit", [False, True])
def test_tiny_update_export_and_optimizer_resume_are_exact(tmp_path, jit):
    states = positions(4)
    ids, baseline = nnue_inputs(states)
    side = np.array([state.turn() for state in states], np.float32)
    batch = (nnue.dense_features(ids), baseline.astype(np.float32), side, np.array([1,0,1,0], np.float32))
    learner = nnue.Learner(jit=jit)
    before = learner.model.features.numpy().copy()
    for _ in range(3):
        metrics = learner.train(batch)
        assert np.isfinite(metrics["loss"]) and metrics["gradient_norm"] > 0
    assert not np.array_equal(before, learner.model.features.numpy())
    encoded = learner.model.export()
    expected = nnue_evaluate(encoded, states)
    np.testing.assert_array_equal(metal_scores(learner.model, ids, baseline, side), expected)
    path = tmp_path / "candidate.safetensors"
    nnue.save(path, learner, {"smoke_test": True})
    restored, info = nnue.load(path, training=True, jit=jit)
    assert info["steps"] == 3 and restored.model.export() == encoded
    a, b = learner.train(batch), restored.train(batch)
    assert a == b
    for key in nnue.SCALES:
        np.testing.assert_array_equal(getattr(learner.model,key).numpy(), getattr(restored.model,key).numpy())
    exported = tmp_path / "candidate.bin"
    report = nnue.export(path, exported)
    assert exported.read_bytes() == encoded
    assert report["sha256"] == hashlib.sha256(encoded).hexdigest()
    with pytest.raises(FileExistsError): nnue.save(path, learner, {})
    with pytest.raises(FileExistsError): nnue.export(path, exported)
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".bin", ".safetensors"]


def fake_game(index=0, *, length=6, outcome=1, run=2):
    ids, baseline = nnue_inputs(positions(length))
    analysis = np.zeros((length,6), np.int64)
    analysis[:,0] = np.arange(length) % 2
    analysis[:,2:4] = 1
    return {"run_id":run, "game_index":index, "white":"cataclysm", "black":"astra", "binary":bytes(32),
            "split":"train", "opening_key":f"train:{index}",
            "rules":"fixture", "trajectory_key":hashlib.sha256(str(index).encode()).digest(),
            "white_value":outcome, "analysis":analysis, "in_check":np.zeros(length,bool),
            "nnue_features":ids, "nnue_baselines":baseline}


def test_source_selection_dedup_filters_and_equal_game_weight(monkeypatch, tmp_path):
    short, long, capped, other = fake_game(1), fake_game(2,length=48,outcome=-1), fake_game(3,outcome=None), fake_game(4,run=9)
    short["analysis"][0,2] = 0       # Opening, no teacher.
    short["analysis"][1,3] = 0       # Incomplete search.
    short["analysis"][2,1] = -29000  # Mate-range score.
    short["in_check"][3] = True
    rows = [short, long, capped, other, copy.deepcopy(short)]
    # The real game stream has explicit close(); keep the stub equally strict.
    def stream(*a, **kw): yield from rows
    monkeypatch.setattr(nnue, "games", stream)
    data = nnue.dataset(tmp_path, runs=[2], split="train", positions_per_game=16)
    assert data.provenance["retained_games"] == 2
    assert data.provenance["duplicate_trajectories"] == 1
    assert np.diff(data.offsets).tolist() == [2,16]
    batch = data.batch(np.random.default_rng(8), 20000)
    # Recover white-relative targets: one win game and one loss game should be
    # equally weighted, despite radically different numbers of positions.
    white = (batch[3] * 2 - 1) * (1 - 2*batch[2])
    assert abs(white.mean()) < .03
    with pytest.raises(ValueError): nnue.dataset(tmp_path, runs=[], split="train")
    with pytest.raises(ValueError): nnue.dataset(tmp_path, runs=[2], split="test")
    with pytest.raises(ValueError, match="no eligible"): nnue.dataset(tmp_path, runs=[99], split="train")


def test_bounded_training_requires_holdout_and_saves_only_models(monkeypatch, tmp_path):
    def stream(path, split, **kw):
        for index, value in [(1,1), (2,-1)]:
            game = fake_game(index, outcome=value)
            game.update(split=split, opening_key=f"{split}:{index}")
            yield game
    monkeypatch.setattr(nnue, "games", stream)
    path = tmp_path / "outcome.safetensors"
    info = nnue.train(tmp_path, path, runs=[2], steps=1, batch_size=2, max_games=2, positions_per_game=2, jit=False)
    assert info["steps"] == 1 and info["training"]["source"] == "bumbledb"
    assert info["validation"]["split"] == "validation" and not info["promotion_ready"]
    assert info["validation_control"]["games"] == 2
    assert list(tmp_path.iterdir()) == [path]
    def leaked(path, split, **kw):
        game = fake_game(1)
        game["split"] = split
        yield game
    monkeypatch.setattr(nnue, "games", leaked)
    with pytest.raises(ValueError, match="overlap"):
        nnue.train(tmp_path, tmp_path/"rejected.safetensors", runs=[2], steps=1)
    assert list(tmp_path.iterdir()) == [path]
    def no_validation(path, split, **kw):
        if split == "train": yield fake_game(1)
    monkeypatch.setattr(nnue, "games", no_validation)
    with pytest.raises(ValueError, match="no eligible terminal validation"):
        nnue.train(tmp_path, tmp_path/"rejected.safetensors", runs=[2], steps=1)
    assert list(tmp_path.iterdir()) == [path]
