"""Exact shared features, signed integer arithmetic, QAT, and game balancing."""
import copy
import hashlib
import numpy as np
import pytest
from tinygrad import Context, Tensor
from pushzero._native import State, NNUE_CONTROL, NNUE_SPEC, Nnue, nnue_inputs
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


def integer_reference(data, ids, baselines, sides):
    """Independent test oracle, deliberately not a production evaluator.

    Pin the format's constants independently so shared-spec mistakes still fail.
    """
    words = np.frombuffer(data, dtype="<i2").astype(np.int64)
    table = np.concatenate((words[:768*32].reshape(768,32), np.zeros((1,32),np.int64)))
    hidden = np.clip(table[ids].sum(axis=2) + words[768*32:769*32], 0, 256)
    total = (hidden[:,0] - hidden[:,1]) @ words[769*32:]
    residual = np.sign(total) * (np.abs(total) // 8192)
    return np.clip((np.asarray(baselines,np.int64) + residual) * (1-2*np.asarray(sides,np.int64)) + 14,
                   -28000,28000).astype(np.int32)


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
    model = nnue.Model()
    assert model.export() == NNUE_CONTROL
    scores = Nnue(NNUE_CONTROL).positions(states)
    np.testing.assert_array_equal(integer_reference(NNUE_CONTROL, ids, baseline, side), scores)
    np.testing.assert_array_equal(model.scores(ids, baseline, side, batch_size=7), scores)
    np.testing.assert_array_equal(metal_scores(model, ids, baseline, side), scores)
    assert nnue_inputs([])[0].shape == (0, 2, 64)
    assert Nnue(NNUE_CONTROL).positions([]).shape == (0,)
    assert model.scores(*nnue_inputs([]), np.zeros(0)).shape == (0,)
    with pytest.raises(ValueError): Nnue(b"bad")
    with pytest.raises(ValueError): nnue_inputs([states[0]] * 4097)


def test_full_i16_decoder_and_signed_division_agree_with_rust():
    states = positions(16)
    ids, baseline = nnue_inputs(states)
    side = np.array([state.turn() for state in states])
    rng = np.random.default_rng(33)
    data = rng.integers(-32768, 32768, (nnue.FEATURES+2)*nnue.WIDTH, dtype=np.int16).astype("<i2").tobytes()
    native = Nnue(data)
    expected = integer_reference(data, ids, baseline, side)
    np.testing.assert_array_equal(expected, native.positions(states))
    np.testing.assert_array_equal(expected, native.scores(ids, baseline, side.astype(np.uint8)))
    with pytest.raises(ValueError, match="exact float32"): nnue.Model(data)
    with pytest.raises(ValueError): nnue.dense_features(np.full((1,2,64), 769, np.uint16))
    with pytest.raises(ValueError): nnue.dense_features(np.zeros((1,2,63), np.uint16))


def test_single_spec_native_validation_and_snapshot_freshness():
    assert NNUE_SPEC == {"format":"cataclysm-residual-v1", "bytes":49280, "features":768,
        "width":32, "slots":64, "hidden_clip":256, "residual_divisor":8192, "tempo":14, "score_limit":28000}
    model = nnue.Model()
    states = positions(5)
    ids, baseline = nnue_inputs(states)
    sides = np.array([s.turn() for s in states],np.uint8)
    before = model.scores(ids,baseline,sides)
    frozen = model.freeze()
    model.output.assign(Tensor(np.zeros(32,np.float32))).realize()
    after = model.scores(ids,baseline,sides)
    np.testing.assert_array_equal(after,Nnue(model.export()).positions(states))
    assert np.any(before != after)
    np.testing.assert_array_equal(before,frozen.scores(ids,baseline,sides))
    native = Nnue(NNUE_CONTROL)
    bad = ids.copy()
    bad[0,0,0] = 769
    with pytest.raises(ValueError): native.scores(bad,baseline,sides)
    with pytest.raises(ValueError): model.scores(bad,baseline,sides)
    bad[0,0,0] = bad[0,0,1]
    with pytest.raises(ValueError,match="duplicate"): native.scores(bad,baseline,sides)
    with pytest.raises(ValueError,match="duplicate"): model.scores(bad,baseline,sides)
    for baselines in (np.full(5,2**31,dtype=np.int64),baseline.astype(float)+.5,np.full(5,np.nan)):
        with pytest.raises(ValueError): model.scores(ids,baselines,sides)
    with pytest.raises(ValueError): native.scores(ids,baseline,np.full(5,2,np.uint8))
    with pytest.raises(ValueError): model.scores(ids,baseline,sides+2)
    with pytest.raises(ValueError): model.scores(ids,baseline,sides,batch_size=0)
    with pytest.raises(ValueError): native.scores(ids[:,:,:63],baseline,sides)


def test_cpu_and_metal_training_use_the_same_complete_update():
    states = positions(8)
    ids,baseline = nnue_inputs(states)
    sides = np.array([s.turn() for s in states],np.float32)
    batch = (nnue.dense_features(ids),baseline.astype(np.float32),sides,np.linspace(0,1,8,dtype=np.float32))
    learners = [nnue.Learner(nnue.Model(device=device)) for device in ("CPU","METAL")]
    for _ in range(4):
        a,b = [learner.train(batch) for learner in learners]
        np.testing.assert_allclose(list(a.values()),list(b.values()),rtol=2e-5,atol=2e-6)
        for key in nnue.SCALES:
            np.testing.assert_allclose(getattr(learners[0].model,key).numpy(),getattr(learners[1].model,key).numpy(),rtol=2e-5,atol=2e-6)


def test_checkpoint_restores_parameters_and_optimizer_on_selected_device(tmp_path):
    source = nnue.Learner(nnue.Model(device="CPU"))
    states = positions(4)
    ids,baseline = nnue_inputs(states)
    batch = (nnue.dense_features(ids),baseline.astype(np.float32),
             np.array([s.turn() for s in states],np.float32),np.array([0,1,.5,0],np.float32))
    source.train(batch)
    path = tmp_path/"cross-device.safetensors"
    nnue.save(path,source,{})
    restored,_ = nnue.load(path,training=True,device="METAL")
    assert restored.steps == source.steps and restored.model.export() == source.model.export()
    a,b = nnue.optimizer_state(source.optimizer),nnue.optimizer_state(restored.optimizer)
    for key in a:
        assert b[key].device == "METAL"
        np.testing.assert_array_equal(a[key].numpy(),b[key].numpy())
    assert all(getattr(restored.model,key).device == "METAL" for key in nnue.SCALES)


def test_half_integer_rounding_and_parameter_clipping_match_export():
    states = positions(12)
    ids, baseline = nnue_inputs(states)
    side = np.array([state.turn() for state in states])
    model, rng = nnue.Model(), np.random.default_rng(32)
    for key, scale in nnue.SCALES.items():
        words = rng.integers(-120, 121, nnue.SHAPES[key]).astype(np.float32) + .5
        words.flat[:4] = [-40000.5, -2047.5, 2047.5, 40000.5]
        getattr(model, key).assign(Tensor(words / scale)).realize()
    data = model.export()
    expected = Nnue(data).positions(states)
    np.testing.assert_array_equal(metal_scores(model, ids, baseline, side), expected)
    np.testing.assert_array_equal(model.scores(ids, baseline, side), expected)
    for key, scale in nnue.SCALES.items():
        limit = (-2047,2047) if key == "output" else (-32768,32767)
        expected = np.rint(np.clip(getattr(model,key).numpy()*scale, *limit))
        np.testing.assert_array_equal(nnue.decode(data)[key], expected)


def test_candidate_actual_search_isolated_identity_and_bounded_analysis():
    from pushzero._native import Opponent
    with pytest.raises(ValueError, match="cannot replace"): Opponent("cataclysm", NNUE_CONTROL)
    with pytest.raises(ValueError): Opponent("aurora", b"bad")
    control, candidate = Opponent("cataclysm"), Opponent("control-copy", NNUE_CONTROL)
    assert control.network_fingerprint() == candidate.network_fingerprint()
    assert Opponent("astra").network_fingerprint() is None
    for state in positions(8):
        control.new_game()
        candidate.new_game()
        original = state.fen()
        a = control.analyse(state, time_ms=0, nodes=1024)
        b = candidate.analyse(state, time_ms=0, nodes=1024)
        for key in a:
            if key != "wall_us": np.testing.assert_array_equal(a[key], b[key])
        assert a["move"] in state.legal_ids() and a["nodes"] <= 1024
        assert state.fen() == original and a["pv"].dtype == np.uint32
        line = state.copy()
        for move in a["pv"]: line.play(int(move))
    zero = bytes(nnue.MODEL_BYTES)
    different = Opponent("zero-fixture", zero)
    assert different.network_fingerprint() != control.network_fingerprint()
    state = positions(4)[-1]
    row = different.analyse(state, time_ms=0, nodes=1)
    assert row["score"] == Nnue(zero).positions([state])[0] and not row["complete"]
    with pytest.raises(ValueError): candidate.analyse(state, time_ms=0, nodes=0)
    with pytest.raises(ValueError): candidate.analyse(state, depth=101)
    with pytest.raises(ValueError): candidate.new_game(side=2)


def test_granite_uses_the_shared_native_search_without_a_network():
    from pushzero._native import Opponent
    abacus, granite = Opponent("abacus"), Opponent("granite")
    assert abacus.network_fingerprint() is not None
    assert granite.network_fingerprint() is None
    with pytest.raises(ValueError, match="cannot replace"): Opponent("granite", NNUE_CONTROL)
    for state in positions(8):
        abacus.new_game()
        granite.new_game()
        for _ in range(2):
            a = abacus.analyse(state, time_ms=0, nodes=2048)
            b = granite.analyse(state, time_ms=0, nodes=2048)
            for key in a:
                if key != "wall_us": np.testing.assert_array_equal(a[key], b[key])


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
    expected = Nnue(encoded).positions(states)
    np.testing.assert_array_equal(metal_scores(learner.model, ids, baseline, side), expected)
    np.testing.assert_array_equal(learner.model.scores(ids, baseline, side), expected)
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
    def stream(*a, runs, **kw):
        yield from (row for row in rows if row["run_id"] in runs)
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
    def unexpected(*a, **kw): yield other
    monkeypatch.setattr(nnue, "games", unexpected)
    with pytest.raises(ValueError, match="source run mismatch"):
        nnue.dataset(tmp_path, runs=[2], split="train")


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
    assert info["validation_handwritten"]["games"] == 2
    assert np.isfinite(info["validation_handwritten"]["loss"])
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


def test_zero_network_validation_is_exactly_the_handwritten_control():
    states = positions(32)
    ids, baseline = nnue_inputs(states)
    sides = np.array([state.turn() for state in states], np.uint8)
    expected = np.clip(baseline.astype(np.int64) * (1 - 2*sides.astype(np.int64)) + 14, -28000, 28000)
    np.testing.assert_array_equal(nnue.Model(bytes(nnue.MODEL_BYTES)).scores(ids,baseline,sides), expected)
