"""bumbledb facts → typed arrays → exact-history targets → Metal update."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from pushzero._native import CorpusReader, State
from pushzero.corpus import games, samples, replay


@pytest.fixture
def corpus(tmp_path):
    lab = Path(__file__).resolve().parents[2] / "target/release/lab"
    assert lab.exists(), "build lab before corpus integration tests"
    db = tmp_path / "corpus"
    def command(*args):
        result = subprocess.run([str(lab), *args], capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr
        return result
    command("init", "--db", str(db))
    command("tournament", "--db", str(db), "--engines", "cataclysm,astra", "--pairs", "3",
            "--workers", "1", "--nodes", "1024", "--max-plies", "8", "--opening-plies", "4",
            "--purpose", "corpus", "--max-seconds", "60", "--seed", "7")
    command("verify", "--db", str(db), "--run", "1")
    return db


def test_native_game_pages_have_owned_arrays_and_exact_targets(corpus):
    with pytest.raises(ValueError): CorpusReader(str(corpus / "missing"))
    reader = CorpusReader(str(corpus))
    totals = json.loads(reader.summary())["totals"]
    assert totals["games"] == 6
    assert totals["positions"] == totals["moves"] + 6
    with pytest.raises(ValueError): reader.page("train", (0,0), 0)
    reader.close()
    reader.close()
    with pytest.raises(ValueError, match="closed"): reader.summary()
    with pytest.raises(ValueError, match="closed"): reader.page()
    all_games = [g for split in ("train", "validation", "test") for g in games(corpus, split)]
    assert len(all_games) == 6
    assert sum(len(g["moves"]) for g in all_games) == totals["moves"]
    for game in all_games:
        assert game["moves"].dtype == np.uint32
        assert game["analysis"].shape == (len(game["moves"]),6)
        assert isinstance(game["trajectory_key"], bytes)
        state = State(game["initial_fen"])
        for move in game["moves"]: state.play(int(move))
        assert state.fen() == game["final_fen"]
        assert state.outcome() == game["white_value"]
        for s in samples(game):
            assert s.ply >= 4 and s.value_weight == int(game["white_value"] is not None)
            assert s.wdl.sum() == s.value_weight
            assert int(s.ids[np.argmax(s.policy)]) == int(game["moves"][s.ply])
    buffer, info = replay(corpus, capacity=8)
    assert 0 < len(buffer.samples) <= 8 and info["source"] == "bumbledb"
    assert all(s.provenance["teacher"] in ("astra","cataclysm") for s in buffer.samples)


def test_teacher_pretraining_keeps_only_tensor_checkpoint_artifacts(corpus, tmp_path):
    from pushzero.pretrain import train
    from pushzero.learning import load_checkpoint
    path = tmp_path / "student.safetensors"
    info = train(corpus,path,steps=1,batch_size=2,channels=8,blocks=1,capacity=8,jit=False)
    assert info["steps"] == 1 and info["teacher_corpus"]["source"] == "bumbledb"
    _, restored = load_checkpoint(path,training=True,jit=False)
    assert restored["steps"] == 1
    assert not list(tmp_path.rglob("*.npz"))
    with pytest.raises(FileExistsError): train(corpus,path,steps=1)


def test_nnue_page_features_are_owned_and_match_exact_replay(corpus):
    from pushzero._native import nnue_inputs
    all_games = [g for split in ("train", "validation", "test") for g in games(corpus, split, nnue=True)]
    assert len(all_games) == 6
    for game in all_games:
        states, state = [], State(game["initial_fen"])
        for move in game["moves"]:
            states.append(state.copy())
            state.play(int(move))
        ids, baselines = nnue_inputs(states)
        np.testing.assert_array_equal(game["nnue_features"], ids)
        np.testing.assert_array_equal(game["nnue_baselines"], baselines)
        assert game["in_check"].dtype == np.bool_ and game["in_check"].shape == (len(states),)


def test_source_run_selection_reaches_native_pages_and_rejects_partial_requests(corpus):
    selected = [g for split in ("train", "validation", "test")
                for g in games(corpus, split, page_size=1, nnue=True, runs=[1,1])]
    assert len(selected) == 6 and {g["run_id"] for g in selected} == {1}
    reader = CorpusReader(str(corpus))
    try:
        for runs, message in (([], "nonempty positive"), ([0,1], "nonempty positive"), ([1,999], "unknown source run")):
            with pytest.raises(ValueError, match=message): reader.page(limit=1, runs=runs)
    finally:
        reader.close()
    # The generator must also close its owner when native selection fails.
    with pytest.raises(ValueError, match="unknown source run"):
        list(games(corpus, runs=[1,999]))
    reader = CorpusReader(str(corpus))
    reader.close()


def test_nnue_training_from_real_terminal_relations(tmp_path):
    from pushzero.nnue import train, load, export
    lab = Path(__file__).resolve().parents[2] / "target/release/lab"
    db = tmp_path / "terminal-corpus"
    def command(*args):
        result = subprocess.run([str(lab), *args], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    command("init", "--db", str(db))
    command("tournament", "--db", str(db), "--engines", "cataclysm,astra", "--pairs", "32",
            "--workers", "1", "--nodes", "256", "--max-plies", "128", "--opening-plies", "4",
            "--purpose", "corpus", "--max-seconds", "90", "--seed", "2026090613")
    audited = command("verify", "--db", str(db), "--run", "1")
    assert audited["verified_games"] == 64 and audited["terminal_games"] > 0
    path = tmp_path / "outcome.safetensors"
    info = train(db, path, runs=[1], steps=1, batch_size=4, max_games=8, validation_games=4,
                 positions_per_game=2, jit=False)
    assert info["steps"] == 1 and info["validation_control"]["games"] > 0
    assert info["training"]["split"] == "train" and info["validation"]["split"] == "validation"
    assert info["training"]["retained_positions"] <= 16
    learner, restored = load(path, training=True, jit=False)
    assert learner.steps == 1 and restored["export_sha256"] == info["export_sha256"]
    export(path, tmp_path / "candidate.bin")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candidate.bin", "outcome.safetensors", "terminal-corpus"]


def test_forensic_states_and_full_search_details_stay_typed(corpus):
    all_games = [g for split in ("train", "validation", "test") for g in games(corpus, split)]
    reader = CorpusReader(str(corpus))
    try:
        for game in all_games:
            n = len(game["moves"])
            assert game["search_details"].shape == (n,5)
            assert game["search_details"].dtype == np.int64
            assert game["pv_offsets"].shape == (n+1,)
            assert game["pv_offsets"][0] == 0 and game["pv_offsets"][-1] == len(game["pv_actions"])
            assert np.all(np.diff(game["pv_offsets"].astype(np.int64)) >= 0)
            for key in ("proof_searches", "mate_proofs"):
                assert game[key].ndim == 2 and game[key].shape[1] == 2 and game[key].dtype == np.uint64
                assert np.all(game[key][:,0] < n)
            state = State(game["initial_fen"])
            for ply in range(n+1):
                restored = reader.state(game["run_id"], game["game_index"], ply)
                assert restored.fen() == state.fen() and restored.outcome() == state.outcome()
                assert restored.legal_ids() == state.legal_ids()
                if ply < n:
                    start, end = game["pv_offsets"][ply:ply+2]
                    preview = state.copy()
                    for move in game["pv_actions"][start:end]: preview.play(int(move))
                    state.play(int(game["moves"][ply]))
            assert state.fen() == game["final_fen"]
        with pytest.raises(ValueError): reader.state(999,0,0)
        with pytest.raises(ValueError): reader.state(1,999,0)
        with pytest.raises(ValueError): reader.state(1,0,999)
    finally:
        reader.close()
    with pytest.raises(ValueError, match="closed"): reader.state(1,0,0)
