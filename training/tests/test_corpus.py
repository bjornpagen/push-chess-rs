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
