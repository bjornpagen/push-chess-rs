import gc
import json
import numpy as np
import pytest
from tinygrad import Tensor
from pushzero._native import State, SearchBatch
from pushzero.model import ModelConfig, Network, Predictor
from pushzero.learning import Learner, save_checkpoint, load_checkpoint
from pushzero.replay import Sample, Replay, save_shard, load_shard
from pushzero.search import search
from pushzero.selfplay import collect, reduced_start, reanalyse
from pushzero.run import TrainConfig, run_lock, train


class ZeroPredictor:
    positions = seconds = native_seconds = search_calls = 0
    def packed(self, boards, actions):
        self.positions += len(boards)
        return np.zeros(actions.shape[:2], np.float32), np.zeros(len(boards), np.float32)


def sample():
    state = State()
    board, ids, actions = state.observation()
    return Sample(board, ids, actions, np.full(len(ids), 1/len(ids), np.float32), np.array([1,0,0], np.float32), 1, state.fen(), [])


def test_state_atomic_and_numpy_ownership():
    state = State()
    original = state.fen()
    with pytest.raises(ValueError): state.play(2**32-1)
    assert state.fen() == original
    observation = state.observation()
    state.play(state.legal_ids()[0])
    del state
    gc.collect()
    board, ids, actions = observation
    assert board.shape == (32,8,8) and board.dtype == np.float32
    assert ids.dtype == np.uint32 and actions.shape == (len(ids),6)
    assert all(a.flags.c_contiguous for a in observation)
    assert board[:12].sum() == 32
    assert board[12:24].sum() == 0


@pytest.mark.parametrize("fen,result", [
    ("7k/6Q1/5K2/8/8/8/8/8 b - - 100 1", 1),
    ("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1", 0),
    ("7k/8/8/8/8/8/8/K7 w - - 100 1", 0)])
def test_terminal(fen,result):
    state = State(fen)
    assert state.outcome() == result and not state.legal_ids()
    with pytest.raises(ValueError): search([state], ZeroPredictor(), np.random.default_rng(0), 8)


def test_batch_validation_retry_and_determinism():
    state = State()
    batch = SearchBatch([state], np.zeros((1,128),np.float32), 4,4)
    with pytest.raises(ValueError): batch.results()
    boards, actions = batch.request()
    with pytest.raises(ValueError): batch.request()
    with pytest.raises(ValueError): batch.submit(np.zeros((1,128),np.float32),np.array([np.nan],np.float32))
    batch.submit(np.zeros((1,128),np.float32),np.zeros(1,np.float32))
    while (features := batch.request()) is not None:
        batch.submit(*ZeroPredictor().packed(*features))
    assert batch.results()[0][2].sum() == 4
    outputs = [search([state],ZeroPredictor(),np.random.default_rng(5),32)[0] for _ in range(2)]
    assert outputs[0].move == outputs[1].move
    np.testing.assert_array_equal(outputs[0].policy,outputs[1].policy)


def test_search_proves_mate():
    state = State("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    n = len(state.legal_ids())
    result = search([state], ZeroPredictor(),np.random.default_rng(1),n,n,explore=False)[0]
    state.play(result.move)
    assert state.outcome() == 1
    assert result.policy.max() > .5


def test_replay_roundtrip_and_truncation(tmp_path):
    samples, games = collect(ZeroPredictor(), np.random.default_rng(1),games=4,actors=4,
                            simulations=2,fast_simulations=1,max_plies=1,curriculum=0,full_fraction=1)
    assert all(g["truncated"] for g in games)
    assert all(s.value_weight == 0 and s.wdl.sum() == 0 for s in samples)
    path = save_shard(tmp_path,samples,{})
    restored,_ = load_shard(path)
    np.testing.assert_array_equal(restored[0].board,samples[0].board)
    np.testing.assert_array_equal(restored[0].actions,samples[0].actions)
    renewed = reanalyse(restored,ZeroPredictor(),np.random.default_rng(2),2)
    assert all(s.value_weight == 0 for s in renewed)
    samples[0].wdl[1] = 1
    with pytest.raises(ValueError): save_shard(tmp_path,samples,{})


def test_rules_only_curriculum_is_valid():
    rng = np.random.default_rng(3)
    for _ in range(32):
        state = reduced_start(rng)
        assert state.outcome() is None
        assert state.observation()[0][:12].sum() <= 6


def test_jit_learning_checkpoint_and_exact_next_update(tmp_path):
    Tensor.manual_seed(3)
    model = Network(ModelConfig(16,1))
    learner = Learner(model,lr=.003)
    replay = Replay()
    replay.extend([sample()])
    batch = replay.batch(np.random.default_rng(0),4)
    predictor = Predictor(model,4)
    obs = State().observation()
    for _ in range(3): before = predictor([obs])[0]
    losses = [learner.train(batch)["loss"] for _ in range(20)]
    assert losses[-1] < losses[0] - .1
    after = predictor([obs])[0]
    assert after[1] > before[1] + .1, "inference JIT must see updated weights"
    path = tmp_path / "model.safetensors"
    save_checkpoint(path,learner,{"test":True})
    with pytest.raises(FileExistsError): save_checkpoint(path,learner,{})
    loaded,info = load_checkpoint(path,training=True)
    assert info["steps"] == 20
    restored = Predictor(loaded.model,4)([obs])[0]
    np.testing.assert_allclose(after[0],restored[0],atol=1e-6)
    assert abs(after[1]-restored[1]) < 1e-6
    for _ in range(3):
        a,b = learner.train(batch),loaded.train(batch)
        for key in a: assert a[key] == pytest.approx(b[key],abs=1e-5)


def test_lock_and_training_resume(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with run_lock(tmp_path): pass
    config = TrainConfig(channels=8,blocks=1,actors=2,games=2,simulations=2,fast_simulations=1,
                         max_plies=2,curriculum=0,full_fraction=1,batch_size=4,reuse=1)
    first = train(tmp_path,config,minutes=1,iterations=1)
    assert first["iteration"] == 1 and first["steps"] > 0
    with pytest.raises(FileExistsError): train(tmp_path,config,minutes=1,iterations=1)
    second = train(tmp_path,minutes=1,iterations=1,resume=True)
    assert second["iteration"] == 2 and second["steps"] > first["steps"]
    _,info = load_checkpoint(tmp_path/second["checkpoint"])
    assert info["config"] == json.loads(json.dumps(config.__dict__))
    assert len(info["shards"]) == 2


def test_evaluation_counts_unfinished_separately(tmp_path):
    from pushzero.evaluation import evaluate
    from pushzero.learning import write_json, resolve_checkpoint
    path = tmp_path / "model.safetensors"
    save_checkpoint(path,Learner(Network(ModelConfig(8,1))),{})
    write_json(tmp_path/"latest.json",{"checkpoint":path.name})
    assert resolve_checkpoint(tmp_path) == path
    result = evaluate(tmp_path,opponent="random",pairs=1,simulations=1,max_plies=1,opening_plies=0)
    assert result["truncated"] == 2 and result["draws"] == 0
    assert result["score_bounds"] == [0,1]


def test_neural_action_buckets_preserve_all_actions():
    from pushzero.model import action_bucket
    assert action_bucket(129) == 256
    predictor = Predictor(Network(ModelConfig(8,1)),2)
    board,_,actions = State().observation()
    wide = np.tile(actions,(6,1))[:257]
    # This is a shape/packing probe, not a legal-position example.
    outputs = predictor([(board,np.arange(len(wide),dtype=np.uint32),wide)])
    assert len(outputs[0][0]) == 257 and np.isfinite(outputs[0][0]).all()
    assert any(key[1] == 512 for key in predictor.compiled)


def test_interior_batching_matches_independent_search():
    states = [State(),State("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")]
    together = search(states,ZeroPredictor(),np.random.default_rng(0),64,explore=False)
    alone = [search([s],ZeroPredictor(),np.random.default_rng(0),64,explore=False)[0] for s in states]
    for a,b in zip(together,alone):
        assert a.move == b.move and a.nodes == b.nodes
        np.testing.assert_array_equal(a.policy,b.policy)
        np.testing.assert_array_equal(a.visits,b.visits)
