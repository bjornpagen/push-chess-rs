"""Metal-first command line. DEV must be selected before importing tinygrad."""
import argparse
from dataclasses import fields
import importlib.metadata
import json
import os
from pathlib import Path
import time
import sys

os.environ.setdefault("DEV", "METAL")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "plan":
        from .experiments import main as write_plan
        return write_plan(argv[1:])
    from tinygrad import Device
    from .run import TrainConfig, train
    parser = argparse.ArgumentParser(description="Rules-only Push Chess self-play on tinygrad/Metal")
    parser.add_argument("--allow-cpu", action="store_true", help="allow explicit DEV=CPU for diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="check Metal, checkpointing, and search/inference throughput")
    doctor.add_argument("--channels", type=int, default=96)
    doctor.add_argument("--blocks", type=int, default=6)
    doctor.add_argument("--actors", type=int, default=32)
    doctor.add_argument("--output", type=Path)
    profile = sub.add_parser("profile", help="opt-in interleaved inference/identity probes; no training")
    for name, default in (("channels", 64), ("blocks", 4), ("global-every", 0), ("effect-channels", 0),
                          ("actors", 32), ("workers", 1), ("cache-mb", 0), ("repeats", 10), ("search-simulations", 0)):
        profile.add_argument("--" + name, type=int, default=default)
    profile.add_argument("--output", type=Path, required=True)
    throughput = sub.add_parser("throughput", help="frozen-weight rolling self-play sweep, no training")
    throughput.add_argument("checkpoint", type=Path)
    throughput.add_argument("--settings", nargs="+", default=["32:32", "64:32", "128:32", "128:64", "256:64", "256:128"],
                            help="actor:inference-batch pairs")
    throughput.add_argument("--workers", type=int, default=2)
    throughput.add_argument("--group-size", type=int, default=16)
    throughput.add_argument("--repeats", type=int, default=2)
    throughput.add_argument("--seconds", type=float, default=15.)
    throughput.add_argument("--warmup", type=float, default=10.)
    throughput.add_argument("--output", type=Path, required=True)
    sub.add_parser("plan", help="write an experiment manifest without compute (use plan --output PATH)")
    training = sub.add_parser("train", help="start or resume a bounded self-play run")
    training.add_argument("--run", type=Path, required=True)
    training.add_argument("--minutes", type=float, default=60)
    training.add_argument("--iterations", type=int, default=10000)
    training.add_argument("--resume", action="store_true", help="restore the saved config, optimizer, RNG and replay")
    training.add_argument("--no-jit", action="store_true")
    defaults = TrainConfig()
    for f in fields(defaults):
        value = getattr(defaults, f.name)
        options = {"action": argparse.BooleanOptionalAction} if isinstance(value, bool) else {"type": type(value)}
        training.add_argument("--" + f.name.replace("_", "-"), default=value, **options)
    evaluation = sub.add_parser("evaluate", help="paired held-out match, never used as training data")
    evaluation.add_argument("checkpoint", type=Path, help="checkpoint file, run directory, or latest/initial.json")
    evaluation.add_argument("--opponent", default="cataclysm", help="engine name, random, or another checkpoint/run")
    evaluation.add_argument("--pairs", type=int, default=8)
    evaluation.add_argument("--simulations", type=int, default=64)
    evaluation.add_argument("--opponent-ms", type=int, default=50)
    evaluation.add_argument("--opponent-nodes", type=int, default=0)
    evaluation.add_argument("--max-plies", type=int, default=512)
    evaluation.add_argument("--opening-plies", type=int, default=6)
    evaluation.add_argument("--seed", type=int, default=918273)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--move-ms", type=float, help="same requested wall time per move for both contestants")
    evaluation.add_argument("--weights", choices=("raw", "ema"), default="raw")
    evaluation.add_argument("--opponent-weights", choices=("raw", "ema"), default="raw")
    analysis = sub.add_parser("analyse", help="choose a legal move with a saved model")
    analysis.add_argument("checkpoint", type=Path, help="checkpoint file or run directory")
    analysis.add_argument("--fen")
    analysis.add_argument("--simulations", type=int, default=128)
    args = parser.parse_args(argv)
    if Device.DEFAULT != "METAL" and not (args.allow_cpu and Device.DEFAULT == "CPU"):
        parser.error(f"expected METAL, got {Device.DEFAULT}; CPU diagnostics require DEV=CPU and --allow-cpu")
    if args.command == "train":
        config = defaults if args.resume else TrainConfig(**{f.name: getattr(args, f.name) for f in fields(defaults)})
        overrides = {f.name: getattr(args, f.name) for f in fields(defaults)
                     if any(a.split("=")[0] in ("--" + f.name.replace("_", "-"), "--no-" + f.name.replace("_", "-"))
                            for a in argv)} if args.resume else None
        train(args.run, config, args.minutes, args.iterations, args.resume, not args.no_jit, system_overrides=overrides)
    elif args.command == "throughput":
        from .throughput import sweep
        from .learning import write_json
        if args.output.exists(): raise FileExistsError(args.output)
        settings = [tuple(map(int, pair.split(":"))) for pair in args.settings]
        if any(len(pair) != 2 or min(pair) < 1 for pair in settings): parser.error("expected positive actors:batch pairs")
        result = sweep(args.checkpoint, settings, repeats=args.repeats, seconds=args.seconds, warmup=args.warmup,
                       workers=args.workers, group_size=args.group_size, progress=lambda row: print(row, flush=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    elif args.command == "profile":
        from .benchmark import probe
        from .learning import write_json
        from .model import ModelConfig
        if args.output.exists(): raise FileExistsError(args.output)
        result = probe(ModelConfig(args.channels, args.blocks, args.global_every, args.effect_channels),
                       args.actors, args.repeats, args.workers, args.cache_mb, args.search_simulations)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        from .learning import write_json
        if args.output.exists(): raise FileExistsError(args.output)
        result = evaluate(args.checkpoint, args.opponent, args.pairs, args.simulations, args.opponent_ms,
                          args.opponent_nodes, args.max_plies, args.seed, args.opening_plies,
                          progress=lambda row: print(json.dumps(row), flush=True), move_ms=args.move_ms,
                          weights=args.weights, opponent_weights=args.opponent_weights)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({k:v for k,v in result.items() if k != "games"}, indent=2))
    elif args.command == "analyse":
        import numpy as np
        from ._native import State
        from .learning import load_checkpoint
        from .model import Predictor
        from .search import search
        state = State(args.fen)
        model, _ = load_checkpoint(args.checkpoint)
        with Predictor(model, 1) as predictor:
            r = search([state], predictor, np.random.default_rng(0), args.simulations, explore=False)[0]
        print(json.dumps({"move_id": r.move, "nodes": r.nodes, "visits": int(r.visits.sum()),
                          "policy": {str(int(i)):float(p) for i,p in zip(r.observation[1], r.policy)}}))
    else:
        doctor_run(args)


def doctor_run(args):
    import tempfile
    import numpy as np
    from tinygrad import Device
    from ._native import State
    from .learning import Learner, load_checkpoint, save_checkpoint, write_json
    from .model import Network, ModelConfig, Predictor
    from .replay import Replay, Sample
    from .search import search
    model = Network(ModelConfig(args.channels, args.blocks))
    predictor = Predictor(model, args.actors)
    roots = [State() for _ in range(args.actors)]
    observations = [s.observation() for s in roots]
    for _ in range(3): predictor(observations)
    begin = time.monotonic()
    for _ in range(10): predictor(observations)
    inference_rate = 10 * args.actors / (time.monotonic() - begin)
    begin = time.monotonic()
    results = search(roots, predictor, np.random.default_rng(0), simulations=16)
    search_seconds = time.monotonic() - begin
    replay = Replay()
    for state, r in zip(roots, results):
        board, ids, actions = r.observation
        replay.extend([Sample(board, ids, actions, r.policy, np.array([1,0,0], np.float32), 1, state.fen(), [])])
    learner = Learner(model)
    for _ in range(3): metrics = learner.train(replay.batch(np.random.default_rng(0), args.actors))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "roundtrip.safetensors"
        save_checkpoint(path, learner, {})
        loaded, _ = load_checkpoint(path)
        expected = predictor([observations[0]])
        actual = Predictor(loaded, args.actors)([observations[0]])
        np.testing.assert_allclose(actual[0][0], expected[0][0], atol=1e-5)
    result = {"device": Device.DEFAULT, "tinygrad": importlib.metadata.version("tinygrad"), "parameters": model.parameter_count,
              "actors": args.actors, "inference_positions_per_second": inference_rate,
              "search_seconds": search_seconds, "rust_search_seconds": predictor.native_seconds,
              "search_ffi_calls": predictor.search_calls, "simulations": 16 * args.actors,
              "gradient_test": metrics, "checkpoint_roundtrip": "passed"}
    if args.output:
        if args.output.exists(): raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
