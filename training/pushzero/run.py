"""Bounded, restartable training transactions. Incumbents never supply labels."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import uuid

import numpy as np
from tinygrad import Device, Tensor
from .learning import Learner, load_checkpoint, resolve_checkpoint, save_checkpoint, write_json
from .model import ModelConfig, Network, Predictor
from .replay import Replay, load_shard, save_shard
from .selfplay import RollingCollector, reanalyse
from .curriculum import RestartArchive


@dataclass(frozen=True)
class TrainConfig:
    channels: int = 96
    blocks: int = 6
    global_every: int = 0
    effect_channels: int = 0
    workers: int = 1
    inference_cache_mb: int = 0
    max_graphs: int = 32
    max_nodes: int = 16384
    reconstruction_cache_mb: int = 64
    ema_decay: float = 0.0
    restart_fraction: float = 0.0
    restart_capacity: int = 2048
    fast_explore: bool = True
    actors: int = 32
    inference_batch_size: int = 64
    search_group_size: int = 16
    games: int = 64
    simulations: int = 64
    fast_simulations: int = 16
    full_fraction: float = .25
    max_plies: int = 512
    curriculum: float = .25
    batch_size: int = 128
    replay_capacity: int = 100_000
    reuse: float = 4.0
    learning_rate: float = 3e-4
    reanalysis: int = 0
    seed: int = 20260906

    def __post_init__(self):
        ModelConfig(self.channels, self.blocks, self.global_every, self.effect_channels)
        for name in ("actors", "inference_batch_size", "search_group_size", "games", "simulations", "fast_simulations", "max_plies", "batch_size", "replay_capacity"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.fast_simulations > self.simulations:
            raise ValueError("fast search cannot exceed full search")
        if not 0 < self.full_fraction <= 1 or not 0 <= self.curriculum <= 1:
            raise ValueError("invalid full-search/curriculum fraction")
        if not math.isfinite(self.reuse) or self.reuse <= 0 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("reuse and learning rate must be finite and positive")
        if self.reanalysis < 0 or self.simulations > 1_000_000:
            raise ValueError("invalid reanalysis/search budget")
        if not 0 <= self.ema_decay < 1 or not 0 <= self.restart_fraction <= 1 - self.curriculum:
            raise ValueError("invalid averaging/restart configuration")
        if not 0 <= self.workers <= 256 or min(self.inference_cache_mb, self.reconstruction_cache_mb, self.restart_capacity) < 0 or min(self.max_graphs, self.max_nodes) < 1:
            raise ValueError("invalid memory/worker budget")
        if max(self.actors, self.inference_batch_size) > 4096 or self.search_group_size > self.inference_batch_size:
            raise ValueError("search groups must fit the bounded inference batch")


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another trainer owns this run") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def stop_signals():
    requested = []
    def handler(signum, _frame):
        requested.append(signum)
    old = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield lambda: bool(requested)
    finally:
        for sig, previous in old.items():
            signal.signal(sig, previous)


def shard_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_distribution(samples):
    """Only newly collected/renewed rows; do not decode the whole replay."""
    def quantiles(values): return np.quantile(values, [0, .5, .95, 1]).tolist()
    return {"action_count_quantiles": quantiles([len(s.ids) for s in samples]),
            "ply_quantiles": quantiles([s.ply if s.ply is not None else len(s.history) for s in samples]),
            "board_density_quantiles": quantiles([s.provenance["board_density"] for s in samples]),
            "policy_entropy_mean": float(np.mean([-np.sum(s.policy * np.log(s.policy.clip(1e-30))) for s in samples]))}


SYSTEM_FIELDS = frozenset({"actors", "inference_batch_size", "search_group_size", "workers", "max_graphs", "inference_cache_mb"})


def train(directory, config=TrainConfig(), minutes=60, iterations=10000, resume=False, jit=True, *, system_overrides=None):
    if not math.isfinite(minutes) or minutes <= 0 or iterations < 1:
        raise ValueError("minutes and iterations must be positive")
    directory = Path(directory).resolve()
    if system_overrides and (not resume or set(system_overrides) - SYSTEM_FIELDS):
        raise ValueError("resume overrides are restricted to explicit systems settings")
    with run_lock(directory), stop_signals() as stop:
        latest = directory / "latest.json"
        if resume:
            if not latest.exists():
                raise FileNotFoundError("no committed checkpoint to resume")
            pointer = json.loads(latest.read_text())
            learner, saved = load_checkpoint(resolve_checkpoint(latest), training=True, jit=jit)
            config = TrainConfig(**saved["config"])
            config = replace(config, **(system_overrides or {}))
            rng = np.random.default_rng()
            rng.bit_generator.state = saved["rng"]
            shards, iteration, games_total = saved["shards"], saved["iteration"], saved["games_total"]
            # The pointer fallback permits annotating an older checkpoint that
            # predates pending-update metadata without rewriting its weights.
            pending_updates = saved.get("pending_updates", pointer.get("pending_updates", 0))
        else:
            if any(p.name != "run.lock" for p in directory.iterdir()):
                raise FileExistsError("run directory is not empty; use --resume or choose a new directory")
            Tensor.manual_seed(config.seed)
            rng = np.random.default_rng(config.seed)
            learner = Learner(Network(ModelConfig(config.channels, config.blocks, config.global_every, config.effect_channels)),
                              lr=config.learning_rate, jit=jit, ema_decay=config.ema_decay)
            shards, iteration, games_total, pending_updates = [], 0, 0, 0
            saved = {}
        if not isinstance(pending_updates, int) or pending_updates < 0:
            raise ValueError("invalid pending-update count")
        replay = Replay(config.replay_capacity, config.reconstruction_cache_mb << 20)
        archive = RestartArchive(config.restart_capacity, saved.get("restart_archive", ()))
        shard_hashes = dict(saved.get("shard_hashes", {}))
        # Only shards recorded in a committed checkpoint are resumed. Files
        # left by an interrupted transaction are intentionally not consumed.
        for shard in shards:
            path = (directory / shard).resolve()
            if not path.is_relative_to(directory): raise ValueError("replay shard escapes run directory")
            if shard in shard_hashes and shard_digest(path) != shard_hashes[shard]:
                raise ValueError("committed replay checksum mismatch")
            samples, _ = load_shard(path)
            replay.extend(samples)
        restored, actor_info = None, None
        if "actors" in saved:
            actor_path = (directory / saved["actors"]["file"]).resolve()
            if not actor_path.is_relative_to(directory) or shard_digest(actor_path) != saved["actors"]["sha256"]:
                raise ValueError("actor checkpoint path/checksum mismatch")
            restored, actor_info = RollingCollector.restore(actor_path)
        predictor = Predictor(learner.model, config.inference_batch_size, jit=jit, workers=config.workers,
                              cache_bytes=config.inference_cache_mb << 20, max_graphs=config.max_graphs,
                              group_size=config.search_group_size)
        collector = RollingCollector(predictor, rng, actors=config.actors, simulations=config.simulations,
            fast_simulations=config.fast_simulations, full_fraction=config.full_fraction, max_plies=config.max_plies,
            curriculum=config.curriculum, archive=archive, restart_fraction=config.restart_fraction,
            fast_explore=config.fast_explore, max_nodes=config.max_nodes, restored=restored)
        if actor_info is not None:
            collector.moves = actor_info["stats"]["moves"]
            collector.completed = actor_info["stats"]["completed_total"]
            collector.started = actor_info["started"]
        begin = time.monotonic()
        deadline = begin + minutes * 60
        def emit(row):
            record = {"time": time.time(), "elapsed": time.monotonic() - begin, **row}
            with (directory / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            write_json(directory / "status.json", record)
            print(json.dumps(record), flush=True)
        def checkpoint():
            name = f"checkpoint-{iteration:06d}-{uuid.uuid4().hex[:8]}.safetensors"
            actor_path = collector.save(directory)
            info = save_checkpoint(directory / name, learner, {"config": asdict(config), "rng": rng.bit_generator.state,
                "shards": shards, "iteration": iteration, "games_total": games_total, "device": Device.DEFAULT,
                "pending_updates": pending_updates, "restart_archive": archive.records(), "shard_hashes": shard_hashes,
                "actors": {"file": actor_path.name, "sha256": shard_digest(actor_path)}})
            write_json(latest, {"checkpoint": name, "iteration": iteration, "steps": learner.steps,
                               "pending_updates": pending_updates})
            return name, info
        def learn_pending():
            nonlocal pending_updates
            metrics = []
            while pending_updates and time.monotonic() < deadline and not stop():
                metrics.append(learner.train(replay.batch(rng, config.batch_size, effects=bool(config.effect_channels))))
                pending_updates -= 1
                if len(metrics) % 100 == 0:
                    emit({"event": "learning_progress", "steps": learner.steps,
                          "pending_updates": pending_updates, **metrics[-1]})
            return metrics
        def commit(metrics):
            name, _ = checkpoint()
            emit({"event": "checkpoint", "checkpoint": name, "iteration": iteration, "steps": learner.steps,
                  "games_total": games_total, "completed_updates": len(metrics), "pending_updates": pending_updates,
                  "metrics": {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {},
                  "inference_positions": predictor.positions, "inference_seconds": predictor.seconds,
                  "native_boundary_seconds": predictor.native_seconds, "search_ffi_calls": predictor.search_calls,
                  "inference": predictor.statistics() if hasattr(predictor, "statistics") else {},
                  "last_search": getattr(predictor, "last_search_metrics", {}), "restart_entries": len(archive),
                  "selfplay": collector.statistics(),
                  "reconstruction_cache_bytes": replay.cache.bytes})
        try:
            if not resume:
                initial, _ = checkpoint()
                write_json(directory / "initial.json", {"checkpoint": initial})
            emit({"event": "started", "device": Device.DEFAULT, "parameters": learner.model.parameter_count,
                  "config": asdict(config), "resume": resume, "iteration": iteration})
            if pending_updates:
                if not replay.samples:
                    raise ValueError("pending updates require committed replay")
                emit({"event": "resuming_learning", "pending_updates": pending_updates})
                commit(learn_pending())
            for _ in range(iterations):
                if pending_updates or time.monotonic() >= deadline or stop():
                    break
                # Reserve some of the bounded run for learning/checkpointing.
                collect_deadline = deadline - min(30, minutes * 6)
                if time.monotonic() >= collect_deadline:
                    break
                samples, games = collector.collect(config.games, deadline=collect_deadline, stop=stop, progress=emit,
                                                   policy_steps=learner.steps, iteration=iteration + 1)
                if not samples:
                    # A deadline may arrive before any game ends. Preserve all
                    # unfinished targets/history/RNG; do not manufacture draws.
                    commit([])
                    break
                if config.reanalysis and replay.samples and time.monotonic() < collect_deadline and not stop():
                    subset = [replay.samples[i] for i in rng.choice(len(replay.samples), min(config.reanalysis, len(replay.samples)), replace=False)]
                    samples.extend(reanalyse(subset, predictor, rng, config.simulations,
                                             deadline=collect_deadline, stop=stop, max_nodes=config.max_nodes))
                shard = save_shard(directory / "replay", samples, {"iteration": iteration + 1,
                    "model": asdict(learner.model.config), "steps": learner.steps, "weights": "raw",
                    "parent_checkpoint": json.loads(latest.read_text())["checkpoint"]})
                shards.append(str(shard.relative_to(directory)))
                shard_hashes[shards[-1]] = shard_digest(shard)
                replay.extend(samples)
                write_json(directory / f"games-{iteration+1:06d}-{uuid.uuid4().hex[:8]}.json", games)
                games_total += len(games)
                pending_updates = max(1, math.ceil(len(samples) * config.reuse / config.batch_size))
                emit({"event": "learning", "new_samples": len(samples), "replay": len(replay.samples), "updates": pending_updates,
                      "terminal_games": sum(g["white_outcome"] is not None for g in games),
                      "truncated_games": sum(g["truncated"] for g in games),
                      "distribution": sample_distribution(samples),
                      "terminal_value_mse": [g["value_mse"] for g in games if g.get("value_mse") is not None],
                      "start_sources": {s: sum(g.get("source", "standard") == s for g in games) for s in ("standard", "sparse", "restart")}})
                metrics = learn_pending()
                iteration += 1
                commit(metrics)
            emit({"event": "stopped", "reason": "signal" if stop() else "budget_or_iterations", "iteration": iteration,
                  "steps": learner.steps, "games_total": games_total, "latest": json.loads(latest.read_text())})
        except Exception as exc:
            emit({"event": "failed", "error": str(exc), "recoverable_from": str(latest)})
            raise
        finally:
            if hasattr(predictor, "close"): predictor.close()
        return json.loads(latest.read_text())
