"""Quantization-aware training of Cataclysm's 768→32 residual, not its search.

All game data comes from typed bumbledb pages. Retained rows are bounded; the
deduplication digest set scales with scanned source games. Only weights,
optimizer state and training provenance may become file artifacts.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time
import uuid

import numpy as np
from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_load_metadata, safe_save
from ._native import RULES_VERSION, nnue_control
from .corpus import games
from .learning import optimizer_state

FEATURES, WIDTH, SLOTS = 768, 32, 64
MODEL_BYTES = (FEATURES + 2) * WIDTH * 2
FORMAT = "cataclysm-residual-v1"
# Q4 feature/bias and Q10 output parameters give useful float optimizer scales.
SCALES = {"features": 16, "bias": 16, "output": 1024}
SHAPES = {"features": (FEATURES, WIDTH), "bias": (WIDTH,), "output": (WIDTH,)}
# Keeps every clipped integer dot product below 2**24, exactly representable
# in float32. Rust accepts the full i16 format; this learner uses a safe subset.
OUTPUT_LIMIT = 2047


def decode(data):
    if len(data) != MODEL_BYTES: raise ValueError("invalid Cataclysm model shape")
    words = np.frombuffer(data, dtype="<i2").astype(np.int64)
    return {"features": words[:FEATURES*WIDTH].reshape(FEATURES, WIDTH),
            "bias": words[FEATURES*WIDTH:(FEATURES+1)*WIDTH], "output": words[(FEATURES+1)*WIDTH:]}


def _feature_ids(ids):
    ids = np.asarray(ids)
    if ids.ndim != 3 or ids.shape[1:] != (2, SLOTS) or ids.dtype.kind not in "iu" or np.any(ids > FEATURES):
        raise ValueError("expected [N,2,64] feature IDs in 0..768")
    if ids.dtype.kind == "i" and np.any(ids < 0): raise ValueError("negative feature ID")
    return ids


def dense_features(ids):
    """Expand only the sampled batch for dense Metal GEMM, never the corpus.

    Sparse IDs remain 256 bytes/position in replay; dense GEMM avoids a large
    scatter-gradient intermediate and Python per-piece calls during training.
    """
    ids = _feature_ids(ids)
    dense = np.zeros((*ids.shape[:2], FEATURES + 1), np.float32)
    np.put_along_axis(dense, ids.astype(np.intp), 1, axis=2)
    return np.ascontiguousarray(dense[:, :, :FEATURES])


def integer_scores(data, ids, baselines, sides):
    """Independent int64 reference for all i16 models, including negative /.

    Rust integer division truncates toward zero, unlike Python's //.
    """
    p = decode(data)
    table = np.concatenate((p["features"], np.zeros((1, WIDTH), np.int64)))
    hidden = np.clip(table[_feature_ids(ids)].sum(axis=2) + p["bias"], 0, 256)
    total = (hidden[:, 0] - hidden[:, 1]) @ p["output"]
    residual = np.sign(total) * (np.abs(total) // 8192)
    return np.clip((np.asarray(baselines, np.int64) + residual) * (1 - 2*np.asarray(sides, np.int64)) + 14,
                   -28000, 28000).astype(np.int32)


def _quantize(x, scale, low, high):
    value = (x * scale).clip(low, high)
    return value + (value.round() - value).detach()


class Residual:
    def __init__(self, data=None):
        params = decode(nnue_control() if data is None else data)
        if np.max(np.abs(params["output"])) > OUTPUT_LIMIT:
            raise ValueError("output weights exceed exact float32 training range")
        for key, value in params.items():
            setattr(self, key, Tensor((value / SCALES[key]).astype(np.float32), device=Device.DEFAULT).realize())

    def __call__(self, features, baselines, sides):
        weight = _quantize(self.features, 16, -32768, 32767)
        bias = _quantize(self.bias, 16, -32768, 32767)
        output = _quantize(self.output, 1024, -OUTPUT_LIMIT, OUTPUT_LIMIT)
        hidden = (features @ weight + bias).clip(0, 256)
        residual = ((hidden[:, 0] - hidden[:, 1]) * output).sum(axis=1) / 8192
        residual = residual + (residual.trunc() - residual).detach()
        return ((baselines + residual) * (1 - 2*sides) + 14).clip(-28000, 28000)

    def export(self):
        pieces = []
        for key, scale in SCALES.items():
            values = getattr(self, key).numpy() * scale
            if not np.isfinite(values).all(): raise FloatingPointError("non-finite NNUE parameters")
            low, high = (-OUTPUT_LIMIT, OUTPUT_LIMIT) if key == "output" else (-32768, 32767)
            pieces.append(np.rint(np.clip(values, low, high)).astype("<i2").tobytes())
        return b"".join(pieces)


class Learner:
    def __init__(self, model=None, *, lr=1e-3, score_scale=400., jit=True):
        if not np.isfinite([lr, score_scale]).all() or min(lr, score_scale) <= 0:
            raise ValueError("positive finite learning rate and score scale required")
        self.model = Residual() if model is None else model
        self.optimizer = nn.optim.AdamW(get_parameters(self.model), lr=lr, weight_decay=0.)
        self.score_scale, self.steps, self.jit, self.compiled = score_scale, 0, jit, {}

    def train(self, batch):
        size = len(batch[0])
        if size not in self.compiled:
            def update(features, baseline, side, target):
                self.optimizer.zero_grad()
                logits = self.model(features, baseline, side) / self.score_scale
                loss = logits.binary_crossentropy_logits(target)
                loss.backward()
                norm = sum(p.grad.square().sum() for p in self.optimizer.params).sqrt()
                scale = (5 / (norm + 1e-6)).minimum(1)
                for p in self.optimizer.params: p.grad = p.grad * scale
                Tensor.realize(loss, norm, *self.optimizer.schedule_step())
                return loss, norm
            self.compiled[size] = TinyJit(update) if self.jit else update
        # No reduced-precision tensor-core multiplication: QAT must match the
        # actual integer accumulator, not a numerically similar float model.
        with Context(TRAINING=1, TC=0):
            loss, norm = self.compiled[size](*[Tensor(x, device=Device.DEFAULT) for x in batch])
            metrics = {"loss": float(loss.item()), "gradient_norm": float(norm.item())}
        if not all(np.isfinite(v) for v in metrics.values()): raise FloatingPointError(metrics)
        self.steps += 1
        return metrics


@dataclass
class Dataset:
    ids: np.ndarray
    baselines: np.ndarray
    sides: np.ndarray
    targets: np.ndarray
    offsets: np.ndarray
    provenance: dict
    opening_families: frozenset

    def batch(self, rng, size):
        # Uniform games, then uniform sampled positions: long games do not
        # become dozens of times more influential than short games.
        game = rng.integers(len(self.offsets) - 1, size=size)
        start, end = self.offsets[game], self.offsets[game + 1]
        indices = start + (rng.random(size) * (end - start)).astype(np.int64)
        return (dense_features(self.ids[indices]), self.baselines[indices], self.sides[indices], self.targets[indices])


def dataset(path, *, runs, split, max_games=10000, positions_per_game=16, seed=1):
    """Explicit, sealed source runs; true outcomes and completed non-check roots.

    Search scores are used only to omit mate-range trivialities, not as mixed,
    uncalibrated teacher labels. Opening/test/arena moves cannot become targets.
    """
    runs = sorted(set(map(int, runs)))
    if not runs or min(runs) < 1: raise ValueError("explicit positive source run IDs required")
    if split not in ("train", "validation"): raise ValueError("NNUE may read train/validation, never test/arena")
    if not 1 <= max_games <= 100000 or not 1 <= positions_per_game <= 64:
        raise ValueError("bounded positive game/position limits required")
    rng, retained, seen, scanned = np.random.default_rng(seed), [], 0, 0
    duplicates, trajectories, identities = 0, set(), set()
    stream = games(path, split, nnue=True)
    try:
        for game in stream:
            if game["run_id"] not in runs: continue
            if game["split"] != split: raise ValueError("corpus split mismatch")
            scanned += 1
            if game["white_value"] is None: continue
            if game["trajectory_key"] in trajectories:
                duplicates += 1
                continue
            trajectories.add(game["trajectory_key"])
            analysis = game["analysis"]
            eligible = np.flatnonzero((analysis[:, 2] == 1) & (analysis[:, 3] == 1)
                & (np.abs(analysis[:, 1]) < 28000) & ~game["in_check"])
            if not len(eligible): continue
            identities.add((game["run_id"], game["white"], game["black"], game["binary"].hex(), game["rules"]))
            selected = np.sort(rng.choice(eligible, size=min(len(eligible), positions_per_game), replace=False))
            sides = analysis[selected, 0].astype(np.float32)
            target = ((1 - 2*sides) * game["white_value"] + 1) / 2
            key = f'{game["run_id"]}:{game["game_index"]}:'.encode() + game["trajectory_key"] + selected.astype("<u4").tobytes()
            row = (game["nnue_features"][selected].copy(), game["nnue_baselines"][selected].astype(np.float32),
                   sides, target.astype(np.float32), key, game["opening_key"])
            seen += 1
            if len(retained) < max_games: retained.append(row)
            else:
                index = int(rng.integers(seen))
                if index < max_games: retained[index] = row
    finally:
        stream.close()
    if not retained: raise ValueError(f"no eligible terminal {split} games in source runs {runs}")
    ids, baselines, sides, targets = [np.concatenate([row[i] for row in retained]) for i in range(4)]
    offsets = np.cumsum([0] + [len(row[0]) for row in retained], dtype=np.int64)
    digest = hashlib.sha256()
    for row in retained: digest.update(row[4])
    info = {"source": "bumbledb", "path": str(Path(path).resolve()), "runs": runs, "split": split,
            "scanned_games": scanned, "eligible_games": seen, "retained_games": len(retained),
            "retained_positions": len(ids), "duplicate_trajectories": duplicates, "seed": seed,
            "max_games": max_games, "positions_per_game": positions_per_game,
            "sample_sha256": digest.hexdigest(), "identities": sorted(identities),
            "targets": "terminal expected score; completed non-check roots; abs(search score)<28000"}
    return Dataset(ids, baselines, sides, targets, offsets, info, frozenset(row[5] for row in retained))


def evaluate(model, data, score_scale=400., batch_size=256):
    """Exact exported evaluator loss, averaged within games then across games."""
    encoded = model.export()
    loss = np.empty(len(data.ids), np.float64)
    for start in range(0, len(loss), batch_size):
        stop = start + batch_size
        scores = integer_scores(encoded, data.ids[start:stop], data.baselines[start:stop], data.sides[start:stop])
        logits = scores.astype(np.float64) / score_scale
        loss[start:stop] = np.logaddexp(0, logits) - data.targets[start:stop] * logits
    means = np.add.reduceat(loss, data.offsets[:-1]) / np.diff(data.offsets)
    return {"loss": float(means.mean()), "games": len(means), "positions": len(loss)}


def save(path, learner, metadata):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {**metadata, "format": FORMAT, "rules": RULES_VERSION, "tinygrad": importlib.metadata.version("tinygrad"),
            "steps": learner.steps, "score_scale": learner.score_scale,
            "control_sha256": hashlib.sha256(nnue_control()).hexdigest(),
            "export_sha256": hashlib.sha256(learner.model.export()).hexdigest()}
    tensors = {"model." + k: v for k, v in get_state_dict(learner.model).items()}
    tensors.update({"optimizer." + k: v for k, v in optimizer_state(learner.optimizer).items()})
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.partial")
    safe_save(tensors, str(temp), metadata={"nnue": json.dumps(info)})
    with temp.open("rb") as stream: os.fsync(stream.fileno())
    # Exclusive publication: a racing run cannot overwrite another candidate.
    os.link(temp, path)
    temp.unlink()
    return info


def load(path, *, training=False, jit=True):
    info = json.loads(safe_load_metadata(str(path))[2]["__metadata__"]["nnue"])
    if info.get("format") != FORMAT or info.get("rules") != RULES_VERSION:
        raise ValueError("NNUE checkpoint rules/format mismatch")
    if info.get("tinygrad") != importlib.metadata.version("tinygrad"):
        raise ValueError("NNUE checkpoint tinygrad differs from pinned runtime")
    model, tensors = Residual(), safe_load(str(path))
    state = {k.removeprefix("model."): v for k, v in tensors.items() if k.startswith("model.")}
    if {k: v.shape for k, v in state.items()} != SHAPES: raise ValueError("NNUE parameter schema mismatch")
    load_state_dict(model, state, verbose=False)
    if hashlib.sha256(model.export()).hexdigest() != info["export_sha256"]: raise ValueError("NNUE export digest mismatch")
    if not training: return model, info
    learner = Learner(model, score_scale=info["score_scale"], jit=jit)
    target = optimizer_state(learner.optimizer)
    saved = {k.removeprefix("optimizer."): v for k, v in tensors.items() if k.startswith("optimizer.")}
    if {k: v.shape for k, v in target.items()} != {k: v.shape for k, v in saved.items()}:
        raise ValueError("NNUE optimizer schema mismatch")
    for key, tensor in target.items(): tensor.assign(saved[key].to(tensor.device)).realize()
    learner.steps = info["steps"]
    return learner, info


def train(db, output, *, runs, steps=1000, batch_size=256, max_games=10000, validation_games=2000,
          positions_per_game=16, seed=1, resume=None, jit=True):
    if Path(output).exists(): raise FileExistsError(output)
    if not 1 <= steps <= 1000000 or not 1 <= batch_size <= 4096: raise ValueError("bounded positive step/batch limits required")
    Tensor.manual_seed(seed)
    source = dataset(db, runs=runs, split="train", max_games=max_games, positions_per_game=positions_per_game, seed=seed)
    validation = dataset(db, runs=runs, split="validation", max_games=validation_games, positions_per_game=positions_per_game, seed=seed)
    if source.opening_families & validation.opening_families:
        raise ValueError("training and validation opening families overlap")
    learner = load(resume, training=True, jit=jit)[0] if resume else Learner(jit=jit)
    before = evaluate(learner.model, validation, learner.score_scale)
    control = evaluate(Residual(), validation, learner.score_scale)
    stopped, rng, started, metrics = False, np.random.default_rng(seed), time.monotonic(), {}
    def stop(*_):
        nonlocal stopped
        stopped = True
    handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for _ in range(steps):
            if stopped: break
            metrics = learner.train(source.batch(rng, batch_size))
            if learner.steps % 25 == 0: print({"nnue_steps": learner.steps, **metrics}, flush=True)
    finally:
        for s, handler in handlers.items(): signal.signal(s, handler)
    after = evaluate(learner.model, validation, learner.score_scale)
    return save(output, learner, {"training": source.provenance, "validation": validation.provenance,
        "parent": str(resume) if resume else None, "parent_sha256": hashlib.sha256(Path(resume).read_bytes()).hexdigest() if resume else None,
        "seconds": time.monotonic()-started, "interrupted": stopped, "metrics": metrics,
        "validation_before": before, "validation_after": after, "validation_control": control, "promotion_ready": False})


def export(checkpoint, output):
    """Candidate artifact only. Never install or replace the embedded control."""
    model, _ = load(checkpoint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = model.export()
    with output.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(output), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
