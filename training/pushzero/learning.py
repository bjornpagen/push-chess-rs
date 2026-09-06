"""tinygrad AdamW updates and atomic model+optimizer checkpoints."""
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import uuid

import numpy as np
from tinygrad import Context, Device, Tensor, TinyJit, nn
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_load, safe_load_metadata, safe_save
from ._native import RULES_VERSION, ENCODING_VERSION, EFFECT_ENCODING_VERSION
from .model import ModelConfig, Network


class Learner:
    def __init__(self, model, lr=3e-4, jit=True, ema_decay=0.0):
        if not 0 <= ema_decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        self.model = model
        self.optimizer = nn.optim.AdamW(get_parameters(model), lr=lr, weight_decay=1e-4)
        self.jit, self.compiled, self.steps = jit, {}, 0
        self.ema_decay = ema_decay
        self.ema = {k: p.detach().clone().realize() for k, p in get_state_dict(model).items()} if ema_decay else {}

    def train(self, batch):
        width = batch[1].shape[1]
        expected = 7 if self.model.config.effect_channels else 6
        if len(batch) != expected:
            raise ValueError("training batch does not match model effect schema")
        key = (len(batch[0]), width, batch[6].shape[1] if expected == 7 else 0)
        if key not in self.compiled:
            def update(x, a, target_policy, mask, target_wdl, value_weight, *effects):
                self.optimizer.zero_grad()
                logits, values = self.model(x, a, effects[0] if effects else None)
                log_policy = (logits + (1 - mask) * -1e9).log_softmax()
                ploss = -(target_policy * log_policy).sum(axis=1).mean()
                vloss = (-(target_wdl * values.log_softmax()).sum(axis=1) * value_weight).sum() / value_weight.sum().maximum(1)
                loss = ploss + vloss
                loss.backward()
                norm = sum(p.grad.square().sum() for p in self.optimizer.params).sqrt()
                scale = (5.0 / (norm + 1e-6)).minimum(1.0)
                for p in self.optimizer.params:
                    p.grad = p.grad * scale
                Tensor.realize(loss, ploss, vloss, norm, *self.optimizer.schedule_step())
                return loss, ploss, vloss, norm
            self.compiled[key] = TinyJit(update) if self.jit else update
        with Context(TRAINING=1):
            result = self.compiled[key](*[Tensor(x, device=Device.DEFAULT) for x in batch])
            metrics = dict(zip(("loss", "policy_loss", "value_loss", "gradient_norm"), [float(x.item()) for x in result], strict=True))
        self.model.revision += 1
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f"non-finite training: {metrics}")
        self.steps += 1
        if self.ema:
            with Context(TRAINING=0):
                for key, p in get_state_dict(self.model).items():
                    self.ema[key].assign(self.ema[key] * self.ema_decay + p.detach() * (1 - self.ema_decay)).realize()
        return metrics


def optimizer_state(opt):
    return get_state_dict({"lr": opt.lr, "b1_t": opt.b1_t, "b2_t": opt.b2_t, "m": opt.m, "v": opt.v})


def save_checkpoint(path, learner, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {**metadata, "format": 1, "rules": RULES_VERSION, "encoding": ENCODING_VERSION,
            "tinygrad": importlib.metadata.version("tinygrad"), "model": asdict(learner.model.config), "steps": learner.steps,
            "effect_encoding": EFFECT_ENCODING_VERSION if learner.model.config.effect_channels else None,
            "ema_decay": learner.ema_decay}
    tensors = {"model." + k: v for k, v in get_state_dict(learner.model).items()}
    tensors.update({"optimizer." + k: v for k, v in optimizer_state(learner.optimizer).items()})
    tensors.update({"ema." + k: v for k, v in learner.ema.items()})
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.partial")
    safe_save(tensors, str(temp), metadata={"pushzero": json.dumps(info)})
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return info


def resolve_checkpoint(path):
    """A run directory or pointer resolves once to an immutable checkpoint."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "latest.json"
    if path.suffix == ".json":
        name = json.loads(path.read_text())["checkpoint"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("checkpoint pointer must name a file in its own run directory")
        path = path.parent / name
    return path


def load_checkpoint(path, training=False, jit=True, *, weights="raw"):
    if weights not in ("raw", "ema") or (training and weights != "raw"):
        raise ValueError("training resumes raw optimizer-matched weights; evaluation may select raw or ema")
    path = resolve_checkpoint(path)
    metadata = safe_load_metadata(str(path))[2]["__metadata__"]
    info = json.loads(metadata["pushzero"])
    if info.get("format") != 1 or info.get("rules") != RULES_VERSION or info.get("encoding") != ENCODING_VERSION:
        raise ValueError("checkpoint rules/format mismatch")
    if info.get("tinygrad") != importlib.metadata.version("tinygrad"):
        raise ValueError("checkpoint tinygrad version differs from the pinned runtime")
    model = Network(ModelConfig(**info["model"]))
    if model.config.effect_channels and info.get("effect_encoding") != EFFECT_ENCODING_VERSION:
        raise ValueError("checkpoint effect schema mismatch")
    tensors = safe_load(str(path))
    prefix = "model." if weights == "raw" else "ema."
    selected = {k.removeprefix(prefix): v for k, v in tensors.items() if k.startswith(prefix)}
    if selected.keys() != get_state_dict(model).keys():
        raise ValueError("missing or incompatible selected checkpoint weights")
    load_state_dict(model, selected, verbose=False)
    if not training:
        return model, info
    learner = Learner(model, jit=jit, ema_decay=info.get("ema_decay", 0.0))
    if learner.ema:
        saved_ema = {k.removeprefix("ema."): v for k, v in tensors.items() if k.startswith("ema.")}
        if learner.ema.keys() != saved_ema.keys():
            raise ValueError("EMA state mismatch")
        for key, tensor in learner.ema.items():
            if tensor.shape != saved_ema[key].shape: raise ValueError("EMA tensor shape mismatch")
            tensor.assign(saved_ema[key].to(tensor.device)).realize()
    target = optimizer_state(learner.optimizer)
    saved = {k.removeprefix("optimizer."): v for k, v in tensors.items() if k.startswith("optimizer.")}
    if target.keys() != saved.keys():
        raise ValueError("optimizer state mismatch")
    for key, tensor in target.items():
        if tensor.shape != saved[key].shape:
            raise ValueError("optimizer tensor shape mismatch")
        tensor.assign(saved[key].to(tensor.device)).realize()
    learner.steps = info["steps"]
    return learner, info


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.partial")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
