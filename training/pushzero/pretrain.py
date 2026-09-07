"""Bounded Metal teacher learner. Only model/optimizer tensors are saved."""
import time
import signal
from pathlib import Path
import numpy as np
from tinygrad import Tensor
from .corpus import replay
from .learning import Learner, save_checkpoint, load_checkpoint
from .model import ModelConfig, Network


def train(db, output, *, steps=1000, batch_size=128, capacity=100_000,
          max_games=10_000, channels=64, blocks=4, seed=1, resume=None, jit=True):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    if min(steps,batch_size,capacity,max_games)<1: raise ValueError("positive training limits required")
    Tensor.manual_seed(seed)
    buffer, provenance = replay(db, capacity=capacity, max_games=max_games, seed=seed)
    learner = load_checkpoint(resume, training=True, jit=jit)[0] if resume else Learner(Network(ModelConfig(channels,blocks)), jit=jit)
    rng, stopped = np.random.default_rng(seed), False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signals = {s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    started, metrics = time.monotonic(), {}
    try:
        for _ in range(steps):
            if stopped: break
            metrics = learner.train(buffer.batch(rng,batch_size,effects=bool(learner.model.config.effect_channels)))
            if learner.steps%25==0: print({"steps":learner.steps,**metrics},flush=True)
    finally:
        for s, handler in signals.items(): signal.signal(s,handler)
    return save_checkpoint(output,learner,{"teacher_corpus":provenance,"parent":str(resume) if resume else None,
        "seconds":time.monotonic()-started,"interrupted":stopped,"metrics":metrics})
