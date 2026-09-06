"""Small residual policy/WDL network; all strategic scores are learned."""
from dataclasses import dataclass
from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters
from .protocol import action_bucket


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 96
    blocks: int = 6
    global_every: int = 0
    effect_channels: int = 0

    def __post_init__(self):
        if self.channels < 8 or self.channels % 8 or not 1 <= self.blocks <= 32:
            raise ValueError("channels must be a positive multiple of 8; blocks must be 1..32")
        if not 0 <= self.global_every <= self.blocks or not 0 <= self.effect_channels <= 128:
            raise ValueError("invalid global/effect architecture")


class Block:
    def __init__(self, channels, global_context=False):
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.context = nn.Linear(channels * 2, channels, bias=False) if global_context else None

    def __call__(self, x):
        y = self.conv1(self.norm1(x).relu())
        if self.context is not None:
            context = y.mean(axis=(2, 3)).cat(y.max(axis=(2, 3)), dim=1)
            y = y + self.context(context).reshape(y.shape[0], y.shape[1], 1, 1)
        return x + self.conv2(self.norm2(y).relu()) * 0.1


class Network:
    def __init__(self, config=ModelConfig()):
        self.config = config
        self.revision = 0
        c = config.channels
        self.stem = nn.Conv2d(32, c, 3, padding=1)
        self.blocks = [Block(c, bool(config.global_every and (i + 1) % config.global_every == 0))
                       for i in range(config.blocks)]
        self.norm = nn.GroupNorm(8, c)
        self.source = nn.Linear(c, c, bias=False)
        self.destination = nn.Linear(c, c, bias=False)
        self.global_policy = nn.Linear(c, c, bias=False)
        self.route = nn.Embedding(3, c)
        self.stop = nn.Embedding(16, c)
        self.promotion = nn.Embedding(7, c)
        self.special = nn.Embedding(4, c)
        self.policy = nn.Linear(c, 1)
        self.value_hidden = nn.Linear(c, c)
        self.value = nn.Linear(c, 3)
        if config.effect_channels:
            e = config.effect_channels
            self.effect_square = nn.Linear(c, e, bias=False)
            self.effect_before = nn.Embedding(13, e)
            self.effect_after = nn.Embedding(13, e)
            self.effect_position = nn.Embedding(64, e)
            self.effect_output = nn.Linear(e, c, bias=False)
        # Uniform initial policy and W/D/L: no piece values or inherited teacher.
        for layer in (self.policy, self.value):
            layer.weight.assign(Tensor.zeros_like(layer.weight)).realize()
            layer.bias.assign(Tensor.zeros_like(layer.bias)).realize()

    def __call__(self, boards: Tensor, actions: Tensor, effects: Tensor | None = None):
        x = self.stem(boards)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).relu()
        pooled = x.mean(axis=(2, 3))
        squares = x.permute(0, 2, 3, 1).reshape(x.shape[0], 64, self.config.channels)
        batch = Tensor.arange(x.shape[0]).to(x.device).reshape(-1, 1)
        h = self.source(squares)[batch, actions[:, :, 0]] + self.destination(squares)[batch, actions[:, :, 1]]
        h = h + self.global_policy(pooled).unsqueeze(1)
        h = h + self.route(actions[:, :, 2]) + self.stop(actions[:, :, 3])
        h = h + self.promotion(actions[:, :, 4]) + self.special(actions[:, :, 5])
        if self.config.effect_channels:
            if effects is None:
                raise ValueError("effect-aware network requires transition tokens")
            # Tokens bind square/before/after jointly, preserving which piece
            # changed where. Project once before gathering into a small width.
            e = self.effect_square(squares)[batch, effects[:, :, 1]]
            e = (e + self.effect_before(effects[:, :, 2]) + self.effect_after(effects[:, :, 3])
                 + self.effect_position(effects[:, :, 1])).relu()
            # Zero is a dedicated padding segment, never action zero. This
            # portable baseline is differentiable; profile before custom kernels.
            groups = effects[:, :, 0].one_hot(actions.shape[1] + 1).cast(e.dtype)[:, :, 1:]
            h = h + self.effect_output(groups.transpose(1, 2).matmul(e))
        return self.policy(h.relu()).squeeze(-1), self.value(self.value_hidden(pooled).relu())

    @property
    def parameter_count(self):
        return sum(p.numel() for p in get_parameters(self))


def unpack(observation):
    # Rust transfers allocation ownership to NumPy; no per-element boxing.
    return observation


# Compatibility import: applications may keep importing Predictor from model.
from .inference import Predictor
