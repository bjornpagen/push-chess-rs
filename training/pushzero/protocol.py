"""Shared wire shapes; importing this module does not initialize a device."""
import numpy as np

BOARD_SHAPE = (32, 8, 8)
ACTION_FIELDS = 6
EFFECT_FIELDS = 4  # action index + 1 (zero = padding), square, before, after
EFFECT_ENCODING_VERSION = 1
ACTION_LIMITS = np.array([64, 64, 3, 16, 7, 4], np.int32)


def bucket(n, minimum=1):
    if n < 0 or minimum < 1:
        raise ValueError("invalid bucket size")
    return max(minimum, 1 << (max(1, int(n)) - 1).bit_length())


def action_bucket(n):
    """A shape choice, never a legal-action cap."""
    return bucket(n, 16)


def observation_parts(observation):
    if len(observation) not in (3, 4):
        raise ValueError("expected board, IDs, actions, and optional effect tokens")
    return (*observation[:3], observation[3] if len(observation) == 4 else None)


def pack_observations(observations, with_effects=False):
    parts = [observation_parts(o) for o in observations]
    lengths = np.asarray([len(p[1]) for p in parts], np.int32)
    width = action_bucket(int(lengths.max(initial=0)))
    boards = np.empty((len(parts), *BOARD_SHAPE), np.float32)
    actions = np.zeros((len(parts), width, ACTION_FIELDS), np.int32)
    tokens = None
    if with_effects:
        if any(p[3] is None for p in parts):
            raise ValueError("effect-aware model requires exact transition inputs")
        tokens = np.zeros((len(parts), bucket(max((len(p[3]) for p in parts), default=0), 16), EFFECT_FIELDS), np.int32)
    for i, (board, _ids, moves, effects) in enumerate(parts):
        boards[i] = board
        actions[i, :lengths[i]] = moves
        if tokens is not None:
            tokens[i, :len(effects)] = effects
    return boards, actions, lengths, tokens
