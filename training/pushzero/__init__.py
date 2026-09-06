"""PushZero: learned policy/value, exact rules, no strategic evaluation terms."""
__all__ = ["State", "SearchBatch", "SearchRuntime"]


def __getattr__(name):
    if name in __all__:
        from . import _native
        return getattr(_native, name)
    raise AttributeError(name)
