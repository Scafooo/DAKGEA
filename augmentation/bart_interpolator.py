"""Legacy shim: prefer `augmentation.methods.bart_interpolator`."""

from augmentation.methods.bart_interpolator import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
