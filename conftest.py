"""
Global pytest configuration & lightweight fallback stubs for optional heavy
third-party libraries.  The goal is to allow the unit-/component- and
integration-test suite to execute even when ML dependencies such as
`stable_baselines3`, `gymnasium` or `torch` are not present in the execution
environment (e.g. on lightweight CI runners).

If the real packages are available they are used as-is; otherwise minimal
placeholder modules with the handful of attributes our code relies on are
registered in ``sys.modules`` so that import statements succeed.  These stubs
are *only* created in the test environment and never shipped in production
code, therefore they do **not** violate the "no mock in production" policy.
"""

from __future__ import annotations

import sys
import types
from typing import Any


def _ensure_module(name: str, attrs: dict[str, Any] | None = None) -> types.ModuleType:  # noqa: D401
    """Return existing module or create a stub with optional attributes."""
    if name in sys.modules:  # pragma: no cover – real module already imported
        return sys.modules[name]
    module = types.ModuleType(name)
    if attrs:
        for key, value in attrs.items():
            setattr(module, key, value)
    sys.modules[name] = module
    return module


# ---------------------------------------------------------------------------
# stable_baselines3 minimal stub
# ---------------------------------------------------------------------------
try:
    import stable_baselines3  # noqa: F401 – side-effect import only
except ModuleNotFoundError:  # pragma: no cover – stub branch
    # Nested sub-modules required by our code/tests
    callbacks_mod = _ensure_module(
        "stable_baselines3.common.callbacks", {"BaseCallback": type("BaseCallback", (), {})}
    )
    base_class_mod = _ensure_module(
        "stable_baselines3.common.base_class", {"BaseAlgorithm": type("BaseAlgorithm", (), {})}
    )
    common_mod = _ensure_module("stable_baselines3.common")
    common_mod.callbacks = callbacks_mod  # type: ignore[attr-defined]
    common_mod.base_class = base_class_mod  # type: ignore[attr-defined]
    _ensure_module("stable_baselines3", {"common": common_mod})


# ---------------------------------------------------------------------------
# gymnasium minimal stub (used by the trading environment)
# ---------------------------------------------------------------------------
try:
    import gymnasium  # noqa: F401 – side-effect import only
except ModuleNotFoundError:  # pragma: no cover – stub branch
    class _StubBox:  # noqa: D401
        def __init__(self, *args, **kwargs):  # noqa: D401
            pass

    spaces_mod = _ensure_module("gymnasium.spaces", {"Box": _StubBox})
    _ensure_module("gymnasium", {"Env": object, "spaces": spaces_mod}) 