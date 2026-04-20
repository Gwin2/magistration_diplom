"""Optional import utilities for UAV ViT framework.

This module provides helpers for optional dependencies and graceful degradation.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from types import ModuleType

T = TypeVar("T", bound=Callable[..., Any])


def optional_import(module_name: str) -> ModuleType | None:
    """Import a module optionally, returning None if not available.

    Args:
        module_name: Name of the module to import.

    Returns:
        The imported module or None if ImportError occurs.
    """
    try:
        return import_module(module_name)
    except ImportError:
        return None


def require_optional(module_name: str, feature_name: str | None = None) -> Callable[[T], T]:
    """Decorator to mark functions that require an optional dependency.

    Raises ImportError with a helpful message if the module is not available.

    Args:
        module_name: Name of the required module.
        feature_name: Optional feature name for error message.

    Returns:
        Decorated function that checks for the dependency before execution.
    """
    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            module = optional_import(module_name)
            if module is None:
                feature = feature_name or module_name
                raise ImportError(
                    f"{feature} requires '{module_name}' package. "
                    f"Install it with: pip install {module_name}"
                )
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


class LazyImport:
    """Lazy import wrapper that imports a module only on first access.

    Example:
        mlflow = LazyImport("mlflow")
        # mlflow module is not imported yet
        result = mlflow.get_tracking_uri()  # Now it's imported
    """

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: ModuleType | None = None

    def _import(self) -> ModuleType:
        if self._module is None:
            module = optional_import(self._module_name)
            if module is None:
                raise ImportError(f"Required module '{self._module_name}' is not installed")
            self._module = module
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._import(), name)

    def __repr__(self) -> str:
        if self._module is None:
            return f"<LazyImport '{self._module_name}' (not loaded)>"
        return repr(self._module)
