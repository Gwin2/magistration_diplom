"""Exception handling utilities for Control API."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

from fastapi import HTTPException

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def handle_workspace_errors(func: F) -> F:
    """Decorate workspace operations to handle common exceptions.

    Converts FileNotFoundError and other workspace exceptions to HTTPException
    with appropriate status codes.

    Args:
        func: The endpoint function to decorate.

    Returns:
        Wrapped function with exception handling.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as exc:
            logger.warning("Resource not found: %s", exc)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            logger.warning("Invalid value: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.error("Runtime error in workspace operation: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in workspace operation")
            raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc

    return wrapper  # type: ignore[return-value]


async def handle_workspace_errors_async(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
    """Async version of handle_workspace_errors.

    Args:
        func: The async endpoint function to decorate.

    Returns:
        Wrapped async function with exception handling.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
        try:
            return await func(*args, **kwargs)
        except FileNotFoundError as exc:
            logger.warning("Resource not found: %s", exc)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            logger.warning("Invalid value: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.error("Runtime error in workspace operation: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error in workspace operation")
            raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc

    return wrapper  # type: ignore[return-value]
