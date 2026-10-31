"""Running expression mutations (apply / undo) safely: chunk lock held, cancellation deferred, worker always drained."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any

from sunwai.control import hooks as ctl_hooks
from sunwai.control.errors import CtlError
from sunwai.control.service.opstate import OperationRecorder

Result = dict[str, Any]
DRAIN_POLL_S = 0.1


@dataclass
class CancelState:
    """Cancellation bookkeeping of one run.

    Attributes:
        deferred: A cancellation arrived while a worker still owned live files; re-raise it once recorded.
        recorded: The result of the worker has been written to the revision log.
    """

    deferred: bool = False
    recorded: bool = False


class NullLock:
    """An async lock that never blocks (used when the host provides no chunk lock)."""

    async def __aenter__(self) -> NullLock:
        """Enter without waiting."""
        return self

    async def __aexit__(self, *exc: object) -> bool:
        """Exit without suppressing exceptions."""
        return False


async def drain(future: concurrent.futures.Future[Result]) -> tuple[Result, bool]:
    """Wait for the REAL worker to finish no matter how often this task is cancelled.

    Never blocks the event loop: polls ``future.done()`` every 0.1 s, remembers every cancellation instead of propagating it,
    and returns only when the future is done.

    Returns:
        ``(result_dict, was_cancelled)``; the caller re-raises the cancel after recording and unlocking.
    """
    cancelled = False
    while not future.done():
        try:
            await asyncio.sleep(DRAIN_POLL_S)
        except asyncio.CancelledError:
            cancelled = True
    try:
        return future.result(), cancelled
    except BaseException as exc:
        return {"ok": False, "error": f"worker raised {type(exc).__name__}: {exc}"}, cancelled


async def lock_chunk(stack: contextlib.AsyncExitStack, chunk_lock: Any, job_id: str, lang: str, chunk: int) -> None:
    """Acquire the shared chunk lock into ``stack``, but never wait forever: about 20 s, then ``busy``.

    Raises:
        CtlError: 409 ``busy`` when the lock is not available in time.
    """
    wait = float(os.environ.get("DUB_CTL_LOCK_WAIT_S", "20"))
    lock = chunk_lock(job_id, lang, chunk) if chunk_lock else NullLock()
    try:
        await asyncio.wait_for(stack.enter_async_context(lock), wait)
    except asyncio.TimeoutError as exc:
        raise CtlError(409, "busy", "this chunk is being modified by another operation; try again in a moment") from exc


async def run_expression_mutation(
    hook_name: str, recorder: OperationRecorder, op: dict[str, Any], cancel: CancelState, *args: Any
) -> Result:
    """Run an expression mutation (apply/undo) WITH the chunk lock held by the caller.

    The worker is a ``concurrent.futures.Future``: the lock is only released and the outcome only recorded after it is done.
    Cancellation is deferred (``cancel.deferred``); a timeout marks the op ``stuck`` and keeps waiting.
    """
    timeout = float(os.environ.get("DUB_CTL_M2_TIMEOUT_S", "180"))
    future = getattr(ctl_hooks, hook_name)(*args)
    try:
        return await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout)  # type: ignore[no-any-return]
    except asyncio.TimeoutError:
        return await _wait_after_timeout(hook_name, recorder, op, cancel, future, timeout)
    except asyncio.CancelledError:
        result, _was = await drain(
            future
        )  # wrapper / task / shutdown cancelled: the worker thread still owns live files
        cancel.deferred = True
        return result
    except BaseException as exc:  # the worker itself raised
        if not future.done():
            result, _ = await drain(future)
            return result
        return {"ok": False, "error": f"worker raised {type(exc).__name__}: {exc}"}


async def _wait_after_timeout(
    hook_name: str,
    recorder: OperationRecorder,
    op: dict[str, Any],
    cancel: CancelState,
    future: concurrent.futures.Future[Result],
    timeout: float,
) -> Result:
    job_id = recorder.job_id
    message = f"[ctl][STUCK] {hook_name} on job {job_id} exceeded {timeout:.0f}s; keeping the chunk lock until the worker ends"
    print(message, flush=True)
    logging.getLogger("ctl").error(message)
    try:
        recorder.stage(op, "stuck")
        recorder.store.emit(
            "op.stuck", job_id=job_id, operation_id=op["operation_id"], action_id=op["action_id"], after_s=timeout
        )
    except Exception:
        pass
    result, was_cancelled = await drain(future)
    cancel.deferred = cancel.deferred or was_cancelled
    op.update(error=None)
    return {**result, "stuck": True}
