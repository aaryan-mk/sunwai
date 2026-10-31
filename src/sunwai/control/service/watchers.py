"""Background watchers: lip-sync readiness of an applied line, and readiness of a newly requested language track."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from typing import Any

from sunwai.control import hooks as ctl_hooks
from sunwai.control.hooks import NotImplementedHook
from sunwai.control.service.opstate import OperationRecorder


async def watch_lipsync(deps: Any, recorder: OperationRecorder, op: dict[str, Any], job_dir: Any, lang: str, chunk: int, rv: dict[str, Any], started: float) -> None:
    """Wait until the line's lip-sync is real, then complete the operation.

    ``lipsync_worker3`` already re-lips any chunk whose ``dub_<lang>.mp4`` is newer than its ``dub_<lang>_lipsync.mp4`` (3 s
    settle), so an Apply or edit that remixes the dub is picked up automatically: nothing is triggered here, only watched.
    Lip sync is ready only when the file exists, is non-empty and is NEWER than the dub; on timeout the audio stays live
    while the mouth still shows the original frames.
    """
    limit = float(os.environ.get("DUB_CTL_LS_WAIT", "900"))
    poll = float(os.environ.get("DUB_CTL_LS_POLL", "5"))
    began = time.time()
    check = getattr(deps, "lipsync_ready", None)
    store, job_id = recorder.store, recorder.job_id
    while check and time.time() - began < limit:
        try:
            if check(job_dir, lang, chunk, rv):
                op.update(lipsync_ready=True)
                recorder.stage(op, "complete")
                store.emit("op.lipsync_ready", job_id=job_id, operation_id=op["operation_id"], revision=rv["revision"])
                store.emit(
                    "op.done", job_id=job_id, operation_id=op["operation_id"], action_id=op["action_id"],
                    revision=rv["revision"], elapsed_s=round(time.time() - started, 2),
                )
                return
        except Exception:
            pass
        await asyncio.sleep(poll)
    op.update(lipsync_ready=False)
    op.update(stage="audio_ready", state="done")
    store.put_op(op)  # audio is live; mouth movements are still the original frames
    store.emit("op.lipsync_pending", job_id=job_id, operation_id=op["operation_id"], revision=rv["revision"])
    store.emit(
        "op.done", job_id=job_id, operation_id=op["operation_id"], action_id=op["action_id"],
        revision=rv["revision"], elapsed_s=round(time.time() - started, 2), lipsync_ready=False,
    )


class TrackWatcher:
    """Holds the synthesis slot of an add-language operation until its track is ready, failed or past its deadline."""

    def __init__(
        self,
        recorder: OperationRecorder,
        unclaim: Callable[[str], None],
        release_language: Callable[[str, str, str], None],
        confirm_done: Callable[[str | None, bool], None],
    ) -> None:
        """Bind the job recorder and the slot/token callbacks of the owning service."""
        self.recorder = recorder
        self._unclaim = unclaim
        self._release_language = release_language
        self._confirm_done = confirm_done

    async def watch(self, op: dict[str, Any], lang: str, lkey: str, token: str | None = None, deadline: float | None = None) -> None:
        """Poll the language status until it is ready, fails, or the deadline passes.

        Done when the status says ready (``language.ready`` plus completion); failed when it says error or the status call
        keeps failing; released at the op's deadline (default 15 min from the request; after a restart the ORIGINAL deadline
        is kept).
        """
        job_id = self.recorder.job_id
        limit = float(os.environ.get("DUB_CTL_LANG_MAX", "900"))
        poll = float(os.environ.get("DUB_CTL_LANG_POLL", "10"))
        started = time.time()
        deadline = deadline or op.get("deadline") or (started + limit)
        ok = False
        try:
            ok = await self._poll(op, job_id, lang, poll, started, deadline)
        except asyncio.CancelledError:
            self.recorder.terminal(op, False, "cancelled")
            raise
        except BaseException as exc:
            self.recorder.terminal(op, False, f"watcher crashed: {type(exc).__name__}")
        finally:
            self._unclaim(lkey)
            if not ok:
                self._release_language(job_id, lang, op.get("error") or "operation did not complete")
            self._confirm_done(token, ok)  # consumed only on terminal SUCCESS; a failure releases it so the confirmation can retry

    async def _poll(self, op: dict[str, Any], job_id: str, lang: str, poll: float, started: float, deadline: float) -> bool:
        max_errors = int(os.environ.get("DUB_CTL_LANG_MAX_ERRS", "5"))
        max_error_s = float(os.environ.get("DUB_CTL_LANG_MAX_ERR_S", "60"))
        errors, first_error = 0, None
        while time.time() < deadline:
            try:
                track = await ctl_hooks.language_status(job_id, lang)
                errors, first_error = 0, None
            except NotImplementedHook as exc:
                self.recorder.terminal(op, False, f"not_implemented: {exc}")
                return False
            except Exception as exc:
                errors += 1
                first_error = first_error or time.time()
                if errors >= max_errors or time.time() - first_error >= max_error_s:
                    self.recorder.terminal(op, False, f"language status unavailable ({errors} consecutive errors: {type(exc).__name__})")
                    return False
                await asyncio.sleep(poll)
                continue
            if track.get("audio_state") == "ready":
                return self._finish_ready(op, lang, track, started)
            if track.get("audio_state") == "error":
                self.recorder.terminal(op, False, "language preparation failed")
                return False
            await asyncio.sleep(poll)
        self.recorder.terminal(op, False, "timed out waiting for the track (deadline passed; it may still finish in the background)")
        return False

    def _finish_ready(self, op: dict[str, Any], lang: str, track: dict[str, Any], started: float) -> bool:
        op.update(audio_ready=True, lipsync_ready=track.get("lipsync_state") == "ready")
        ready = ("language.ready", {"lang": lang, "ready_intervals": track.get("ready_intervals"), "audio_state": "ready", "lipsync_state": track.get("lipsync_state")})
        done = ("op.done", {"operation_id": op["operation_id"], "action_id": op["action_id"], "elapsed_s": round(time.time() - started, 2)})
        return self.recorder.terminal(op, True, None, [ready, done])
