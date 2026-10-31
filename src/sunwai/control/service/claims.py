"""Synthesis slots: at most a few concurrent GPU jobs per job (store) and globally (guard)."""

from __future__ import annotations

from sunwai.control.errors import CtlError
from sunwai.control.guard import Guard
from sunwai.control.store import JobStore


class SynthClaims:
    """Claims and releases synthesis slots, keeping the per-job map and the global counter consistent."""

    def __init__(self, guard: Guard | None) -> None:
        """Bind the optional global guard."""
        self.guard = guard

    def claim(self, store: JobStore, key: str, max_synth: int) -> None:
        """Claim the slot ``key`` of a job.

        Raises:
            CtlError: 409 ``busy`` when the key is taken, 429 when the per-job or global cap is reached.
        """
        with store.lock:
            if key in store.running:
                raise CtlError(409, "busy", "this line already has an operation running")
            if len(store.running) >= max_synth:
                raise CtlError(
                    429,
                    "one_synthesis_per_job",
                    "another synthesis is running for this job; try again when it finishes",
                    retry_after=5,
                )
            if self.guard:
                self.guard.synth_acquire()
            store.running[key] = "pending"

    def reclaim(self, store: JobStore, key: str, operation_id: str) -> None:
        """Re-claim the slot of an operation that survived a restart.

        Raises:
            CtlError: 409 ``busy`` when the key is already taken; 429 from the global guard.
        """
        with store.lock:
            if key in store.running:
                raise CtlError(409, "busy", "slot taken")
            if self.guard:
                self.guard.synth_acquire()
            store.running[key] = operation_id

    def release(self, store: JobStore, key: str) -> None:
        """Release the slot ``key`` if it is held."""
        with store.lock:
            if store.running.pop(key, None) is not None and self.guard:
                self.guard.synth_release()
