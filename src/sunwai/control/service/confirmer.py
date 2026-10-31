"""Server-issued confirmation tokens for actions that commit, spend GPU time or fork."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any

from sunwai.control.codec import canon, sha
from sunwai.control.errors import CtlError


class Confirmer:
    """Issues and redeems confirmation tokens.

    Tokens are bound to ``sha256(canonical{job, type, args, expected_revision})`` and expire after a TTL. A token is consumed
    only when the confirmed work SUCCEEDS: it is "in use" while running (so it cannot be double-spent) and released again on
    failure so a transient error can be retried.
    """

    def __init__(self) -> None:
        """Create an empty token table with a fresh random secret."""
        self.secret = secrets.token_bytes(32)
        self.live: dict[str, list[Any]] = {}  # token -> [binding, expiry, in_use]

    @staticmethod
    def bind(job_id: str, typ: str, args: dict[str, Any], expected: int | None) -> str:
        """Return the binding hash of a proposal."""
        return sha(canon({"job": job_id, "type": typ, "args": args, "expected": expected}))

    def issue(self, job_id: str, typ: str, args: dict[str, Any], expected: int | None, ttl: float) -> tuple[str, float]:
        """Issue a token for a proposal and drop expired ones.

        Returns:
            ``(token, expiry_epoch_seconds)``.
        """
        binding = self.bind(job_id, typ, args, expected)
        token = hmac.new(self.secret, (binding + secrets.token_hex(8)).encode(), hashlib.sha256).hexdigest()[:40]
        now = time.time()
        for stale in [t for t, entry in self.live.items() if entry[1] < now]:
            self.live.pop(stale, None)
        self.live[token] = [binding, now + ttl, False]
        return token, now + ttl

    def hold(self, token: str | None, job_id: str, typ: str, args: dict[str, Any], expected: int | None) -> None:
        """Mark a token in use for exactly this proposal.

        Raises:
            CtlError: 403 ``bad_confirmation`` when the token is missing, expired, in use, or bound to a different proposal.
        """
        entry = self.live.get(token or "")
        if (
            not entry
            or entry[1] < time.time()
            or entry[2]
            or not hmac.compare_digest(entry[0], self.bind(job_id, typ, args, expected))
        ):
            raise CtlError(
                403,
                "bad_confirmation",
                "confirmation token missing, expired, in use, or not bound to this exact proposal",
            )
        entry[2] = True

    def done(self, token: str | None, ok: bool) -> None:
        """Finish a held token: consume it on success, release it for a retry on failure."""
        if not token or token not in self.live:
            return
        if ok:
            self.live.pop(token, None)
        else:
            self.live[token][2] = False
