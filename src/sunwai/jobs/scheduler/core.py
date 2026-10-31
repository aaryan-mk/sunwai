"""The add-language scheduler: idempotent track requests, the global in-flight cap and crash recovery.

Decisions live in :mod:`sunwai.jobs.scheduler.requests` and :mod:`sunwai.jobs.scheduler.ordering`; this class owns
the in-flight state and sequences the filesystem transaction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sunwai.jobs.scheduler import requests, store
from sunwai.jobs.scheduler.config import enabled, max_inflight
from sunwai.jobs.scheduler.errors import Hooks, LangSchedError
from sunwai.jobs.scheduler.focus import focus_adapter
from sunwai.jobs.scheduler.ordering import plan_order
from sunwai.jobs.scheduler.readiness import track_readiness
from sunwai.jobs.scheduler.reconcile import LEDGER
from sunwai.jobs.scheduler.requests import TrackRequest
from sunwai.jobs.scheduler.transaction import JobSnapshot, merge_lang_into_job
from sunwai.paths import DATA_DIR

_FINISHED_AUDIO = ("ready", "error", "absent")
_OPEN_PHASES = ("scheduling", "scheduled", "abandoned")
TrackKey = tuple[str, str]  # (job_dir, lang)


@dataclass
class _Outcome:
    """Result of applying a request under the lock."""

    created: bool
    reconcile: dict[str, Any] | None
    meta: dict[str, Any]


class LanguageScheduler:
    """Schedules extra target languages onto existing jobs and tracks which are still preparing."""

    def __init__(self) -> None:
        """Create a scheduler with no in-flight tracks."""
        self.inflight: dict[TrackKey, float] = {}  # in-memory hint of preparing tracks (disk is authoritative)

    # ------------------------------------------------------------------ in-flight accounting
    @staticmethod
    def _job_alive(hooks: Hooks, job_dir: Path) -> bool | None:
        if hooks.job_active is None:
            return None
        try:
            return bool(hooks.job_active(Path(job_dir).name))
        except Exception:  # the liveness hook is advisory
            return None

    def _frees_cap(self, job_dir: Path, track: dict[str, Any], hooks: Hooks) -> bool:
        return requests.releases_cap(track, self._job_alive(hooks, job_dir))

    def release_track(self, job_dir: Path, lang: str, reason: str = "") -> bool:
        """Durably mark a track abandoned (failed, cancelled or timed out).

        It stops counting toward the cap only once the job's task is dead; a later request for the same
        language re-schedules it.

        Returns:
            True when the track was marked, False when it is unknown or already abandoned.
        """
        job_dir = Path(job_dir)
        with store.LOCK:
            tracks_path = job_dir / "tracks.json"
            state = store.read_json(tracks_path)
            track = store.track_map(state).get(lang)
            if not isinstance(track, dict) or track.get("phase") == "abandoned":
                return False
            track["phase"], track["abandoned_reason"], track["abandoned_ts"] = "abandoned", reason[:200], time.time()
            store.atomic_json(tracks_path, state)
        self.inflight.pop((str(job_dir), lang), None)
        return True

    def inflight_on_disk(
        self, root: Path, exclude: TrackKey | None = None, hooks: Hooks | None = None
    ) -> set[TrackKey]:
        """Return the tracks still preparing according to ``tracks.json`` files under ``root``.

        Derived from disk so a crash cannot bypass the global cap; the in-memory dict is only a hint.
        """
        hooks = hooks or Hooks()
        out: set[TrackKey] = set()
        try:
            files = list(Path(root).glob("*/tracks.json"))
        except OSError:
            return out
        for tracks_file in files:
            job_dir = tracks_file.parent
            for lang, track in store.track_map(store.read_json(tracks_file)).items():
                if not isinstance(track, dict) or track.get("phase") not in _OPEN_PHASES:
                    continue
                if (str(job_dir), lang) == exclude:
                    continue
                if track_readiness(job_dir, lang)["audio_state"] in _FINISHED_AUDIO:
                    continue
                if self._frees_cap(job_dir, track, hooks):
                    continue
                out.add((str(job_dir), lang))
        return out

    def _prune_inflight_hints(self, hooks: Hooks) -> None:
        for job_str, lang in list(self.inflight):
            job_dir = Path(job_str)
            track = store.track_map(store.read_json(job_dir / "tracks.json")).get(lang) or {}
            finished = track_readiness(job_dir, lang)["audio_state"] in _FINISHED_AUDIO
            if finished or self._frees_cap(job_dir, track, hooks):
                self.inflight.pop((job_str, lang), None)

    # ------------------------------------------------------------------ crash recovery
    def recover_job(self, job_dir: Path, hooks: Hooks) -> list[dict[str, Any]]:
        """Roll forward or drop tracks left in phase ``scheduling`` by a crash.

        A language already merged into ``job.json`` is rolled forward (``ensure_running`` is idempotent, then the
        phase is committed); one that never reached ``job.json`` has its intent dropped. The server's own startup
        reconcile resumes only queued/running jobs, so a finished job with a newly merged language would otherwise
        never be dubbed.

        Returns:
            One action record per recovered track.
        """
        job_dir = Path(job_dir)
        actions: list[dict[str, Any]] = []
        if not enabled():
            return actions  # kill switch: intents stay intact, nothing is (re)started
        with store.LOCK:
            tracks_path = job_dir / "tracks.json"
            state = store.read_json(tracks_path)
            tracks = store.track_map(state)
            pending = [lg for lg, t in tracks.items() if isinstance(t, dict) and t.get("phase") == "scheduling"]
            if not pending:
                return actions
            meta = store.read_json(job_dir / "job.json")
            for lang in pending:
                actions.append(self._recover_track(job_dir, meta, tracks, lang, hooks))
            try:
                store.atomic_json(tracks_path, state)
                for act in actions:
                    if act["action"] != "roll_forward_failed":
                        LEDGER.clear(job_dir, act["lang"])
            except OSError as exc:
                actions.append({"action": "persist_failed", "error": f"{type(exc).__name__}: {exc}"})
        return actions

    @staticmethod
    def _recover_track(
        job_dir: Path, meta: dict[str, Any], tracks: dict[str, Any], lang: str, hooks: Hooks
    ) -> dict[str, Any]:
        if lang not in (meta.get("langs") or []):
            tracks.pop(lang, None)
            return {"lang": lang, "action": "intent_dropped"}
        try:
            if not hooks.ensure_running:
                raise RuntimeError("no ensure_running hook: cannot schedule")
            hooks.ensure_running(meta.get("job_id") or job_dir.name)
            tracks[lang]["phase"] = "scheduled"
        except Exception as exc:
            return {"lang": lang, "action": "roll_forward_failed", "error": f"{type(exc).__name__}: {exc}"}
        return {"lang": lang, "action": "rolled_forward"}

    def recover_all(self, root: Path, hooks: Hooks) -> list[dict[str, Any]]:
        """Run :meth:`recover_job` for every job directory under ``root`` that has a ``tracks.json``."""
        out: list[dict[str, Any]] = []
        if not enabled():
            return out
        try:
            for tracks_file in sorted(Path(root).glob("*/tracks.json")):
                for act in self.recover_job(tracks_file.parent, hooks):
                    out.append({"job": tracks_file.parent.name, **act})
        except Exception as exc:
            out.append({"action": "recover_all_failed", "error": f"{type(exc).__name__}: {exc}"})
        return out

    # ------------------------------------------------------------------ track requests
    def request_track(
        self,
        job_dir: Path,
        lang: str,
        from_ms: Any,
        playback_generation: Any,
        *,
        watching_lang: str | None = None,
        hooks: Hooks | None = None,
    ) -> dict[str, Any]:
        """Request that ``lang`` be dubbed for a job, idempotently.

        The same ``(job, lang)`` again returns ``created=False`` and only re-prioritises; a generation older than
        the newest seen is rejected (409 ``obsolete_generation``) so stale switch responses never act.

        Raises:
            LangSchedError: The request is disabled, malformed, unsupported, over a cap or could not be persisted.
        """
        hooks = hooks or Hooks()
        if not enabled():
            raise LangSchedError(503, "disabled", "add-language is disabled (DUB_ADD_LANG)")
        job_dir = Path(job_dir)
        if not store.read_json(job_dir / "job.json").get("langs"):
            raise LangSchedError(404, "job_not_found", "job not found")
        req = requests.parse_request(lang, from_ms, playback_generation)
        self._check_supported(req.lang, hooks)
        self.recover_job(job_dir, hooks)  # a previous crash/failed commit for this job is rolled forward first
        with store.LOCK:
            outcome = self._apply(job_dir, req, watching_lang, hooks)
        chunks = store.job_chunks(job_dir)
        job_id = outcome.meta.get("job_id") or job_dir.name
        return {
            "lang": req.lang,
            "created": outcome.created,
            "reconcile": outcome.reconcile,
            "from_ms": req.from_ms,
            "generation": req.generation,
            "order": plan_order(chunks, req.from_ms),
            "focus": focus_adapter(job_dir, job_id, watching_lang, req.lang, req.from_ms, hooks),
            "readiness": track_readiness(job_dir, req.lang, hooks),
        }

    @staticmethod
    def _check_supported(lang: str, hooks: Hooks) -> None:
        if hooks.target_codes:
            codes = hooks.target_codes()
        else:
            codes = {t.get("code") for t in store.read_json(DATA_DIR / "languages.json").get("targets", [])}
        if codes and lang not in codes:
            raise LangSchedError(422, "unsupported", f"unsupported language {lang}")

    def _apply(self, job_dir: Path, req: TrackRequest, watching_lang: str | None, hooks: Hooks) -> _Outcome:
        """Validate, then schedule or touch the track, then commit ``tracks.json``. Caller holds the lock."""
        meta = store.read_json(job_dir / "job.json")
        langs = list(meta["langs"])
        tracks_path = job_dir / "tracks.json"
        state = store.read_json(tracks_path)
        if not isinstance(state.get("tracks"), dict):
            state = {"gen": 0, "tracks": {}}
        if req.generation < int(state.get("gen", 0)):
            raise LangSchedError(409, "obsolete_generation", f"playback_generation {req.generation} < {state['gen']}")
        prev = state["tracks"].get(req.lang)
        phase = (prev or {}).get("phase")
        is_new = req.lang not in langs
        if not is_new and phase == "scheduling":
            raise LangSchedError(
                503, "recovery_pending", f"{req.lang} was merged but could not be scheduled yet; retry shortly"
            )
        retry = not is_new and phase == "abandoned"
        if is_new or retry:
            self._admit(job_dir, req.lang, langs if is_new else None, hooks)
            self._schedule(job_dir, req, watching_lang, langs, is_new, state, meta, hooks)
            created = prev is None or retry
        else:
            created = prev is None
            state["gen"] = max(int(state.get("gen", 0)), req.generation)
            state["tracks"][req.lang] = requests.touched_record(prev, req, watching_lang, time.time())
        reconcile = self._commit(job_dir, req.lang, state, scheduled=is_new or retry)
        return _Outcome(created, reconcile, meta)

    def _admit(self, job_dir: Path, lang: str, new_langs: list[str] | None, hooks: Hooks) -> None:
        """Enforce target, voice-plan, global-cap and busy limits before anything is mutated.

        ``new_langs`` is the job's current language list for a brand-new language, None for a retry.
        """
        if new_langs is not None:
            if len(new_langs) + 1 > hooks.max_targets:
                raise LangSchedError(
                    422, "max_targets", f"a job may have at most {hooks.max_targets} target languages (has {new_langs})"
                )
            ok, why = self.check_voice_plan(job_dir, lang)
            if not ok:
                raise requests.voice_plan_error(why)
        self._prune_inflight_hints(hooks)
        key = (str(job_dir), lang)
        others = (self.inflight_on_disk(job_dir.parent, exclude=key, hooks=hooks) | set(self.inflight)) - {key}
        if len(others) >= max_inflight():
            raise LangSchedError(
                429,
                "busy",
                f"{len(others)} added-language track(s) already preparing "
                f"(limit {max_inflight()}); try again when it finishes",
            )
        if hooks.busy is not None and hooks.busy():
            raise LangSchedError(429, "busy", "the demo is busy with other dubbing jobs; try again in a few minutes")

    @staticmethod
    def check_voice_plan(job_dir: Path, lang: str) -> tuple[bool, str]:
        """Check the job's frozen voice plan admits ``lang`` (never mutates the plan)."""
        path = Path(job_dir) / "speakers_global.json"
        if not path.is_file():
            return False, "planner_not_ready"
        return requests.voice_plan_verdict(store.read_json(path), lang)

    def _schedule(
        self,
        job_dir: Path,
        req: TrackRequest,
        watching_lang: str | None,
        langs: list[str],
        is_new: bool,
        state: dict[str, Any],
        meta: dict[str, Any],
        hooks: Hooks,
    ) -> None:
        """Write-ahead transaction: durable intent, merge into ``job.json``, ensure running, mark scheduled.

        A crash anywhere leaves ``phase=scheduling`` on disk, which counts toward the cap and which
        :meth:`recover_job` rolls forward. A failure rolls everything back.
        """
        key = (str(job_dir), req.lang)
        snapshot = JobSnapshot(job_dir, req.lang)
        state["tracks"][req.lang] = requests.new_intent(req, watching_lang, langs, time.time())
        state["gen"] = max(int(state.get("gen", 0)), req.generation)
        try:
            store.atomic_json(job_dir / "tracks.json", state)
        except Exception as exc:  # nothing mutated yet
            raise LangSchedError(
                500, "persist_failed", f"could not persist the scheduling intent: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            if is_new:
                (hooks.merge_lang or merge_lang_into_job)(job_dir, req.lang)
            if hooks.ensure_running:
                hooks.ensure_running(meta.get("job_id") or job_dir.name)
        except BaseException as exc:
            snapshot.restore(restore_job=is_new)
            self.inflight.pop(key, None)
            if isinstance(exc, (KeyboardInterrupt, SystemExit, LangSchedError)):
                raise
            raise LangSchedError(
                502,
                "schedule_failed",
                f"could not schedule {req.lang}: {type(exc).__name__}: {exc}; no changes were kept",
            ) from exc
        self.inflight[key] = time.time()
        state["tracks"][req.lang]["phase"] = "scheduled"

    @staticmethod
    def _commit(job_dir: Path, lang: str, state: dict[str, Any], scheduled: bool) -> dict[str, Any] | None:
        """Persist ``tracks.json``; return a reconcile note if the commit failed after work was scheduled."""
        try:
            store.atomic_json(job_dir / "tracks.json", state)
        except Exception as exc:
            if not scheduled:  # nothing was scheduled by this call: plain structured failure
                raise LangSchedError(
                    500, "persist_failed", f"could not persist track state: {type(exc).__name__}: {exc}"
                ) from exc
            # Work is scheduled and the durable intent (phase=scheduling) is on disk: report it;
            # recovery rolls it forward.
            return LEDGER.mark(job_dir, lang, f"tracks.json commit failed: {type(exc).__name__}: {exc}")
        try:
            LEDGER.clear(job_dir, lang)
        except OSError:
            pass
        return None
