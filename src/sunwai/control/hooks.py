"""Seams between the control plane and the expression and language-scheduler services. Stubs raise NotImplementedHook (HTTP 501) until the adapters are installed.
Replace by assigning ctl_hooks.expression_resynth / expression_apply / request_language at registration time."""
import asyncio


class NotImplementedHook(Exception):
    pass


async def expression_resynth(job_dir, lang, chunk, segment_id, intent, base_text, tempo_factor, out_candidate_dir):
    """expression. Returns {"ok","audio_hash","duration_ms","spoken_text","style","tempo_factor","error"} (CANDIDATE, never touches active audio)."""
    raise NotImplementedHook("expression pipeline not installed")


async def expression_apply(job_dir, lang, chunk, segment_id, candidate):
    """expression. Remix chunk with the candidate and queue LatentSync; returns {"ok","audio_hash","error"}."""
    raise NotImplementedHook("expression apply not installed")


async def request_language(job_id, lang, from_ms, generation):
    """Language scheduler request_language. Returns {"track":{lang,ready_intervals,audio_state,lipsync_state},"duplicate":bool}."""
    raise NotImplementedHook("language scheduler not installed")


async def regenerate_chunk(job_id, lang, chunk):
    raise NotImplementedHook("chunk regeneration not wired")


async def expression_undo(job_dir, lang, chunk, segment_id):
    """expression undo_revision for revisions created by expression Apply."""
    raise NotImplementedHook("expression undo not installed")


async def language_status(job_id, lang):
    """language scheduler track_readiness (with hooks). -> {lang, ready_intervals, audio_state(absent|preparing|partial|ready|error), lipsync_state}."""
    raise NotImplementedHook("language status not installed")


async def media_path(job_dir, lang, chunk, segment_id, kind, m2_candidate_id, is_head=True, m2_revision=None):
    """-> Path of the A/B audio: kind 'cand' = the candidate preview wav, 'base' = the current line wav. Must lie under the chunk dir."""
    raise NotImplementedHook("media not available")


def live_state(job_dir, lang, chunk, segment_id):
    """Expression live_state (read-only, non-blocking): {"m2_head", "owned", "in_flight"}. Default: no expression service -> nothing is owned."""
    return {"m2_head": None, "owned": False, "in_flight": False}


# One dedicated pool for expression mutations: the worker is tracked as a concurrent.futures.Future (NOT an asyncio Task), so no event-loop
# cancellation (wrapper, task or shutdown) can detach us from a thread that is still mutating live files.
import concurrent.futures as _cf
EXEC = _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="m2")


def _run_coro_in_thread(coro_fn, *a):
    return asyncio.run(coro_fn(*a))


def expression_apply_start(job_dir, lang, chunk, segment_id, candidate):
    """-> concurrent.futures.Future resolving to the apply result dict."""
    return EXEC.submit(_run_coro_in_thread, expression_apply, job_dir, lang, chunk, segment_id, candidate)


def expression_undo_start(job_dir, lang, chunk, segment_id):
    return EXEC.submit(_run_coro_in_thread, expression_undo, job_dir, lang, chunk, segment_id)
