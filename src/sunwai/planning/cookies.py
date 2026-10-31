"""Private, short-lived copy of the operator's YouTube cookie file.

yt-dlp rewrites its cookie jar, so the master file is never handed to it: a mode-600 copy is made in a private
directory and deleted after the download. Contents are never read into Python strings, printed or logged.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import stat
import tempfile
import time

log = logging.getLogger("planner")

_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_STALE_AFTER_S = 3600


def _tmp_dir() -> str:
    """Directory that holds the private copies (``YTDLP_COOKIES_TMPDIR`` or a per-user temp dir)."""
    return os.environ.get("YTDLP_COOKIES_TMPDIR") or os.path.join(tempfile.gettempdir(), "sunwai_yt_cookies")


def pid_alive(pid: int) -> bool:
    """True when a process with this pid exists (or may exist: permission errors count as alive)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _open_source(path: str) -> int | None:
    """Open the cookie file once with ``O_NOFOLLOW | O_NONBLOCK`` (a FIFO cannot hang us) and validate the descriptor.

    Returns the descriptor, or ``None`` when the file is missing, not a regular file owned by us, or not mode 0600.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _O_CLOEXEC)
    except OSError:
        return None
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
        os.close(fd)
        return None
    if stat.S_IMODE(st.st_mode) != 0o600:
        log.warning("[1] YouTube cookie file ignored: its mode is %o, it must be exactly 600", stat.S_IMODE(st.st_mode))
        os.close(fd)
        return None
    return fd


def _sweep_stale_copies(dir_fd: int) -> None:
    """Delete copies left by killed planners: dead pid AND older than an hour (all ops relative to the dir fd)."""
    now = time.time()
    for name in os.listdir(dir_fd):
        parts = name.split("_")
        is_copy = name.startswith("ytc_") and len(parts) > 2 and parts[1].isdigit()
        if not (is_copy and not pid_alive(int(parts[1]))):
            continue
        try:
            entry = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
            if stat.S_ISREG(entry.st_mode) and now - entry.st_mtime > _STALE_AFTER_S:
                os.unlink(name, dir_fd=dir_fd)
        except OSError:
            pass


def _open_private_dir(path: str) -> int | None:
    """Create/open the private directory (never a symlink, owned by us, mode 0700); ``None`` if it is not ours."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    dir_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _O_CLOEXEC)
    info = os.fstat(dir_fd)
    if info.st_uid != os.getuid():
        log.warning("[1] cookie temp dir is not owned by us: downloading without cookies")
        os.close(dir_fd)
        return None
    if stat.S_IMODE(info.st_mode) != 0o700:
        os.fchmod(dir_fd, 0o700)
    return dir_fd


def _copy_into(dir_fd: int, source_fd: int) -> str:
    """Copy from the open source descriptor into a new exclusive mode-600 file; return its name."""
    name = f"ytc_{os.getpid()}_{secrets.token_hex(8)}.txt"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _O_CLOEXEC
    target_fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
    try:
        os.fchmod(target_fd, 0o600)
        with os.fdopen(target_fd, "wb") as out, os.fdopen(source_fd, "rb", closefd=False) as src:
            shutil.copyfileobj(src, out)
    except BaseException:
        try:
            os.unlink(name, dir_fd=dir_fd)  # a half-written secret copy never stays
        except OSError:
            pass
        raise
    return name


def private_cookie_copy() -> str | None:
    """Make a private mode-600 COPY of the YouTube cookie file named by ``YTDLP_COOKIES``.

    Returns:
        The copy's path (the caller deletes it), or ``None`` when no usable cookie file is configured, in which
        case no cookies are used.
    """
    source_path = (os.environ.get("YTDLP_COOKIES") or "").strip()
    source_fd = _open_source(source_path)
    if source_fd is None:
        return None
    dir_fd = None
    try:
        directory = _tmp_dir()
        dir_fd = _open_private_dir(directory)
        if dir_fd is None:
            return None
        _sweep_stale_copies(dir_fd)
        return os.path.join(directory, _copy_into(dir_fd, source_fd))
    except Exception as exc:
        log.warning("[1] cannot make a private cookie copy (%s): downloading without cookies", type(exc).__name__)
        return None
    finally:
        if dir_fd is not None:
            os.close(dir_fd)
        os.close(source_fd)
