"""General utilities: prevent the computer from sleeping and exact version of the code."""

from __future__ import annotations

import contextlib
import functools
import os
import subprocess
import sys

_DEPTH = 0  # nesting of keep_awake (only the outermost one changes the state)


@contextlib.contextmanager
def keep_awake(enabled: bool = True):
    """Prevent the computer from going to sleep from inactivity while the block lasts.

    On Windows it uses ``SetThreadExecutionState``; on macOS it launches ``caffeinate``.
    It does not prevent a forced sleep (closing the lid, switching off by hand). On other
    systems it does nothing.

        with keep_awake():
            mc = monte_carlo(study, n=30)
    """
    global _DEPTH
    if not enabled or _DEPTH > 0:
        _DEPTH += 1
        try:
            yield
        finally:
            _DEPTH -= 1
        return
    _DEPTH += 1
    proc = None
    try:
        if sys.platform == "win32":
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        elif sys.platform == "darwin":
            try:
                proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            except OSError:
                proc = None
        yield
    finally:
        _DEPTH -= 1
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        elif proc is not None:
            proc.terminate()


def keeps_awake(fn):
    """Decorator: the function runs inside ``keep_awake``."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with keep_awake():
            return fn(*a, **k)
    return wrapper


def save_checkpoint(path: str, obj) -> None:
    """Save ``obj`` with pickle atomically (does not leave a half-written file if interrupted)."""
    import pickle
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


def load_checkpoint(path: str | None, signature: dict):
    """Load a checkpoint if it exists and was created with the same ``signature``; otherwise None."""
    import pickle
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception:
        return None
    return obj if obj.get("signature") == signature else None


def code_version() -> dict:
    """Exact version of the code: package version, git commit and whether there are uncommitted changes.

    ``label`` is a short text such as ``"0.1.0 (commit abc1234)"`` or
    ``"0.1.0 (commit abc1234 + cambios sin commit)"``. Outside a git
    repository it only gives the package version.
    """
    from . import __version__
    info = {"version": __version__, "commit": None, "dirty": None}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def git(*args):
        r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    try:
        info["commit"] = git("rev-parse", "--short", "HEAD")
        if info["commit"]:
            info["dirty"] = bool(git("status", "--porcelain", "--untracked-files=no", "--", "solarff"))
    except (OSError, subprocess.SubprocessError):
        pass
    label = __version__
    if info["commit"]:
        label += f" (commit {info['commit']}" + (" + cambios sin commit" if info["dirty"] else "") + ")"
    info["label"] = label
    return info
