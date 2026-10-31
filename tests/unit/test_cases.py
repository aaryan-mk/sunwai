"""Runs the self-checking case scripts in ``tests/unit/cases`` as subprocesses.

Each script prints PASS/FAIL lines and exits non-zero on failure; running them out of process keeps their
module-level environment changes away from the rest of the suite.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import pytest

HERE = Path(__file__).parent / "cases"
SRC = Path(__file__).parents[2] / "src" / "sunwai"

SCRIPTS = {
    "expression": ["case_expression.py"],
    "control_api": ["case_control_api.py"],
    "fit_guards": ["case_fit_guards.py", str(SRC / "planning" / "planner.py"), str(SRC / "pipeline" / "engine.py")],
}


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_case_script(name: str, tmp_path: Path) -> None:
    """Each case script must exit 0."""
    argv = SCRIPTS[name]
    result = subprocess.run(
        [sys.executable, str(HERE / argv[0]), *argv[1:]],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def test_pasteback_and_gate_cases() -> None:
    """The unittest module for the active-speaker gate and ROI paste-back hooks passes."""
    pytest.importorskip("cv2")
    pytest.importorskip("python_speech_features")
    pytest.importorskip("scipy")
    sys.path.insert(0, str(HERE))
    try:
        import case_pasteback_gate  # type: ignore[import-not-found]

        suite = unittest.defaultTestLoader.loadTestsFromModule(case_pasteback_gate)
        result = unittest.TextTestRunner(verbosity=0).run(suite)
    finally:
        sys.path.remove(str(HERE))
    assert result.wasSuccessful()
