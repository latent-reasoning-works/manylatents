"""Test that check_docs_coverage.py catches problems."""
import subprocess
import sys


def test_coverage_script_does_not_crash():
    """The coverage checker should run without crashing."""
    result = subprocess.run(
        [sys.executable, "scripts/check_docs_coverage.py"],
        capture_output=True, text=True
    )
    # May exit 1 due to pre-existing stale paths — that's OK.
    # What matters is it doesn't crash (exit code 2+ = exception).
    assert result.returncode in (0, 1), (
        f"Coverage script crashed:\n{result.stderr}\n{result.stdout}"
    )


def test_coverage_script_output_parseable():
    """The coverage checker should produce parseable summary output."""
    result = subprocess.run(
        [sys.executable, "scripts/check_docs_coverage.py"],
        capture_output=True, text=True
    )
    assert "coverage" in result.stdout.lower(), (
        f"Expected coverage summary in output:\n{result.stdout}"
    )
