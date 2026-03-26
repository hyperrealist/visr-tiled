import subprocess
import sys

from visr_tiled import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "visr_tiled", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
