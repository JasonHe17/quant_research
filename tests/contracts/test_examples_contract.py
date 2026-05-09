from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_real_data_smoke_example_has_cli_help() -> None:
    script = Path("examples/real_data_smoke.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--workspace-root" in result.stdout
    assert "--quant-dataset-root" in result.stdout
