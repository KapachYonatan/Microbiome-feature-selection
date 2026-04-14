from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_help(script_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_commands_exit_zero():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "scripts" / "run_knockoffs_pipeline.py",
        root / "scripts" / "preprocess_gene_abundance.py",
        root / "scripts" / "compare_knockoffs_classifiers.py",
        root / "scripts" / "lgbm_feature_learning_sandbox.py",
        root / "scripts" / "generate_visualizations.py",
        root / "scripts" / "test_legacy_flow_compat.py",
    ]

    for script in scripts:
        result = _run_help(script)
        assert result.returncode == 0, f"{script} failed: {result.stderr}"
        assert "usage:" in result.stdout.lower()
