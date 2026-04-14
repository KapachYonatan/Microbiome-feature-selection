from __future__ import annotations

import re

from microbiome_knockoffs.contracts import RunConfig
from microbiome_knockoffs.io_data import create_run_directory


def test_create_run_directory_uses_study_timestamp_name(tmp_path):
    study_name = "WirbelJ_2018"
    (tmp_path / study_name).mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        study_name=study_name,
        base_dir=tmp_path,
        sparsity_threshold=0.91,
        k_neighbors=77,
    )

    run_dir = create_run_directory(config)

    assert run_dir.parent == tmp_path / study_name / "runs"
    assert run_dir.is_dir()
    assert re.match(rf"^{re.escape(study_name)}_\d{{8}}_\d{{6}}$", run_dir.name)
