#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess microbiome gene abundance matrix for one study.")
    parser.add_argument("--base-dir", required=True, help="Base directory containing study folders")
    parser.add_argument("--study", required=True, help="Study folder name")
    parser.add_argument("--condition-col", required=True, help="Metadata column used for case/control labels")
    parser.add_argument("--healthy-label", required=True, help="String label mapped to class 0")
    parser.add_argument("--condition-label", required=True, help="String label mapped to class 1")
    parser.add_argument("--rare-threshold", type=float, default=0.05, help="Per-group prevalence threshold")
    return parser


def main() -> None:
    from microbiome_knockoffs.preprocessing_gene_abundance import (
        PreprocessConfig,
        preprocess_study,
        save_preprocess_outputs,
    )

    args = build_parser().parse_args()

    config = PreprocessConfig(
        base_dir=Path(args.base_dir),
        study_name=args.study,
        condition_col=args.condition_col,
        healthy_label=args.healthy_label,
        condition_label=args.condition_label,
        rare_threshold=args.rare_threshold,
    )

    artifacts = preprocess_study(config)
    x_path, y_path, genes_path = save_preprocess_outputs(config.study_dir, artifacts)

    print("Preprocessing complete.")
    print(f"  X_clean: {x_path}")
    print(f"  y_clean: {y_path}")
    print(f"  genes_clean: {genes_path}")
    print(f"  shape: {artifacts.X_clean.shape}")


if __name__ == "__main__":
    main()
