# Microbiome Knockoffs

Modular refactor of the microbiome knockoff pipeline and notebook workflows.

## Scope

This project reorganizes the original workflow into reusable Python modules and thin CLI scripts.

Included refactors:

- `combined_knockoffs_flow.py` -> modular pipeline modules + `scripts/run_knockoffs_pipeline.py`
- `gene_abundance_preprocessing.ipynb` -> `scripts/preprocess_gene_abundance.py`
- `knockoffs_classifier_comparison.ipynb` -> `scripts/compare_knockoffs_classifiers.py`
- `visualizations.ipynb` -> `scripts/generate_visualizations.py`

## Directory Layout

```text
Microbiome_Knockoffs/
  configs/
    defaults.yaml
  scripts/
    run_knockoffs_pipeline.py
    preprocess_gene_abundance.py
    compare_knockoffs_classifiers.py
    generate_visualizations.py
  microbiome_knockoffs/
    contracts.py
    io_data.py
    logging_utils.py
    pipeline_orchestrator.py
    filtering_star.py
    analysis_covariance.py
    analysis_rsp.py
    preprocessing_gene_abundance.py
    evaluation_classifier_comparison.py
    visualization_plots.py
    knockoffs/
      neighbor_index_interface.py
      neighbor_index_faiss.py
      distribution_interface.py
      distribution_hurdle_lgbm.py
      tuning_interface.py
      tuning_optuna_lgbm.py
      generators_base.py
      generators_binary.py
  tests/
    test_rsp_math.py
    test_knockoff_interfaces.py
    test_cli_smoke.py
```

## Extension Points

The generator is intentionally modular so model changes remain local:

- Distribution learning: implement `DistributionLearner` in `distribution_interface.py`
- Hyperparameter tuning: implement `HyperparameterTuner` in `tuning_interface.py`
- Neighbor search backend: implement `NeighborIndex` in `neighbor_index_interface.py`

Default implementations:

- `HurdleLGBMDistribution`
- `OptunaLGBMTuner`
- `FaissHNSWIndex`

## Script Usage

### 1) Preprocessing

```bash
python scripts/preprocess_gene_abundance.py \
  --base-dir /home1/kapachy/Microbium/projects/cMD_downloads \
  --study NielsenHB_2014 \
  --condition-col disease \
  --healthy-label healthy \
  --condition-label IBD \
  --rare-threshold 0.05
```

### 2) Knockoff pipeline

```bash
python scripts/run_knockoffs_pipeline.py \
  --base-dir /home1/kapachy/Microbium/projects/cMD_downloads \
  --study WirbelJ_2018 \
  --sparsity-threshold 0.85 \
  --k-neighbors 40 \
  --target-fdr 0.05 \
  --num-shuffles 20
```

### 3) Classifier comparison

```bash
python scripts/compare_knockoffs_classifiers.py \
  --base-dir /home1/kapachy/Microbium/projects/cMD_downloads \
  --study WirbelJ_2018 \
  --run-folder WirbelJ_2018_sparse085_k40_20260324_153038 \
  --k-grid-points 20 \
  --k-end 5000 \
  --random-trials 20 \
  --save-csv
```

Classifier, metric, and feature-selection toggles are configured at the top of
`scripts/compare_knockoffs_classifiers.py` (script-level config, no extra CLI required).

Default enabled feature-selection methods:

- `random_raw`: random K features from raw MTX (`{study}_gene_families.mtx`)
- `bh_raw`: BH-ranked top-K from raw MTX
- `random_clean`: random K from `X_clean`
- `random_filtered`: random K from `X_filtered`
- `knockoff_topk`: top-K features from knockoff-selected ranking (`rsp_results.npy`)
- `bacteria_constant`: bacteria aggregated baseline (constant feature count line)

### 4) Visualizations

```bash
python scripts/generate_visualizations.py \
  --base-dir /home1/kapachy/Microbium/projects/cMD_downloads \
  --studies HMP_2012 NielsenHB_2014 WirbelJ_2018 HMP_2019_ibdmdb \
  --study-run HMP_2012:RUN_A \
  --study-run NielsenHB_2014:RUN_B \
  --study-run WirbelJ_2018:RUN_C \
  --study-run HMP_2019_ibdmdb:RUN_D \
  --mode all
```

### 5) Legacy Flow Compatibility Test (WirbelJ_2018 example)

Run the new modular pipeline with legacy-equivalent parameters and compare outputs against
the historical reference run folder:

```bash
python scripts/test_legacy_flow_compat.py \
  --base-dir /home1/kapachy/Microbium/projects/cMD_downloads \
  --study WirbelJ_2018 \
  --reference-run-dir /home1/kapachy/Microbium/projects/cMD_downloads/WirbelJ_2018/runs/WirbelJ_2018_sparse085_k40_20260324_153038
```

If you already have a new run folder and only want comparison:

```bash
python scripts/test_legacy_flow_compat.py \
  --compare-only \
  --reference-run-dir /home1/kapachy/Microbium/projects/cMD_downloads/WirbelJ_2018/runs/WirbelJ_2018_sparse085_k40_20260324_153038 \
  --candidate-run-dir /path/to/new/run/folder
```

## Testing

Run from project root:

```bash
pytest -q
```

The test suite covers:

- RSP math utilities
- Generator dependency-injection seams (distribution/tuning/index)
- CLI smoke (`--help`) for scripts
