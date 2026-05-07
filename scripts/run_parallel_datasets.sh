#!/usr/bin/env bash
# Compare knockoffs classifiers over multiple datasets, max 4 in parallel.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT="$SCRIPT_DIR/compare_knockoffs_classifiers.py"
BASE_DIR="/home/kapachy/Microbium/projects/cMD_downloads"
LOG_DIR="$SCRIPT_DIR/../logs"

mkdir -p "$LOG_DIR"

DATASETS=(
    BrooksB_2017
    HallAB_2017
    JieZ_2017
    KosticAD_2015
    LiJ_2017
    QinJ_2012
    RubelMA_2020
    ZhuF_2020
)

run_study() {
    local study="$1"
    local log_file="$LOG_DIR/${study}_compare.log"
    
    # Find the most recent run folder for this study
    local run_folder=$(ls -dt "$BASE_DIR/$study/runs"/*/ 2>/dev/null | head -1 | xargs basename)
    
    if [ -z "$run_folder" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP  $study (no runs found)"
        return 1
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $study (run: $run_folder)"
    if python -u "$COMPARE_SCRIPT" \
        --base-dir "$BASE_DIR" \
        --study "$study" \
        --run-folder "$run_folder" \
        > "$log_file" 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $study"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL  $study (see $log_file)"
    fi
}

export -f run_study
export COMPARE_SCRIPT BASE_DIR LOG_DIR

printf '%s\n' "${DATASETS[@]}" | xargs -P 4 -I {} bash -c 'run_study "$@"' _ {}
