#!/bin/bash
# Extract final iteration output and evaluation results from a training log.
# Usage: bash tools/extract_results.sh [work_dir]
# Example: bash tools/extract_results.sh work_dirs/bevformer_project

WORK_DIR="${1:-work_dirs/bevformer_project}"
LOG=$(ls -t "$WORK_DIR"/*.log 2>/dev/null | head -1)
OUT="$WORK_DIR/final_results.txt"

if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    echo "No log file found in $WORK_DIR"
    exit 1
fi

echo "# BEVFormerFusion Training Results" > "$OUT"
echo "# Log: $LOG" >> "$OUT"
echo "# Date: $(date)" >> "$OUT"
echo "" >> "$OUT"

echo "## Last iteration:" >> "$OUT"
grep -E "^.*Iter \[" "$LOG" | tail -1 >> "$OUT"
echo "" >> "$OUT"

echo "## Evaluation Results:" >> "$OUT"
grep -A 100 "Saving metrics to:" "$LOG" | head -120 >> "$OUT" 2>/dev/null
echo "" >> "$OUT"

echo "## Per-class Results:" >> "$OUT"
grep -A 15 "Per-class results:" "$LOG" | tail -15 >> "$OUT" 2>/dev/null

echo "Results saved to: $OUT"
