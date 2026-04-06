#!/usr/bin/env bash
set -euo pipefail

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running}"

RUNS=1

for i in $(seq 1 $RUNS); do
    echo "========================================"
    echo "Run $i / $RUNS - Starting at $(date)"
    echo "========================================"

    harbor run \
        --agent-import-path "terminus_kira.terminus_kira:TerminusKira" \
        -d "terminal-bench@2.0" \
        -m "anthropic/claude-haiku-4-5-20251001" \
        -e runloop \
        --n-concurrent 4

    echo "========================================"
    echo "Run $i / $RUNS - Finished at $(date)"
    echo "========================================"
    echo ""
done

echo "All $RUNS runs completed!"

# distribution-search
# 