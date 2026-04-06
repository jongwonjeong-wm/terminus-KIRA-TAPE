#!/bin/bash
# Run TAPE tests with LLM logging
# Usage:
#   ANTHROPIC_API_KEY=<your-key> bash tests/run_tape_test.sh
#   ANTHROPIC_API_KEY=<your-key> bash tests/run_tape_test.sh --verbose

module load conda && conda activate tb

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running}"

python tests/test_tape.py "$@"

if [ -f tests/tape_llm_log.jsonl ]; then
    echo ""
    echo "LLM log saved to: tests/tape_llm_log.jsonl"
    echo "View with: cat tests/tape_llm_log.jsonl | python -m json.tool --no-ensure-ascii"
    echo "Or per-call: jq -s '.[]' tests/tape_llm_log.jsonl"
fi
