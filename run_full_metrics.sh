#!/bin/bash

# Multi-Model Benchmark Orchestrator
# This will execute profile_runner.py across both massive architectures sequentially
# parsing the Vanilla, SSD, and TurboQuant variants over deep context.

CONTEXTS="512,40000,100000"

echo "============================================="
echo " Starting Full Suite Re-Benchmarking         "
echo " Context Windows: $CONTEXTS                  "
echo "============================================="

# 1. Qwen 3.6 35B
echo "[1/2] Evaluating Qwen3.6-35B-A3B-4bit..."
python3 -u scripts/profiling/profile_runner.py \
    --model mlx-community/Qwen3.6-35B-A3B-4bit \
    --contexts "$CONTEXTS" \
    --out qwen_metrics_final.md

echo "✅ Qwen3.6-35B Complete."
echo ""

# 2. Gemma 4 26B
echo "[2/2] Evaluating gemma-4-26b-a4b-it-4bit..."
python3 -u scripts/profiling/profile_runner.py \
    --model mlx-community/gemma-4-26b-a4b-it-4bit \
    --contexts "$CONTEXTS" \
    --out gemma_metrics_final.md

echo "✅ Gemma4-26B Complete."
echo "============================================="
echo " All pipeline metrics complete. Check markdown artifacts."
echo "============================================="
