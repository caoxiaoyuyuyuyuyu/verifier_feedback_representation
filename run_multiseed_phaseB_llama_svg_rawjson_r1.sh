#!/bin/bash
# Phase B multiseed: Llama SVG raw_json — R1 rerun with temperature=0.0
# Previous run used temp=0.1, inconsistent with Phase B baseline (temp=0.0)

set -e
cd /root/autodl-tmp/verifier_feedback_representation

MODEL=/root/autodl-tmp/models/llama-3.1-8b-instruct
DOMAIN=svg
FMT=raw_json
OUTDIR=results/multiseed_phaseB
LOGDIR=logs

mkdir -p "$OUTDIR" "$LOGDIR"

for SEED in 42; do
    echo "=== Starting seed=$SEED at $(date) ==="
    nohup python3 -m src.phaseB_runner \
        --model-path "$MODEL" \
        --device cuda:0 \
        --domains "$DOMAIN" \
        --formats "$FMT" \
        --temperature 0.0 \
        --seed "$SEED" \
        --max-tokens 8192 \
        --max-model-len 16384 \
        --svg-split medium \
        --min-samples 30 \
        --output "${OUTDIR}/llama_svg_rawjson_r1_seed${SEED}.json" \
        > "${LOGDIR}/multiseed_phaseB_llama_svg_rawjson_r1_seed${SEED}.log" 2>&1 &

    PID=$!
    echo "  seed=$SEED launched as PID=$PID"

    # Wait for this seed to finish before starting next (single GPU)
    wait $PID
    echo "  seed=$SEED finished at $(date)"
done

echo "=== All seeds complete ==="
