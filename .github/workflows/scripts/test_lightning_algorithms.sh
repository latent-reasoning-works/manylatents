#!/bin/bash
# Smoke test for all LightningModule algorithm configs

set -e

echo "=========================================="
echo "Smoke Testing All LightningModule Algorithms"
echo "=========================================="

ALGO_DIR="manylatents/configs/algorithms/lightning"
ALGORITHMS=($(ls -1 "$ALGO_DIR"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//' | grep -v __init__))

echo "Discovered ${#ALGORITHMS[@]} algorithms from $ALGO_DIR:"
printf '  - %s\n' "${ALGORITHMS[@]}"
echo ""

FAILED=()

for algo in "${ALGORITHMS[@]}"; do
    echo "→ Testing: $algo"

    CMD="python -m manylatents.main \
        algorithms/lightning=$algo \
        data=swissroll \
        data.n_distributions=5 \
        data.n_points_per_distribution=20 \
        data.rotate_to_dim=50 \
        metrics=noop \
        callbacks/embedding=minimal \
        trainer=default \
        trainer.max_epochs=1 \
        trainer.fast_dev_run=true \
        logger=none"

    if $CMD > /tmp/test_lightning_${algo}.log 2>&1; then
        echo "  ✅ $algo"
    else
        echo "  ❌ $algo FAILED"
        tail -5 /tmp/test_lightning_${algo}.log | sed 's/^/    /'
        FAILED+=("$algo")
    fi
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Tested: ${#ALGORITHMS[@]}"
echo "Passed: $((${#ALGORITHMS[@]} - ${#FAILED[@]}))"
echo "Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✅ All LightningModule smoke tests passed"
    exit 0
else
    echo "❌ Failed algorithms:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
