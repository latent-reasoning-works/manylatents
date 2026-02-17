#!/bin/bash
# Smoke test for all metric configs
# Discovers metric configs across embedding/dataset/module and validates each

set -e

echo "=========================================="
echo "Smoke Testing All Metric Configs"
echo "=========================================="

METRICS_DIR="manylatents/configs/metrics"
FAILED=()
TESTED=0

for subdir in embedding dataset module; do
    DIR="$METRICS_DIR/$subdir"
    [ -d "$DIR" ] || continue

    CONFIGS=($(ls -1 "$DIR"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//' | grep -v __init__ | grep -v noop))

    for config in "${CONFIGS[@]}"; do
        echo "→ Testing: metrics/$subdir=$config"
        TESTED=$((TESTED + 1))

        CMD="python -m manylatents.main \
            algorithms/latent=pca \
            data=swissroll \
            data.n_distributions=5 \
            data.n_points_per_distribution=20 \
            data.rotate_to_dim=50 \
            metrics/$subdir=$config \
            callbacks/embedding=minimal \
            logger=none"

        if $CMD > /tmp/test_metric_${config}.log 2>&1; then
            echo "  ✅ $config"
        else
            echo "  ❌ $config FAILED"
            tail -5 /tmp/test_metric_${config}.log | sed 's/^/    /'
            FAILED+=("$subdir/$config")
        fi
    done
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Tested: $TESTED"
echo "Passed: $((TESTED - ${#FAILED[@]}))"
echo "Failed: ${#FAILED[@]}"

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✅ All metric smoke tests passed"
    exit 0
else
    echo "❌ Failed metrics:"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi
