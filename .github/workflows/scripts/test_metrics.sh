#!/bin/bash
# Smoke test for all metric configs
# Discovers metric configs from flat configs/metrics/ directory and validates each

set -e

echo "=========================================="
echo "Smoke Testing All Metric Configs"
echo "=========================================="

METRICS_DIR="manylatents/configs/metrics"
FAILED=()
TESTED=0

# Skip bundles (configs with 'defaults:' key), null, and noop
CONFIGS=($(ls -1 "$METRICS_DIR"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/.yaml$//' | grep -v __init__ | grep -v noop | grep -v null))

for config in "${CONFIGS[@]}"; do
    # Skip bundle configs (those containing a defaults: key)
    if grep -q '^defaults:' "$METRICS_DIR/${config}.yaml" 2>/dev/null; then
        echo "→ Skipping bundle: metrics=$config"
        continue
    fi

    echo "→ Testing: metrics=$config"
    TESTED=$((TESTED + 1))

    CMD="python -m manylatents.main \
        algorithms/latent=pca \
        data=swissroll \
        data.n_distributions=5 \
        data.n_points_per_distribution=20 \
        data.rotate_to_dim=50 \
        metrics=$config \
        callbacks/embedding=minimal \
        logger=none"

    if $CMD > /tmp/test_metric_${config}.log 2>&1; then
        echo "  ✅ $config"
    else
        echo "  ❌ $config FAILED"
        tail -5 /tmp/test_metric_${config}.log | sed 's/^/    /'
        FAILED+=("$config")
    fi
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
