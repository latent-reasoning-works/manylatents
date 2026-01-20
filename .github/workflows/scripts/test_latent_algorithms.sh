#!/bin/bash
# Smoke test for all LatentModule algorithms
# Dynamically discovers algorithm configs and tests instantiation
# This ensures new algorithms added in PRs are automatically validated

set -e  # Exit on error

echo "=========================================="
echo "Smoke Testing All LatentModule Algorithms"
echo "=========================================="

# Dynamically discover all latent algorithm configs
ALGO_DIR="manylatents/configs/algorithms/latent"
ALGORITHMS=($(ls -1 "$ALGO_DIR"/*.yaml | xargs -n1 basename | sed 's/.yaml$//' | grep -v __init__))

echo "Discovered ${#ALGORITHMS[@]} algorithms from $ALGO_DIR:"
printf '  - %s\n' "${ALGORITHMS[@]}"
echo ""

# Minimal fast configuration
# Note: --config-path required since shop (SearchPathPlugin) not installed in CI
BASE_CMD="python -m manylatents.main \
    --config-path=manylatents/configs \
    --config-name=base_config \
    data=swissroll \
    data.n_distributions=10 \
    data.n_points_per_distribution=20 \
    data.rotate_to_dim=50 \
    metrics=test_metric \
    callbacks/embedding=minimal \
    logger=none"

echo "Running smoke tests..."
echo ""

FAILED=()

for algo in "${ALGORITHMS[@]}"; do
    echo "→ Testing: $algo"

    # Run without n_components override for algorithms that don't support it
    # The config will use its default value
    if $BASE_CMD algorithms/latent=$algo > /tmp/test_${algo}.log 2>&1; then
        echo "  ✅ $algo"
    else
        echo "  ❌ $algo FAILED"
        echo "  Error log:"
        tail -10 /tmp/test_${algo}.log | sed 's/^/    /'
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
    echo ""
    echo "✅ All LatentModule smoke tests passed"
    exit 0
else
    echo ""
    echo "❌ Failed algorithms:"
    for algo in "${FAILED[@]}"; do
        echo "  - $algo"
    done
    exit 1
fi
