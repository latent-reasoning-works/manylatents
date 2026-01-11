# Integration Testing Guide: manylatents + manylatents-omics

## How to Test Integration End-to-End

This guide shows how to verify that the namespace integration works in a real workflow by launching a manylatents experiment that can load omics metrics.

## Step 1: Setup Local Environment

```bash
# Install manylatents core package
uv add -e /path/to/manylatents --no-deps

# Install mock manylatents-omics locally
uv add -e /path/to/manylatents/tests/mock_omics_package --no-deps

# Verify imports work
python3 << 'EOF'
import manylatents
from manylatents.omics.data import PlinkDataset
from manylatents.omics.metrics import GeographicPreservation
print("✅ Namespace integration successful!")
EOF
```

## Step 2: Create Test Experiment Config

The namespace integration enables plugins from `manylatents-omics` to be loaded in the main workflow. Create a config that demonstrates this:

### Option A: Using Mock Metrics (Recommended for Testing)

Create `manylatents/configs/experiment/integration_test.yaml`:

```yaml
# Integration test: manylatents core + omics metrics
data:
  _target_: manylatents.data.synthetic_dataset.SwissRoll
  n_samples: 100
  noise: 0.01
  seed: 42

algorithms:
  latent:
    _target_: manylatents.algorithms.latent.pca.PCAModule
    n_components: 2

# Metrics from core package
metrics:
  embedding:
    - continuity
    - trustworthiness

# This is where omics metrics would be integrated
# (currently uses mock classes for testing)
callbacks:
  embedding:
    - minimal

logger: none
trainer:
  max_epochs: 1
```

### Option B: Using Real Omics Metrics (When Available)

When you have the actual `manylatents-omics` package installed, you could use:

```yaml
# This would be the full integration (requires actual omics package)
metrics:
  embedding:
    - continuity
    - trustworthiness
  dataset:
    - geographic_preservation  # From manylatents-omics

callbacks:
  embedding:
    - minimal
```

## Step 3: Run the Experiment

```bash
# Run with minimal config to test core integration
python -m manylatents.main \
  experiment=integration_test \
  logger=none

# Run with specific omics-compatible metrics
python -m manylatents.main \
  data=swissroll \
  algorithms/latent=pca \
  metrics=test_metric \
  callbacks/embedding=minimal \
  logger=none
```

## Step 4: Verify Integration in Code

Create a test script `test_integration_workflow.py`:

```python
#!/usr/bin/env python3
"""
Test that manylatents can dynamically load omics metrics/data.
This validates the namespace integration works end-to-end.
"""

def test_omics_import():
    """Test that omics namespace is accessible."""
    import manylatents
    from manylatents.omics.data import PlinkDataset
    from manylatents.omics.metrics import GeographicPreservation
    
    print("✅ Omics imports successful")
    
    # Test instantiation
    plink = PlinkDataset()
    geo = GeographicPreservation()
    print(f"✅ Can instantiate: {plink.name}, {geo.name}")


def test_workflow_with_omics():
    """Test running a workflow that could use omics metrics."""
    from manylatents.data.synthetic_dataset import SwissRoll
    from manylatents.algorithms.latent.pca import PCAModule
    
    # Create data
    data = SwissRoll(n_samples=50, noise=0.01, seed=42)
    print(f"✅ Created dataset with {len(data)} samples")
    
    # Create algorithm module
    algo = PCAModule(n_components=2)
    print(f"✅ Created {algo.__class__.__name__}")
    
    # In a real workflow, you'd also compute metrics from omics namespace
    # This validates the namespace is ready for metric computation


def test_dynamic_metric_loading():
    """Test that metrics can be dynamically discovered."""
    import importlib
    
    # Try to load a metric from omics namespace
    try:
        metrics_module = importlib.import_module("manylatents.omics.metrics")
        print(f"✅ Successfully imported omics metrics: {dir(metrics_module)}")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        raise


if __name__ == "__main__":
    print("=" * 70)
    print("Testing manylatents-omics integration")
    print("=" * 70)
    
    test_omics_import()
    print()
    test_workflow_with_omics()
    print()
    test_dynamic_metric_loading()
    
    print("\n" + "=" * 70)
    print("✅ All integration tests passed!")
    print("=" * 70)
```

Run it:
```bash
python3 test_integration_workflow.py
```

## Step 5: Monitor Integration Points

The namespace integration works at several levels:

### Level 1: Package Discovery
```python
# Check if namespace is properly extended
import manylatents
print(manylatents.__path__)  # Should show multiple paths if namespace is extended

import manylatents.omics
print(manylatents.omics.__path__)  # Should show omics namespace path
```

### Level 2: Module Loading
```python
# Check if submodules are discoverable
from manylatents.omics import data, metrics
print(dir(data))
print(dir(metrics))
```

### Level 3: Class Access
```python
# Check if classes can be instantiated
from manylatents.omics.data import PlinkDataset
from manylatents.omics.metrics import GeographicPreservation

dataset = PlinkDataset()
metric = GeographicPreservation()
print(f"Dataset: {dataset.name}")
print(f"Metric: {metric.name}")
```

### Level 4: Workflow Integration
```python
# This is what you'd do in a real metric computation
from manylatents.omics.metrics import GeographicPreservation

# Instantiate the metric
geo_metric = GeographicPreservation()

# In the real implementation, you'd compute with embeddings/data
# result = geo_metric.compute(embeddings, metadata)
print(f"Ready to compute: {geo_metric.name}")
```

## Troubleshooting Integration Issues

### Issue: "No module named 'manylatents.omics'"

**Cause**: The omics package isn't installed.

**Solution**:
```bash
# Install the mock package for testing
uv add -e ./tests/mock_omics_package --no-deps

# Or install the real package if you have access
uv add -e /path/to/manylatents-omics --no-deps
```

### Issue: "ImportError: cannot import name 'PlinkDataset'"

**Cause**: The mock package structure isn't set up correctly.

**Solution**: Check that the file structure exists:
```bash
ls -R tests/mock_omics_package/manylatents/omics/
# Should show: data/, metrics/, __init__.py
```

### Issue: Namespace not extending properly

**Cause**: Missing `pkgutil.extend_path()` in `__init__.py` files.

**Solution**: Ensure both packages have:
```python
# In manylatents/__init__.py and manylatents/omics/__init__.py
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
```

## Testing Checklist

- [ ] Core manylatents imports work
- [ ] Omics namespace is importable: `import manylatents.omics`
- [ ] Omics submodules exist: `from manylatents.omics import data, metrics`
- [ ] Mock classes can be instantiated
- [ ] Workflow can be created and run
- [ ] Metrics from omics namespace can be discovered dynamically
- [ ] Both packages can coexist in the same Python environment

## Next Steps: Real Omics Integration

When the actual `manylatents-omics` package is available:

1. Update the config files to include omics metrics
2. Create integration configs that demonstrate omics functionality
3. Update the mock package to match the real structure exactly
4. Run end-to-end tests with real genetics data

For now, the mock package validates the **namespace architecture** works correctly, so adding the real package later will be straightforward.
