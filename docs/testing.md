# Testing Strategy

This document outlines the GitHub Actions testing strategy and local testing practices for ManyLatents.

## Overview

Our testing pipeline ensures code quality and validates algorithm functionality across different datasets and configurations.

## GitHub Actions Pipeline

### Three-Phase Testing

1. **Build & Unit Tests** (20 min): Fast validation of code quality and unit tests
2. **LatentModule Smoke Tests** (conditional): Automatic testing of all DR algorithms when configs change
3. **Integration Tests** (25 min): Matrix testing of algorithm combinations with real data

### Current Test Matrix

#### LatentModule Smoke Tests (Conditional)

**Purpose**: Automatically test all LatentModule algorithms when configs change

**Triggers on changes to**:
- `manylatents/algorithms/latent/**`
- `manylatents/configs/algorithms/latent/**`

**Script**: `.github/workflows/scripts/test_latent_algorithms.sh`

**What it does**:
- Dynamically discovers all algorithm configs in `manylatents/configs/algorithms/latent/`
- Runs quick instantiation test for each algorithm
- Uses minimal test data (swissroll: 10 distributions × 20 points)
- Currently tests: `aa`, `diffusionmap`, `mds`, `noop`, `pca`, `phate`, `tsne`, `umap`

**What it validates**:
- ✅ Algorithm config is valid YAML
- ✅ Algorithm class can be instantiated via Hydra
- ✅ Algorithm can fit and transform data
- ✅ Full pipeline completes without errors

**What it does NOT test**:
- ❌ Mathematical correctness (covered by unit tests)
- ❌ Lightning modules (future work)
- ❌ Comprehensive data/metric combinations (covered by integration tests)

**Run locally**:
```bash
.github/workflows/scripts/test_latent_algorithms.sh
```

#### Basic Smoke Tests
- **smoke-test**: Basic functionality validation
  - Algorithm: `latent/noop`
  - Data: `test_data`
  - Timeout: 2 minutes

#### Traditional Dimensionality Reduction
- **pca-swissroll**: PCA on SwissRoll manifold data
  - Algorithm: `latent/pca`
  - Data: `swissroll`
  - Timeout: 8 minutes

- **umap-swissroll**: UMAP embedding validation
  - Algorithm: `latent/umap`
  - Data: `swissroll`
  - Timeout: 10 minutes

#### Neural Network Algorithms
- **autoencoder-swissroll**: Autoencoder training validation
  - Algorithm: `lightning/ae_reconstruction`
  - Data: `swissroll`
  - Timeout: 15 minutes

## Local Testing

### Running Matrix Tests Locally

You can run any of the matrix configurations locally:

```bash
# Smoke test
python -m manylatents.main \
  algorithm=latent/noop \
  data=test_data \
  metrics=test_metric \
  debug=true

# PCA + SwissRoll  
python -m manylatents.main \
  algorithm=latent/pca \
  data=swissroll \
  metrics=synthetic_data_metrics \
  debug=true \
  trainer.max_epochs=1

# Autoencoder + SwissRoll
python -m manylatents.main \
  algorithm=lightning/ae_reconstruction \
  data=swissroll \
  metrics=synthetic_data_metrics \
  debug=true \
  trainer.max_epochs=1
```

### Unit Tests

Run the full pytest suite:

```bash
# Full test suite with coverage
pytest --cov=manylatents --cov-report=xml

# Quick tests only
pytest --maxfail=1 --disable-warnings -q
```

## Future Expansion

### Adding New Algorithm Tests

To add a new algorithm to the CI matrix, edit `.github/workflows/build.yml`:

```yaml
- name: "new-algorithm-test"
  algorithm: "latent/your_algorithm"  # or lightning/your_model
  data: "test_dataset"
  metrics: "appropriate_metrics"
  timeout: 5  # minutes
```

### Sequential Workflow Testing

When ready, enable sequential algorithm testing:

```yaml
# Uncomment in build.yml
- name: "sequential-pca-umap"
  algorithms: "[latent/pca,latent/umap]"
  data: "swissroll"
  metrics: "synthetic_data_metrics"
  timeout: 12
```

### Dataset-Specific Testing

Examples for specialized algorithm-dataset combinations:

```yaml
# Genomic data validation
- name: "pca-genomic"
  algorithm: "latent/pca"
  data: "hgdp_split"
  metrics: "genomic_metrics"
  timeout: 20

# Single-cell data validation
- name: "phate-singlecell"
  algorithm: "latent/phate"
  data: "anndata"
  metrics: "singlecell_metrics"
  timeout: 15
```

## Testing Guidelines

### Algorithm Integration Checklist

When adding a new algorithm:

- [ ] Unit tests for the algorithm class
- [ ] Integration test in CI matrix
- [ ] Config file validation
- [ ] Documentation examples
- [ ] Performance benchmark (if applicable)

### Timeout Guidelines

- **Smoke tests**: 2-3 minutes
- **Traditional DR**: 5-10 minutes  
- **Neural networks**: 10-20 minutes
- **Large datasets**: 20-30 minutes

### Data Size Considerations

- **CI Tests**: Use small/synthetic datasets for speed
- **Integration**: Focus on algorithm correctness, not performance
- **Nightly**: Reserve large datasets for scheduled runs

## Monitoring and Debugging

### Test Artifacts

All test runs generate artifacts:
- Test outputs saved to `outputs/`
- Coverage reports from unit tests
- Individual test logs for debugging

### Failure Investigation

1. Check test-specific artifacts in GitHub Actions
2. Run the same configuration locally
3. Enable `HYDRA_FULL_ERROR=1` for detailed stack traces
4. Use `debug=true` for verbose logging

### Performance Monitoring

Key metrics to track:
- **Test duration**: Monitor for performance regressions
- **Failure patterns**: Identify problematic combinations
- **Coverage**: Ensure new code is tested

## Best Practices

### Algorithm Development

1. **Start with smoke test**: Ensure basic instantiation works
2. **Add integration test**: Validate with real data
3. **Document configuration**: Update examples and docs
4. **Monitor CI**: Watch for failures across environments

### Configuration Testing

- Test with `debug=true` for development
- Use `trainer.max_epochs=1` for quick validation
- Include both `latent/` and `lightning/` algorithm types
- Test with different dataset sizes and types

### Contributing

When submitting PRs:

1. Ensure CI passes for all matrix tests
2. Add new tests for new algorithms/features
3. Update documentation with examples
4. Consider performance implications of new tests