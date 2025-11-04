# Local Namespace Package Integration Testing

Since the organization is on a free plan, GitHub organization secrets cannot be used in CI workflows. This means we cannot directly test with the private `manylatents-omics` repository in CI.

## Solution: Mock Package Local Testing

We've created a **mock `manylatents-omics` package** that simulates the structure of the actual omics extension without requiring the private repository.

### Directory Structure

```
tests/
├── mock_omics_package/
│   ├── setup.py                          # Package configuration
│   └── manylatents/
│       ├── __init__.py                   # Namespace package declaration
│       └── omics/
│           ├── __init__.py               # Omics namespace
│           ├── data/
│           │   └── __init__.py           # Mock data module with PlinkDataset
│           └── metrics/
│               └── __init__.py           # Mock metrics module with GeographicPreservation
└── test_namespace_integration.py         # Integration test script
```

### How It Works

The mock package:
1. Declares `manylatents` as a namespace package using `pkgutil.extend_path()`
2. Provides mock implementations of key classes (`PlinkDataset`, `GeographicPreservation`)
3. Allows testing the namespace integration without the private repo

### Testing Locally

```bash
# Run the integration test script
python3 tests/test_namespace_integration.py
```

The script will:
1. Create a temporary virtual environment
2. Install `manylatents` (core package)
3. Install the mock `manylatents-omics` package
4. Verify both packages coexist as namespace packages
5. Test that all imports work correctly
6. Clean up the temporary environment

### CI Workflow Changes

The CI workflow has been updated to:
- **Remove** the `test-omics-integration` job that required the private repository
- **Add comments** documenting that namespace integration is tested locally
- **Focus on** the core `build-and-test` job which validates the main package

### Why This Approach Works

✅ **No CI secrets needed** - Tests run on any machine  
✅ **Validates the namespace architecture** - Ensures the package structure is correct  
✅ **Fast** - Doesn't require cloning large repositories  
✅ **Maintainable** - Mock package is part of the repo and under version control  
✅ **Scalable** - Easy to add more mock modules as the omics package grows

### When to Update the Mock Package

Update the mock package's structure if:
- The actual `manylatents-omics` package adds new top-level modules
- Namespace package organization changes
- You want to test new integration scenarios

The mock classes don't need implementation details - they just need to exist so Python can import them.
