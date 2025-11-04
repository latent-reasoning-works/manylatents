# Welcome to ManyLatents

ManyLatents is a framework for learning on manifolds...
todo: add descriptive info

---

## What You'll Find Here

- **Getting Started**: Learn how to install and use ManyLatents.
- **[Extensions Guide](extensions.md)**: Install domain-specific extensions (genetics, imaging, etc.).
- **[API Usage Guide](api_usage.md)**: Programmatic API for chaining algorithms and agent workflows.
- **[Metrics Architecture](metrics_architecture.md)**: Three-level metrics system design and usage.
- **[Testing Guide](testing.md)**: Testing infrastructure and best practices.
- **[Contributing Guide](../CONTRIBUTING.md)**: Guidelines for adding new metrics, algorithms, and datasets.
- **[TODO & Future Work](TODO.md)**: Planned enhancements and testing improvements.

---

## Extensions

ManyLatents supports domain-specific extensions through namespace packages:

- **[manylatents-omics](https://github.com/latent-reasoning-works/manylatents-omics)**: Genetics and population genetics
  - PLINK/VCF data loaders
  - Geographic preservation metrics
  - Ancestry-specific algorithms

Install extensions easily:
```bash
pip install git+https://github.com/latent-reasoning-works/manylatents-omics.git
```

[Learn more about extensions →](extensions.md)

---

## About ManyLatents

ManyLatents is designed to [highlight the unique aspects of your project]. Whether you're a beginner or an advanced user, this documentation will guide you through everything you need to know.
