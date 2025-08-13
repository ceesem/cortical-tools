---
title: Cortical Tools
---

Cortical Tools offers dataset-specific Python clients for interacting with several CAVE datasets, in particular the Microns65 dataset and V1dd dataset.
It wraps CAVEclient and related tooling with a more task-focused design than the infrastructure-focused CAVEclient.
The public entry point is a preconfigured client for datasets.

- Get started with the [Examples](examples.md)
- Browse the [API Reference](reference/api.md)
- See what's new in the [Changelog](changelog.md)

## Who can use it?

Cortical Tools can be used if you are working with either the publicly accessible datastacks `minnie65_public` or `v1dd_public`.
In addition, if you are using the production datastacks for these same datastacks, you can use those as well.

## Installation

```bash
pip install cortical-tools
```

Cortical Tools is opinionated, and installs not only CAVEclient, but a number of related tools like `nglui`, `pcg_skel`, `cloudvolume`, and `standard_transform` that are commonly used in conjunction with CAVE datasets.

## Quick start

```python
from cortical_tools.datasets.minnie_public import client
client = MinniePublicClient()
client
```

This will return a `MinniePublicClient` object that is ready to use. For example, `client.neuroglancer_url()` will return a URL that opens the Neuroglancer viewer at the specified location.
Your standard `CAVEclient` functionality is available under `client.cave`.

## Who made this?

This is a totally unofficial tool that [Casey Schneider-Mizell](https://www.csdashm.com) made to use for himself, and maybe you'll find some useful things here too.