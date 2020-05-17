# JAX Random Projection Transformers
Using JAX to speed up sklearn's random projection transformers

## Installation

**Note: Installation with pip will install the CPU-only version of JAX**

To use a GPU follow [JAX's installation guide](https://github.com/google/jax#installation) before installing `jax-random_projections`.
```
pip install jax-random_projections
```

## Usage
```python
from jax_random_projections.sparse import SparseRandomProjectionJAX

transfomer = SparseRandomProjectionJAX()
transfomer.fit_transform(X)
```

For the API documentation, refer to [sklearn's SparseRandomProjection documentation](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html).
The only difference is that `jax-random_projections` currently only supports `xla.DeviceArray` and doesn't support `dense_output=False` and `y` for `fit()`
This library currently only includes the `SparseRandomProjection` but a future release will also include `GaussianRandomProjection`.

`jax-random_projections` also includes `SparseRandomProjectionJAXCached` which uses a lru cache (`maxsize=5`) to speed up repeated calls by caching the random matrix for data with the same input dimension.
