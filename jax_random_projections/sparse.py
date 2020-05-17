import jax
import jax.numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils.validation import check_is_fitted 
from functools import lru_cache

class SparseRandomProjectionJAX(SparseRandomProjection):
    def __init__(self, n_components='auto', density='auto', eps=0.1, dense_output=False, random_state=None):
        if dense_output:
            raise NotImplementedError("Dense output is not supported right now")
        super().__init__(n_components=n_components, density=density, eps=eps, dense_output=dense_output, random_state=random_state)
        
    def fit(self, X, y = None):
        if y is not None:
            raise NotImplementedError("y is not supported right now")
        super().fit(X)
        return self
    
    def _validate_data(self, X, *args, **kwargs):
        if isinstance(X, jax.interpreters.xla.DeviceArray):
            return X
        else:
            raise TypeError("Only jax.interpreters.xla.DeviceArray supported")
    
    def transform(self, X, *args, **kwargs):
        if len(args) > 1 or len(kwargs)>1:
                raise NotImplementedError("y is not supported right now")
        check_is_fitted(self)

        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(
                'Impossible to perform projection:'
                'X at fit stage had a different number of features. '
                '(%s != %s)' % (X.shape[1], self.components_.shape[1]))

        return np.matmul(X, self.components_.T.toarray())

class SparseRandomProjectionJAXCached(SparseRandomProjectionJAX):
    @lru_cache(maxsize=5)
    def _make_random_matrix(self, n_components, n_features):
        return super()._make_random_matrix(n_components, n_features)
