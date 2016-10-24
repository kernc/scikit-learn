import numpy as np
from sklearn.utils import check_X_y
from sklearn.neighbors import BallTree
from sklearn.neighbors.base import _check_weights, _get_weights
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets


def relieff(X, y, n_neighbors=10, n_iter=100, neighbors_per_class=True,
            weights='uniform', metric='minkowski', metric_params=None,
            random_state=None):
    """Estimate feature relevance in classification problems using ReliefF.

    ReliefF [1]_, an extension of RELIEF algorithm [2]_, is a robust
    feature scoring algorithm that can detect conditional dependencies
    between features.

    The function returns relative feature importance (higher is better).

    It can be used for univariate features selection, read more in the
    :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Feature matrix.

    y : array_like, shape (n_samples,)
        Target vector.

    n_neighbors : int, (default = 10)
        Number of neighbors to use for scores adjustment in each iteration.

    n_iter : bool, (default = 100)
        Number of iterations to perform.

    neighbors_per_class : bool, (default = True)
        Find `(n_neighbors * n_classes)` neighbors instead of just
        `n_neighbors`. This more closely matches original ReliefF algorithm.

    weights : str or callable (default = 'uniform')
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood
          are weighted equally.
        - 'distance' : weigh points by the inverse of their distance.
          In this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    metric : string or DistanceMetric object (default = 'minkowski')
        The distance metric to use for determining nearest neighbors.  The
        default metric is minkowski, and with p=2 is equivalent to the
        standard Euclidean metric. See the documentation of the DistanceMetric
        class for a list of available metrics.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    random_state : int seed, RandomState instance or None, default None
        The seed of the pseudo random number generator for adding small noise
        to continuous variables in order to remove repeated values.

    Returns
    -------
    scores : ndarray, shape (n_features,)
        Estimated relevance of each feature.

    Notes
    -----
    1. Contrary to ReliefF, as described in [1]_ or [3]_, this implementation
       does *not* search for `n_neighbors` neighbors of each distinct class.
       Instead, as a performance optimization, `n_neighbors` neighbors
       (`(n_neighbors * n_classes)` if `neighbors_per_class` = True)
       are sought in total, and the scores are adjusted accordingly. In
       testing, this hasn't proved a hindrance. If your scores are all negative
       or non-discerning, try to increase the number of sought neighbors.
    2. This implementation does not handle missing (nan) values. If your
       data contains missing values, it makes sense to impute them by feature
       mean (for ordinal features) or mode (for nominal features).

    References
    ----------
    .. [1] I. Kononenko, E. Šimec, M. Robnik-Šikonja, "Overcoming the myopia
           of inductive learning algorithms with RELIEFF". Applied Intel. 7:1,
           1997, 39-55.
    .. [2] `RELIEF <https://en.wikipedia.org/wiki/Relief>`_
           on Wikipedia.
    .. [3] M. Robnik-Šikonja, I. Kononenko, "Theoretical and Empirical
           Analysis of ReliefF and RReliefF". MLJ 53, 2003, 23-69.
    """
    X, y = check_X_y(X, y, order='C')
    check_classification_targets(y)
    _check_weights(weights)
    rng = check_random_state(random_state)

    classes, priors = np.unique(y, return_inverse=True)
    priors = np.bincount(priors)
    priors = dict(zip(classes, priors / priors.sum()))

    if neighbors_per_class:
        n_neighbors = classes.size * n_neighbors

    n_neighbors = min(n_neighbors, X.shape[0] - 1)

    tree = BallTree(X, metric=metric, **(metric_params or {}))
    subset = rng.randint(0, X.shape[0], size=n_iter)

    # Add one more neighbor and skip the first result column as this is
    # the instance itself
    dist, ind = tree.query(X[subset], n_neighbors + 1, dualtree=True)
    dist, ind = dist[:, 1:], ind[:, 1:]
    influences = _get_weights(dist, weights)
    if influences is None:
        influences = np.tile(None, n_iter)

    scores = np.zeros(X.shape[1])

    for inst, neighs, influence in zip(subset, ind, influences):
        y_true, y_neighs = y[inst], y[neighs]

        inst_dists = np.abs(X[neighs] - X[inst])
        if influence is not None:
            inst_dists = inst_dists * influence.reshape(-1, 1)

        for cls in classes:
            adjustment = inst_dists[y_neighs == cls].sum(axis=0)

            if cls == y_true:
                scores -= adjustment
            else:
                scores += (priors[cls] / (1 - priors[y_true])) * adjustment

    return scores


def rrelieff(X, y, n_neighbors=10, n_iter=100, weights='distance',
             sigma='auto', metric='minkowski', metric_params=None, random_state=None):
    """Estimate feature relevance in regression problems using RReliefF.

    RReliefF [4]_, an extension of ReliefF [5]_ of the RELIEF family of
    algorithms [6]_, is a robust feature scoring algorithm that can detect
    conditional dependencies between features.

    The function returns relative feature importance (higher is better).

    It can be used for univariate features selection, read more in the
    :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Feature matrix.

    y : array_like, shape (n_samples,)
        Target vector.

    n_neighbors : int, (default = 10)
        Number of neighbors to use for scores adjustment in each iteration.

    n_iter : bool, (default = 100)
        Number of iterations to perform.

    weights : str or callable, (default = 'uniform')
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood
          are weighted equally.
        - 'distance' : weigh points by rank of their their distance.
          In this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.

          As described in [5]_, the influence exerted equals to:

          .. math:: I = e^{-(\\frac{r}{\sigma})^2}

          where I is the influence of point with distance rank :math:`r`.
          Smoothing parameter :math:`\sigma` can be controlled with
          parameter `sigma`. Higher values result in more smooth, less severe
          influence drop-off.

        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    sigma : float, (default = 'auto')
        The influence smoothing parameter. Only used when
        `weights` = 'distance'. When set to 'auto',
        ``sigma = n_neighbors``.

    metric : string or DistanceMetric object (default = 'minkowski')
        The distance metric to use for determining nearest neighbors.  The
        default metric is minkowski, and with p=2 is equivalent to the
        standard Euclidean metric. See the documentation of the DistanceMetric
        class for a list of available metrics.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    random_state : int seed, RandomState instance or None, default None
        The seed of the pseudo random number generator for adding small noise
        to continuous variables in order to remove repeated values.

    Returns
    -------
    scores : ndarray, shape (n_features,)
        Estimated relevance of each feature.

    References
    ----------
    .. [4] M. Robnik-Šikonja, I. Kononenko, "An adaptation of Relief for
           attribute estimation in regression". ICML, 1997, 296-304.
    .. [5] M. Robnik-Šikonja, I. Kononenko, "Theoretical and Empirical
           Analysis of ReliefF and RReliefF". MLJ 53, 2003, 23-69.
    .. [6] `RELIEF <https://en.wikipedia.org/wiki/Relief>`_
           on Wikipedia.
    """
    X, y = check_X_y(X, y, y_numeric=True, order='C')
    _check_weights(weights)
    rng = check_random_state(random_state)
    n_neighbors = min(n_neighbors, X.shape[0] - 1)
    sigma = n_neighbors if sigma == 'auto' else sigma

    tree = BallTree(X, metric=metric, **(metric_params or {}))
    subset = rng.randint(0, X.shape[0], size=n_iter)

    # Add one more neighbor and skip the first result column as this is
    # the instance itself
    dist, ind = tree.query(X[subset], n_neighbors + 1, dualtree=True)
    dist, ind = dist[:, 1:], ind[:, 1:]

    if weights == 'distance':
        # Distance-weighing is actually weighing by smoothed rank
        influences = np.exp(-(np.arange(n_neighbors) / sigma) ** 2)
        influences /= influences.sum()
        influences = np.tile(influences, (n_iter, 1))
    else:
        influences = _get_weights(dist, weights)
        if influences is None:
            influences = np.tile(1 / n_neighbors, n_iter)

    Nc = 0
    Na = np.zeros(X.shape[1])
    Nca = np.zeros(X.shape[1])
    for inst, neighs, influence in zip(subset, ind, influences):
        inst_dists = np.abs(X[neighs] - X[inst]) * influence.reshape((-1, 1))
        target_dists = np.abs(y[inst] - y[neighs])
        Nc += (target_dists * influence).sum()
        Na += inst_dists.sum(axis=0)
        Nca += (inst_dists * target_dists.reshape((-1, 1))).sum(axis=0)

    scores = Nca / Nc - (Na - Nca) / (n_iter - Nc)
    return scores


if __name__ == '__main__':
    import time
    from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
    # np.random.seed(0)
    X = np.random.random((1000, 50))
    y = ((X[:, 0] > .5) ^ (X[:, 1] > .5))
    weights = relieff(X, y)
    print(weights)
