"""
Fuzzy C-Means clustering optimized for Apple Silicon using MLX.
"""

import mlx.core as mx


class FuzzyCMeans:
    """
    Fuzzy C-Means clustering algorithm.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    m : float, default=2.0
        Fuzziness parameter (m > 1). Higher values produce fuzzier clusters.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    cluster_centers_ : array of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : array of shape (n_samples,)
        Labels of each point (cluster with highest membership).
    u_ : array of shape (n_samples, n_clusters)
        Fuzzy membership matrix.
    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import mlx.core as mx
    >>> from fcm import FuzzyCMeans
    >>> X = mx.random.normal((100, 5))
    >>> fcm = FuzzyCMeans(n_clusters=3, m=2.0)
    >>> fcm.fit(X)
    >>> print(fcm.cluster_centers_.shape)
    (3, 5)
    """

    def __init__(self, n_clusters, m=2.0, max_iter=100, tol=1e-4, random_state=None):
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if m <= 1:
            raise ValueError("m (fuzziness) must be > 1")

        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_ = None
        self.u_ = None
        self.labels_ = None
        self.n_iter_ = None

    def fit(self, X):
        """
        Compute Fuzzy C-Means clustering.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        if self.random_state is not None:
            mx.random.seed(self.random_state)

        centers = self._init_centers(X)

        for iteration in range(self.max_iter):
            distances = self._compute_distances(X, centers)
            u = self._update_memberships(distances)
            centers_new = self._update_centers(X, u)

            diff = mx.sum((centers_new - centers) ** 2)
            mx.eval(diff)

            if float(diff) < self.tol:
                self.n_iter_ = iteration + 1
                break

            centers = centers_new
        else:
            self.n_iter_ = self.max_iter

        mx.eval(centers, u)

        self.cluster_centers_ = centers
        self.u_ = u
        self.labels_ = mx.argmax(u, axis=1)

        return self

    def fit_predict(self, X):
        """
        Compute cluster labels for samples in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data to cluster.

        Returns
        -------
        labels : array of shape (n_samples,)
            Cluster labels.
        """
        return self.fit(X).labels_

    def predict(self, X):
        """
        Predict the closest cluster for each sample.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : array of shape (n_samples,)
            Cluster labels.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet")

        distances = self._compute_distances(X, self.cluster_centers_)
        u = self._update_memberships(distances)

        return mx.argmax(u, axis=1)

    def _init_centers(self, X):
        """Initialize cluster centers randomly from data points."""
        n_samples = X.shape[0]
        indices = mx.random.randint(0, n_samples, (self.n_clusters,))
        return X[indices]

    def _compute_distances(self, X, centers):
        """
        Compute Euclidean distances between all points and centers.

        Uses broadcasting for efficiency:
        X: (n, d), centers: (c, d) -> distances: (n, c)
        """
        X_expanded = mx.expand_dims(X, axis=1)
        centers_expanded = mx.expand_dims(centers, axis=0)
        diff = X_expanded - centers_expanded
        distances = mx.sum(diff**2, axis=2)
        return mx.maximum(distances, 1e-10)

    def _update_memberships(self, distances):
        """
        Update fuzzy membership matrix.

        Formula: u_ik = 1 / sum_j (d_ik / d_ij)^(2/(m-1))
        """
        exp = 2.0 / (self.m - 1.0)

        d_ik = mx.expand_dims(distances, axis=2)
        d_ij = mx.expand_dims(distances, axis=1)
        ratio = d_ik / d_ij
        powered = ratio**exp
        sum_powered = mx.sum(powered, axis=2)

        return 1.0 / sum_powered

    def _update_centers(self, X, u):
        """
        Update cluster centers.

        Formula: c_j = sum_i (u_ij^m * x_i) / sum_i (u_ij^m)
        """
        u_m = u**self.m

        u_m_expanded = mx.expand_dims(u_m, axis=2)
        X_expanded = mx.expand_dims(X, axis=1)
        weighted = u_m_expanded * X_expanded
        numerator = mx.sum(weighted, axis=0)

        denominator = mx.sum(u_m, axis=0)
        denominator = mx.maximum(denominator, 1e-10)

        return numerator / mx.expand_dims(denominator, axis=1)
