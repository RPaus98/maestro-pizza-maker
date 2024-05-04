import numpy as np


def _generate_positive_semi_definite_matrix(dim: int) -> np.array:
    """
    Generates a positive semi-definite matrix of dimension dim to be used as a covariance matrix.
    """
    dummy_matrix = np.random.rand(dim, dim)
    return np.dot(dummy_matrix, dummy_matrix.transpose())


def _generate_normal_vector(dim: int) -> np.array:
    """
    Generates a vector of dimension dim with values from a normal distribution.
    """
    return np.random.normal(size=dim, loc=30, scale=5).clip(min=1)


def _generate_multivariate_normal_vector(dim: int) -> np.array:
    """
    Generates a vector of dimension dim with values from a multivariate normal distribution.
    """
    mean = _generate_normal_vector(dim)
    cov = _generate_positive_semi_definite_matrix(dim)
    return np.random.multivariate_normal(mean, cov, 1000).clip(min=0.1)


FAT_SIMULATIONS = _generate_multivariate_normal_vector(16).transpose()
