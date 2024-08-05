import numpy as np


def compute_cosine_weights(x):
    """ Computes weights to be used in the id loss function with minimum value of 0.5 and maximum value of 1. """
    values = np.abs(x.cpu().detach().numpy())
    assert np.min(values) >= 0. and np.max(
        values) <= 1., "Input values should be between 0. and 1!"
    weights = 0.25 * (np.cos(np.pi * values)) + 0.75
    return weights
