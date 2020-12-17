import numpy as np

# Utilities only meant for the doubly-robust branch


def tmle_unit_bounds(y, mini, maxi, bound):
    # bounding for continuous outcomes
    v = (y - mini) / (maxi - mini)
    v = np.where(np.less(v, bound), bound, v)
    v = np.where(np.greater(v, 1-bound), 1-bound, v)
    return v


def tmle_unit_unbound(ystar, mini, maxi):
    # unbounding of bounded continuous outcomes
    return ystar*(maxi - mini) + mini
