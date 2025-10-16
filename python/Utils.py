import numpy as np
from scipy.interpolate import RegularGridInterpolator
import math


# Global constant here (to avoid hard code repeated)
# GC location in RA and Dec [in deg]:
GCRA = 266.4167
GCDec = -29.0078


#------------------------------------------------------------
## Get the open angle [rad] from GC, psi from RA and DEC [rad]

def psi_f(RA,decl):
    return np.arccos(np.cos(np.pi/2.-(GCDec*np.pi/180))*np.cos(np.pi/2.-decl)
                      +np.sin(np.pi/2.-(GCDec*np.pi/180))*np.sin(np.pi/2.-decl)*
                       np.cos(RA-GCRA*np.pi/180))


#------------------------------------------------------------
# Other useful functions


def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)