'''
Functions in this module compute "data-series" level transport properties
(i.e. properties for a material v.s. temperature or carrier-density)
'''

import numpy as np
from data_points import extract_transport_function


def jonker_analysis(seebecks, conductivities, temperature, s=1):
    '''
    a jonker analysis is typicaly applied to a series of samples
    with the same electronic structure but different doping concentrations
    the properties of each sample should be measured at the same temperature

    Args:
      seebecks: ([float]) the Seebeck coefficients, V/K
      conductivities: ([float]) electrical conductivities, S/m
      temperature: (float) the absolute temperature, K
      s: (int|half-integer) assumption of the transport exponent (mechanism)

    Returns: (touple of floats) transport function prefactors sigma_E_0, S/m
      - average value from all samples
      - minimum value from one sample
      - maximum value from one sample
    '''

    trans_funcs = []
    for seeb, cond in zip(seebecks, conductivities):
        trans_funcs.append(
            extract_transport_function(seeb, cond, temperature, s))
    return (np.mean(trans_funcs), min(trans_funcs), max(trans_funcs))


def temperature_analysis(seebecks, conductivities, temperatures, s=1):
    '''
    temperature analysis is carried out for a single sample
    the temperature dependence of the transport function is informative
    when checking for multi-band behavior or strange scattering mechanisms

    Args:
      seebecks: ([float]) the Seebeck coefficients, V/K
      conductivities: ([float]) electrical conductivities, S/m
      temperatures: ([float]) the absolute temperatures, K
      s: (int|half-integer) assumption of the transport exponent (mechanism)

    Returns: ([float]) transport function prefactors sigma_E_0, S/m
    '''

    trans_funcs = []
    for seeb, cond, temp in zip(seebecks, conductivities, temperatures):
        trans_funcs.append(
            extract_transport_function(seeb, cond, temp, s))
    return trans_funcs
