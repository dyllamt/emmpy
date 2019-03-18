from semitransport.base.powerlaw_model import powerlaw_conductivity,\
    powerlaw_seebeck
from semitransport.base.models.sphere_model import sphere_seebeck,\
    sphere_carriers, constant

from scipy.optimize import minimize

import numpy as np

'''
this module implements functions to extract transport coefficients from
transport data (Seebeck, conductivity, etc.)
'''

_m_e = constant['m_e']


def extract_transport_function(seebeck, conductivity, temperature, s=1):
    '''
    returns the transport function prefactor (sigma_E_0) given Seebeck-
    conductivity data. the transport exponent (s) is typically 1 for
    deformation potential scattering and band transport

    Args:
      seebeck (float) the Seebeck coefficient, V/K
      conductivity (float) electrical conductivity, S/m
      temperature (float) the absolute temperature, K
      s (int|half-integer) assumption of transport exponent (mechanism)


    Returns: (float) the transport function prefactor sigma_E_0, S/m
    '''
    cp = minimize(
        lambda cp: np.abs(powerlaw_seebeck(cp, s) - np.abs(seebeck)),
        method='Nelder-Mead', x0=[0.]).x[0]
    return minimize(lambda sigma_E_0: np.abs(
        powerlaw_conductivity(cp, s, sigma_E_0) - conductivity),
        method='Nelder-Mead', x0=[0.]).x[0]


def extract_effective_mass(seebeck, carrier_density, temperature):
    '''
    returns the equivalent mass for a spherical pocket with a specific carrier
    density and Seebeck coefficient. the transport exponent (s) is implicitly
    1 in this analysis, which indicates deformation potential scattering

    Args:
      seebeck (float) the Seebeck coefficient, V/K
      carrier_density (float) the carrier density, 1/m^3
      temperature (float) the absolute temperature, K


    Returns: (float) the effective mass m*, m_e
    '''
    cp = minimize(
        lambda cp: np.abs(sphere_seebeck(cp) - np.abs(seebeck)),
        method='Nelder-Mead', x0=[0.]).x[0]
    return minimize(lambda mstar: np.abs(
        sphere_carriers(cp, temperature, mstar * _m_e) - carrier_density),
        method='Nelder-Mead', x0=[0.]).x[0]
