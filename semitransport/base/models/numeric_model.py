import numpy as np

'''
this module implements basic functions for transport coefficients, which are
computed by integrating an arbitrary sigma_E function with the appropriate
Fermi-Dirac weighting functions

Model variables:
  reduced chemical potential (cp) unitless (mu/kT)
  transport function exponent (s) unitless
  transport function prefactor (sigma_E_0) same units as conductivity (S/m)
'''

constant = {'e': 1.60217662e-19,  # physical constants
            'k': 1.38064852e-23}


def numeric_conductivity(energy, sigma_E, mu, T):
    '''
    returns the electrical conductivity (S/m)

    Args:
      energy: (ndarray) The x-points for sigma_E v.s. energy (units of eV).
      sigma_E: (ndarray) The y-points for sigma_E v.s. energy (units of S/m).
      mu: (float) The electron chemical potential (units of eV).
      T: (float) The absolute temperature (units of K).

    Returns: (float)
    '''

    beta = 1. / constant['k'] / T * constant['e']  # units of eV
    derivative_of_fermi_dirac = -(
        beta * np.exp(beta * (energy - mu)) / (
            np.exp(beta * (energy - mu)) + 1.)**2.)
    return np.trapz(
        y=sigma_E * (-derivative_of_fermi_dirac),
        x=energy)


def numeric_nu(energy, sigma_E, mu, T):
    '''
    returns the transport coefficient for a temperature gradient

    Args:
      energy: (ndarray) The x-points for sigma_E v.s. energy (units of eV).
      sigma_E: (ndarray) The y-points for sigma_E v.s. energy (units of S/m).
      mu: (float) The electron chemical potential (units of eV).
      T: (float) The absolute temperature (units of K).

    Returns: (float)
    '''

    beta = 1. / constant['k'] / T * constant['e']  # units of eV
    derivative_of_fermi_dirac = -(
        beta * np.exp(beta * (energy - mu)) / (
            np.exp(beta * (energy - mu)) + 1.)**2.)
    return np.trapz(
        y=(constant['k'] / constant['e'] * sigma_E *
           (-derivative_of_fermi_dirac) * (energy - mu) * beta),
        x=energy)


def numeric_seebeck(energy, sigma_E, mu, T):
    '''
    returns the Seebeck coefficient (V/K)

    Args:
      energy: (ndarray) The x-points for sigma_E v.s. energy (units of eV).
      sigma_E: (ndarray) The y-points for sigma_E v.s. energy (units of S/m).
      mu: (float) The electron chemical potential (units of eV).
      T: (float) The absolute temperature (units of K).

    Returns: (float)
    '''

    return (numeric_nu(energy, sigma_E, mu, T) /
            numeric_conductivity(energy, sigma_E, mu, T))
