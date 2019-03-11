from fdint import fdk  # function that implements Fermi-Dirac integrals

import numpy as np

'''
this module implements basic functions for transport coefficients, which are
computed using the Fermi-Dirac integrals. more information on this model can be
found at: https://www.nature.com/articles/nmat4784

Model variables:
  reduced chemical potential (cp) unitless (mu/kT)
  transport function exponent (s) unitless
  transport function prefactor (sigma_E_0) same units as conductivity (S/m)
'''

constant = {'e': 1.60217662e-19,  # physical constants
            'k': 1.38064852e-23}


def model_conductivity(cp, s, sigma_E_0):
    '''
    returns the electrical conductivity (S/m)

    Args:
      cp: (float/ndarray) reduced chemical potential, unitless
      s: (int) energy exponent restricted to integer/half-integer, unitless
      sigma_E_0: (float) powerlaw prefactor, S/m

    Returns: (float/ndarray)
    '''

    if s == 0:  # s=0 requires analytic simplification
        return sigma_E_0 / (1. + np.exp(-cp))
    else:
        return sigma_E_0 * s * fdk(s - 1, cp)


def model_seebeck(cp, s):
    '''
    returns the seebeck coeficient (V/K)

    Args:
      cp: (float/ndarray) reduced chemical potential, unitless
      s: (int) energy exponent restricted to integer/half-integer, unitless

    Returns: (float/ndarray)
    '''

    if s == 0:  # s=0 requires analytic simplification
        return constant['k'] / constant['e'] * (((1. + np.exp(-cp)) *
                                                 fdk(0, cp)) - cp)
    else:
        return constant['k'] / constant['e'] * (((s + 1.) * fdk(s, cp) / s /
                                                 fdk(s - 1, cp)) - cp)
