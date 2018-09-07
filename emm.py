# Maxwell T. Dylla
# Northwestern University
# M-Dylla@u.northwestern.edu


'''
This module computes the theoretical transport coeficients of semiconductors.

The model assumes that there is an underlying transport function (sigma_E),
which is a powerlaw in energy space. The powerlaw exponent (s) is indicitive of
the charge transport mechanism. For example, s=1 corresponds to band transport
limited by non-polar phonon scattering. The powerlaw prefactor (sigma_E_0)
indicates the "quality" of transport.

Every transport coefficient is a function of the reduced chemical potential
(cp), which is the chemical potential (mu) divided by boltzmann's constant and
temperature (mu/kT).

The reduced chemical potential (cp) is unitless, the exponent (s) is unitless,
and the prefactor (sigma_E_0) has the same units as conductivity (S/m).

More information on this model can be found at 
https://www.nature.com/articles/nmat4784
'''

from fdint import fdk  # function that implements Fermi-Dirac integrals

constant = {'e': 1.60217662e-19,  # physical constants
            'k': 1.38064852e-23,
            'h': 6.62607004e-34,
            'm_e': 9.10938356e-31,
            'pi': 3.14159265,
            'hbar': 1.054571800e-34}


def conductivity(cp, s, sigma_E_0):
    '''
    returns the electrical conductivity (S/m)
    
    Args:
      cp: (float/ndarray) reducted chemical potential, unitless
      s: (int) energy exponent restricted to integer/half-integer, unitless
      sigma_E_0: (float) powerlaw prefactor, S/m
      
    Returns: (float/ndarray)
    '''

    if s==0:  # s=0 requires analytic simplification
      return sigma_E_0 / (1. + np.exp(-cp))
    else:
      return sigma_E_0 * s * fdk(s - 1, )


def seebeck(cp, s):
    '''
    returns the seebeck coeficient (V/K)
    
    Args:
      cp: (float/ndarray) reducted chemical potential, unitless
      s: (int) energy exponent restricted to integer/half-integer, unitless
      
    Returns: (float/ndarray)
    '''

    if s==0:  # s=0 requires analytic simplification
        return constant['k'] / constant['e'] * (((1. + np.exp(-cp)) *
                                                 fdk(0, cp)) - cp)
    else:
        return constant['k'] / constant['e'] * (((s + 1.) * fdk(s, cp) / s /
                                                 fdk(s - 1, cp)) - cp)
