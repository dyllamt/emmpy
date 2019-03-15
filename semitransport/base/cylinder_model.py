from fdint import fdk  # function that implements Fermi-Dirac integrals


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
            'k': 1.38064852e-23,
            'm_e': 9.10938356e-31,
            'pi': 3.14159265,
            'hbar': 1.054571800e-34}


def cylinder_dos(mstar, l):
    '''density of states (1/Jm3)'''

    return (l * mstar * constant['m_e'] /
            (4. * constant['pi'] ** 2. * constant['hbar'] ** 2.))


def cylinder_carriers(cp, T, mstar, l):
    '''carrier concentration (1/m3)'''

    return fdk(0, cp) * (l * mstar / 2. / constant['pi']**2. /
                         constant['hbar']**2. * constant['k'] * T)


def cylinder_conductivity(cp, T, tau_0, l):
    '''electrical conductivity (S/m)'''

    return (constant['e']**2. * l / 2. / constant['pi']**2. /
            constant['hbar']**2. * constant['k'] * T * tau_0 * fdk(0, cp))


def cylinder_seebeck(cp):
    '''seebeck coefficient (V/K)'''

    try:
        return (constant['k'] / constant['e']) * (2.0 * fdk(1, cp) /
                                                  fdk(0, cp) - cp)
    except ValueError:
        return 0.
