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
            'h': 6.62607004e-34,
            'hbar': 1.054571800e-34}


def sphere_dos(mstar, energy):
    '''density of states (1/Jm3)'''

    return ((2. * mstar * constant['m_e']) ** 1.5 * energy ** 0.5 /
            (2. * constant['pi']**2. * constant['hbar'] ** 3.))


def sphere_carriers(cp, T, mstar):
    '''carrier concentration (1/m3)'''

    return fdk(0.5, cp) * (4. * constant['pi'] *
                           (2. * mstar * constant['k'] * T /
                            constant['h']**2.) ** 1.5)


def sphere_conductivity(cp, T, tau_0, mstar):
    '''electrical conductivity (S/m)'''

    return (tau_0 * fdk(0, cp) * 8. * constant['pi'] *
            mstar ** 0.5 * constant['e'] ** 2. *
            (2. * constant['k'] * T) ** 1.5 / 3. / constant['h']**3.)


def sphere_seebeck(cp):
    '''seebeck coeficient (V/K)'''

    try:
        return (constant['k'] / constant['e']) * (2.0 * fdk(1, cp) /
                                                  fdk(0, cp) - cp)
    except ValueError:
        return 0.
