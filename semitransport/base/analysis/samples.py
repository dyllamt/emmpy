from semitransport.base.analysis.transport_coefficients import\
    extract_transport_function, extract_effective_mass

from os import listdir

import numpy as np

'''
this module abstracts experimental data into structured objects
'''


class Sample(object):
    '''
    abstraction of experimental transport data for a single sample

    Attributes:
        conductivity (ndarray) the conductivity data. column one is the
            temperature (K) and column two is conductivity (S/m)
        seebeck (ndarray) the Seebeck data. column one is the temperature (K)
            and column two is seebeck (V/K)
        carrier_density (float) the carrier density (1/m^3)
        name (str) a unique identifier for this sample
    '''
    def __init__(self, conductivity=None, seebeck=None, name=None,
                 carrier_density=None):
        # sorts conductivity and seebeck by temperature
        self.conductivity = conductivity[conductivity[:, 0].argsort()]
        self.seebeck = seebeck[seebeck[:, 0].argsort()]
        # assigns name and carrier_density attributes
        self.name = name
        self.carrier_density = carrier_density

    @property
    def temperature_window(self):
        '''
        touple of the overlapping temperature range in conductivity and Seebeck
        '''
        conductivity_temperatures = self.conductivity[:, 0]
        seebeck_temperatures = self.seebeck[:, 0]

        minimum_permisable = max((min(conductivity_temperatures),
                                  min(seebeck_temperatures)))
        maximum_permisable = min((max(conductivity_temperatures),
                                  max(seebeck_temperatures)))
        return (minimum_permisable, maximum_permisable)

    def get_interpolated_data(self, n_temperatures=15):
        '''
        interpolates conductivity and Seebeck data in the temperature_window

        Args:
            n_temperatures (int) number of temperatures to linearly interpolate

        Returns: (touple) arrays of temperature, Seebeck, and conductivity
        '''
        T_min, T_max = self.temperature_window
        temperatures = np.linspace(T_min, T_max, n_temperatures)
        seebecks = np.interp(
            x=temperatures, xp=self.seebeck[:, 0], fp=self.seebeck[:, 1])
        conductivities = np.interp(
            x=temperatures, xp=self.conductivity[:, 0],
            fp=self.conductivity[:, 1])

        return (temperatures, seebecks, conductivities)

    def extract_transport_coefficients(self, n_temperatures=15, s=1):
        '''
        temperature analysis is carried out for a single sample by extracting
        the transport function verses temperature. it is informative when
        checking for multi-band behavior or strange scattering mechanisms

        Args:
            n_temperatures (int) number of temperatures to analyze
            s (int) the transport exponent (specifies mechanism)

        Returns: (touple) arrays of temperatures, transport function
            prefactors, and effective masses (if known carrier-density)
        '''

        # collects interpolated data for analysis
        temperatures, seebecks, conductivities = self.get_interpolated_data(
            n_temperatures=n_temperatures)

        # extract the transport function
        transport_functions = []
        for seeb, cond, temp in zip(seebecks, conductivities, temperatures):
            transport_functions.append(
                extract_transport_function(seeb, cond, temp, s))

        # extract the effective mass if carrier-density is known
        if self.carrier_density:
            effective_masses = []
            for seeb, temp in zip(seebecks, temperatures):
                effective_masses.append(
                    extract_effective_mass(seeb, self.carrier_density, temp))
        else:
            effective_masses = None

        return (temperatures, transport_functions, effective_masses)

    @classmethod
    def from_csv(cls, name, path, carrier_density=None):
        '''
        load conductivity and seebeck from data files. the data files must be
        named according to the following scheme -- name_cond.csv/name_seeb.csv

        Args:
            path (str) path to data files, both conductivity and seebeck data
                files should be in the same path
            name (str) sample name, should be the same for both seeb and cond
            carrier_density (float) the carrier density in 1/m^3 if known
        '''

        conductivity = np.loadtxt(
            '{}/{}_conductivity.csv'.format(path, name), delimiter=', ')
        seebeck = np.loadtxt(
            '{}/{}_seebeck.csv'.format(path, name), delimiter=', ')

        return cls(conductivity=conductivity, seebeck=seebeck,
                   carrier_density=carrier_density, name=name)


class SampleSeries(object):
    '''
    a sample series contains multiple samples. a sample series will typically
    be the same base material with different doping concentrations

    Attributes:
        samples ([Sample]) list of Samples, which have cond and seeb attributes
    '''
    def __init__(self, samples=None):
        if not samples:
            self.samples = []
        else:
            self.samples = samples

    @classmethod
    def from_path(cls, path):
        '''
        load a series of sample data in path. the data files must be named
        according to the following scheme -- name_cond.csv/name_seeb.csv

        Args:
            path (str) path to data files, both conductivity and seebeck data
                files should be in the same path
        '''

        names = {
            file.split('_')[0] for file in listdir(path) if '.csv' in file}

        series = cls()
        for name in names:
            series.samples.append(Sample.from_csv(name, path))
        return series

    @staticmethod
    def jonker_analysis(seebecks, conductivities, temperature, s=1):
        '''
        a jonker analysis is typicaly applied to a series of samples with the
        same electronic structure but different doping concentrations.
        properties of each sample should be measured at the same temperature

        Args:
          seebecks ([float]) the Seebeck coefficients, V/K
          conductivities ([float]) electrical conductivities, S/m
          temperature (float) the absolute temperature, K
          s (int|half-integer) assumption of transport exponent (mechanism)

        Returns: (touple of floats) transport prefactors sigma_E_0, S/m
          - average value from all samples
          - minimum value from the samples
          - maximum value from the samples
        '''

        trans_funcs = []
        for seeb, cond in zip(seebecks, conductivities):
            trans_funcs.append(
                Sample.extract_transport_function(seeb, cond, temperature, s))
        return (np.mean(trans_funcs), min(trans_funcs), max(trans_funcs))
