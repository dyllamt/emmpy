from semitransport.base.transport_model import model_conductivity,\
    model_seebeck

from os import listdir
from scipy.optimize import minimize

import numpy as np

'''
this module abstracts experimental data into structured objects
'''


class Sample(object):
    '''
    abstraction of experimental transport data for a single sample

    Attributes:
        conductivity (ndarray) the experimental conductivity data. column one
            is the temperature (K) and column two is conductivity (S/m)
        seebeck (ndarray) the experimental seebeck data. column one
            is the temperature (K) and column two is seebeck (V/K)
    '''
    def __init__(self, conductivity=None, seebeck=None, name=None):
        self.conductivity = conductivity
        self.seebeck = seebeck
        self.name = name

    @property
    def temperature_overlap(self):
        """A touple of min and max overlapping temperature from seeb and cond.
        """
        conductivity_temperatures = self.conductivity[:, 0]
        seebeck_temperatures = self.seebeck[:, 0]

        minimum_permisable = max((min(conductivity_temperatures),
                                  min(seebeck_temperatures)))
        maximum_permisable = min((max(conductivity_temperatures),
                                  max(seebeck_temperatures)))
        return (minimum_permisable, maximum_permisable)

    def get_interpolated_data(self, density=15):
        """Interpolates the data in the permisable temperature range.
        """

        # sorts the conductivity and seebeck data (increasingly) by temperature
        self.seebeck = self.seebeck[self.seebeck[:, 0].argsort()]
        self.conductivity = self.conductivity[
            self.conductivity[:, 0].argsort()]

        # interpolate seebeck and conductivity over mutual temperatures
        T_min, T_max = self.temperature_overlap
        temperatures = np.linspace(T_min, T_max, density)
        seebecks = np.interp(
            x=temperatures, xp=self.seebeck[:, 0], fp=self.seebeck[:, 1])
        conductivities = np.interp(
            x=temperatures, xp=self.conductivity[:, 0],
            fp=self.conductivity[:, 1])

        return (temperatures, seebecks, conductivities)

    @classmethod
    def from_csv(cls, name, path):
        '''
        load conductivity and seebeck from data files. the data files must be
        named according to the following scheme -- name_cond.csv/name_seeb.csv

        Args:
            path (str) path to data files, both conductivity and seebeck data
                files should be in the same path
            name (str) sample name, should be the same for both seeb and cond
        '''

        conductivity = np.loadtxt(
            '{}/{}_conductivity.csv'.format(path, name), delimiter=', ')
        seebeck = np.loadtxt(
            '{}/{}_seebeck.csv'.format(path, name), delimiter=', ')
        name = name

        return cls(conductivity=conductivity, seebeck=seebeck, name=name)

    @staticmethod
    def extract_transport_function(seebeck, conductivity, temperature, s=1):
        '''
        given an assumption of the transport mechanism (knowledge of s), the
        transport function can be extracted from Seebeck-conductivity data on
        a single sample. optimium is found by minimizing the absolute error

        Args:
          seebeck (float) the experimental Seebeck coefficient, V/K
          conductivity (float) experimental electrical conductivity, S/m
          temperature (float) the absolute temperature, K
          s (int|half-integer) assumption of transport exponent (mechanism)


        Returns: (float) the transport function prefactor sigma_E_0, S/m
        '''
        cp = minimize(
            lambda cp: np.abs(model_seebeck(cp, s) - np.abs(seebeck)),
            method='Nelder-Mead', x0=[0.]).x[0]
        return minimize(lambda sigma_E_0: np.abs(
            model_conductivity(cp, s, sigma_E_0) - conductivity),
            method='Nelder-Mead', x0=[0.]).x[0]

    @staticmethod
    def temperature_analysis(seebecks, conductivities, temperatures, s=1):
        '''
        temperature analysis is carried out for a single sample by extracting
        the transport function verses temperature. it is informative when
        checking for multi-band behavior or strange scattering mechanisms

        Args:
          seebecks: ([float]) the Seebeck coefficients, V/K
          conductivities: ([float]) electrical conductivities, S/m
          temperatures: ([float]) the absolute temperatures, K
          s: (int|half-integer) assumption of transport exponent (mechanism)

        Returns: ([float]) transport function prefactors sigma_E_0, S/m
        '''

        trans_funcs = []
        for seeb, cond, temp in zip(seebecks, conductivities, temperatures):
            trans_funcs.append(
                Sample.extract_transport_function(seeb, cond, temp, s))
        return trans_funcs


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
