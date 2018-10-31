from transport_model import model_conductivity, model_seebeck

from os import listdir
from scipy.optimize import minimize

import numpy as np

'''
this module abstracts experimental data as objects
'''


class Sample(object):
    '''
    handles experimental transport data for a single sample

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

    def from_csv(self, name, path):
        '''
        load conductivity and seebeck from data files. the data files must be
        named according to the following scheme -- name_cond.csv/name_seeb.csv

        Args:
            path (str) path to data files, both conductivity and seebeck data
                files should be in the same path
            name (str) sample name, should be the same for both seeb and cond
        '''
        self.conductivity = np.loadtxt('{}/{}_cond.csv'.format(path, name),
                                       delimiter=',')
        self.seebeck = np.loadtxt('{}/{}_seeb.csv'.format(path, name),
                                  delimiter=',')
        self.name = name

    @staticmethod
    def extract_transport_function(seebeck, conductivity, temperature, s=1):
        '''
        given an assumption of the transport mechanism (knowledge of s),
        the transport function can be extracted from Seebeck-conductivity data
        on a single sample. optimium is found by minimizing the absolute error

        Args:
          seebeck: (float) the experimental Seebeck coefficient, V/K
          conductivity: (float) experimental electrical conductivity, S/m
          temperature: (float) the absolute temperature, K

        Returns: (float) the transport function prefactor sigma_E_0, S/m
        '''
        cp = minimize(
            lambda cp: np.abs(model_seebeck(cp, s) - np.abs(seebeck)),
            method='Nelder-Mead', x0=[0.]).x[0]
        return minimize(lambda sigma_E_0: np.abs(
            model_conductivity(cp, s, sigma_E_0) - conductivity),
            method='Nelder-Mead', x0=[0.]).x[0]


class SampleSeries(object):
    '''
    a sample series contains multiple samples of the same material but
    different doping levels (assumed that bands do not change with doping)

    Attributes:
        samples ([Sample]) list of Samples, which have cond and seeb attributes
    '''
    def __init__(self, samples=None):
        if not samples:
            self.samples = []
        else:
            self.samples = samples

    def from_path(self, path):
        '''
        load a series of sample data in path. files should be named according
        to the specifications in Sample.from_csv()

        Args:
            path (str) path to data files, both conductivity and seebeck data
                files should be in the same path
        '''
        names = {file.split('_')[0] for file in listdir(path)
                 if '.csv' in file}
        for name in names:
            self.samples.append(Sample.from_csv(name, path))

    @staticmethod
    def jonker_analysis(seebecks, conductivities, temperature, s=1):
        '''
        a jonker analysis is typicaly applied to a series of samples
        with the same electronic structure but different doping concentrations.
        properties of each sample should be measured at the same temperature

        Args:
          seebecks: ([float]) the Seebeck coefficients, V/K
          conductivities: ([float]) electrical conductivities, S/m
          temperature: (float) the absolute temperature, K
          s: (int|half-integer) assumption of transport exponent (mechanism)

        Returns: (touple of floats) transport prefactors sigma_E_0, S/m
          - average value from all samples
          - minimum value from one sample
          - maximum value from one sample
        '''

        trans_funcs = []
        for seeb, cond in zip(seebecks, conductivities):
            trans_funcs.append(
                Sample.extract_transport_function(seeb, cond, temperature, s))
        return (np.mean(trans_funcs), min(trans_funcs), max(trans_funcs))

    @staticmethod
    def temperature_analysis(seebecks, conductivities, temperatures, s=1):
        '''
        temperature analysis is carried out for a single sample
        the temperature dependence of the transport function is informative
        when checking for multi-band behavior or strange scattering mechanisms

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
