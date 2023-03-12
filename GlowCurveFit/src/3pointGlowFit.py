# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import numpy as np
import pandas as pd

import scipy.special as special
import random


class AdaGlowFit:
    """
    Adaptive Differential Evolution Algorithm fitting a glow curve
    """

    def __init__(self, data_df, metric, beta):
        """
        Initialization

        """

        self.data = data_df  # Takes data with two columns, [Intensity, temperature]
        self.data.columns = ['Intensity', 'Temperature']
        self.data.reset_index(drop=True)
        self.beta = beta  # heating rate

        self.mod_data = None

        # Parameter borders
        self.params_min = {'b': 1.0,
                           'n0': 1000000,
                           'E': 0.1,
                           'Sdd': 10 ** (18)}

        self.params_max  = {'b': 6.0,
                           'n0': 20000000,
                           'E': 5.0,
                           'Sdd': 10 ** (21)}
        self.mod_params = []

        self.params = ['b', 'n0', 'E', 'Sdd']

        #  Algorithm parameters
        self.N_pop = 30  # Number of candidate solutions in each generations, usually 10-50, more=more diversity pr. g
        self.g_max = 300  # Maximum number of iteration cycles
        self.fitnesses = []



        # Fitting parameters
        self.metrics = ['rmsle', 'rmse']
        if metric in self.metrics:
            self.metric = metric
            print(f'evaluation metric: {self.metric}')
        else:
            print('invalid metric selected, default to rmse')
            self.metric = 'rmse'

        # Tolerances, adjustable
        self.tol = {'rmse': 10e-2, 'rmsle': 0.001}
        print(f'convergence tolerance: {self.tol[self.metric]}')

        self.kb = 8.617e-5  # Boltzmann constant
        self.q = 1.602176634 * 10 ** (-19)  # elementary charge

        # Fitted parameters
        self.best_params = None
        self.best_fit = None

    def ThreePointAnalysis(self):

        inds = self.data.index.sample(n=3)
        A_intensity = self.data.Intensity.cumsum()

        [Ix, Iy, Iz] = self.data.Intensity.iloc[inds].values
        [Tx, Ty, Tz] = self.data.Temperature.iloc[inds].values
        [Ax, Ay, Az] = A_intensity.iloc[inds].values

        y = Ix/Iy
        z =

        b1 = Ty*(Tx - Tz)*np.log(y)





    @staticmethod
    def print_formatting(params):
         """
         Formats params for printing
        :param params: parameters
        :return: rounded parameters
         """
         print_params = params.copy()
         for param in print_params:
            if param == 'Is2' or param == 'Is1':
                print_params[param] = format(print_params[param], '.4e')
            elif param == 'Rsh':
                print_params[param] = round(print_params[param])
            else:
                print_params[param] = round(print_params[param], 4)
         return print_params

    def fitness(self, params):

        errors = self.error(params)
        #print(params)

        error_sqrd_sum = np.sum([(error ** 2) for error in errors], axis=0)
        N = len(errors)
        fitness = np.sqrt((1 / N) * error_sqrd_sum)

        return fitness

    def random_params_x(self):
        new_x = dict()
        for param in self.params:
            mini = self.params_min[param]
            maxi = self.params_max[param]
            new_x[param] = mini + (maxi - mini) * random.uniform(0, 1)
        return new_x

    def error(self, params):
        """
        Calculates the error of each set of parameters for the given dataset.
        :param params:
        :param data:
        :return:
        """

        model_data = self.make_fit(params, self.data)
        errors = None

        if self.metric == 'rmsle':
            try:
                errors = np.log(self.data.Intensity) - np.log(model_data.Intensity)
            except RuntimeWarning as e:
                print(e)
                print('log(zero) or log(negative) warning')
        elif self.metric == 'rmse':
            errors = self.data['Intensity'].values - model_data['Intensity'].values
        else:
            print(f'invalid metric name "{self.metric}" given, default set to rmse')
            errors = self.data.Intensity - model_data.Intensity

        return errors

    def make_fit(self, params, data):
        """
        create glow curve
        :param data:
        :param params: dictionary of optimal params
        """

        temperatures = data['Temperature'].values

        intensity = [self.solve_general_equation(params=params, t=t) for t in temperatures]

        data_mod_df = pd.DataFrame(columns=['Temperature', 'Intensity'], dtype=np.longfloat)
        data_mod_df['Temperature'] = temperatures
        data_mod_df['Intensity'] = intensity

        return data_mod_df

    def solve_general_equation(self, params, t):
        """
        Evaluates the Single Diode Equation with the given parameters
        :param params:

        :return: Intensitu
        """
        #print(params)
        try:
            Sdd = params['Sdd']
        except ValueError:
            print('lol')
        b = params['b']
        E = params['E']
        n0 = params['n0']
        beta = self.beta
        kb = self.kb

        #print(params)

        p1 = n0 * Sdd * np.exp((-E)/(kb*t))

        p21 = 1
        p221 = ((b-1)*(Sdd/beta)) * ((kb*t*t)/E) * np.exp(-E/(kb*t))
        p222 = (1 - 2*((kb*t)/E) + 6*((kb*t)/E)**2) - 24*(((kb*t)/E)**3)
        p22 = p221 * p222

        p2 = (p21 + p22)

        if b-1 == 0:
            p2b = p2 ** (-b / (1*10**(-20)))
        else:
            p2b = p2 ** (-b / (b - 1))


        intensity = p1 * p2b

        return intensity
