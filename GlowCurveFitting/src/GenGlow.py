# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import numpy as np
import pandas as pd

import scipy.special as special
import random


class AdaDa:
    """
    Adaptive Differential Evolution Algorithm
    """

    def __init__(self, data_df, metric, beta):
        """
        Initialization

        """

        self.data = data_df  # Takes data with two columns, [Intensity, temperature]
        self.data.columns = ['Intensity', 'Temperature']
        self.beta = beta  # heating rate

        self.mod_data = None

        # Parameter borders
        self.params_min = {'b': 0.1,
                           'n0': 100,
                           'E': 0.1,
                           'Sdd': 10 ** (17)}

        self.params_max  = {'b': 5,
                           'n0': 10000,
                           'E': 100,
                           'Sdd': 10 ** (21)}
        self.params = []

        self.params = ['b', 'n0', 'E', 'Sdd']

        #  Algorithm parameters
        self.N_pop = 20  # Number of candidate solutions in each generations, usually 10-50, more=more diversity pr. g
        self.g_max = 4000  # Maximum number of iteration cycles



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

    def algorythm(self):
        """
        Adaptive Evolutionary Algorithm implemented as described in
        Parameter estimation of solar cells and modules using an improved adaptive differential evolution algorithm
        by Lian Lian Jang et al. Adapted to Glow curves
        """

        # Initialization of population of random params
        pop = []
        fitness = []

        for i in range(self.N_pop):
            x = self.random_params_x()
            pop.append(x)
            fitness.append(self.fitness(x))

        # Creating initial values and constants
        a = np.log(2)  # Compression constant for adaptive crossover and mutation
        b = 1 / 2  # Compression constant for adaptive crossover and mutation
        fitness_last_200 = 100  # Initial value for comparing fitness change every 200 generations

        fitness_best_last1 = min(fitness)
        fitness_best_last2 = fitness_best_last1
        x_best_ind = fitness.index(np.min(fitness))
        x_best = pop[x_best_ind]

        # Iteration
        g = 1  # start generation
        convergence = False

        while g <= self.g_max and not convergence:
            A = fitness_best_last1 / fitness_best_last2
            for i in range(self.N_pop):

                # Choosing r1 and r2
                [r1, r2] = [1, 1]

                while r1 == r2:
                    r1 = random.randint(0, self.N_pop - 1)
                    r2 = random.randint(0, self.N_pop - 1)

                x_r1 = pop[r1]
                x_r2 = pop[r2]

                # Calculating A, F and Cr
                F = b * np.exp(a * A * random.uniform(0, 1))
                CR = b * np.exp(a * A * random.uniform(0, 1))

                v_i = dict([(param, (x_best[param] + F * (x_r1[param] - x_r2[param]))) for param in
                            self.params])
                # Checking Boundary conditions
                for param in self.params:
                    if v_i[param] >= self.params_max[param]:
                        v_i[param] = self.random_params_x()[param]

                    elif v_i[param] <= self.params_min[param]:
                        v_i[param] = self.random_params_x()[param]

                # Checking and doing Crossover
                u_i = dict()
                x_i = pop[i]
                for j, param in enumerate(self.params):
                    rm_ij = random.uniform(0, 1)
                    if rm_ij <= CR:
                        u_i[param] = v_i[param]
                    elif rm_ij > CR:
                        u_i[param] = x_i[param]

                # Replace individual if trial vector u is better
                u_fit = self.fitness(u_i)
                x_fit = self.fitness(x_i)
                if u_fit < x_fit:
                    pop[i] = u_i
                    fitness[i] = u_fit
                else:
                    pop[i] = x_i
                    fitness[i] = x_fit

            # Updating best fitness
            fitness_best_last2 = fitness_best_last1
            fitness_best_last1 = min(self.fitness(param) for param in pop)

            x_best_ind = fitness.index(np.min(fitness))
            x_best = pop[x_best_ind]

            g = g + 1

            # Prints current values every 200 generation
            if g % 200 == 0:
                print(f'\ngeneration: {g}, '
                      f'best params: {self.print_formatting(x_best)}, '
                      f'fitness: {format(fitness_best_last1, ".4e")}')

            # if fitness under tolerance, assume convergence
            if self.tol[self.metric] > fitness_best_last1:
                convergence = True

            # if fitness doesn't significantly change in x generations, assume convergence
            if g % 100 == 0:
                if (fitness_last_200 - fitness_best_last1) < 10e-7:
                    convergence = True
                fitness_last_200 = fitness_best_last1


        print(f'\n Algorithm  finished '
              f'\n generation: {g}'
              f'\n best params: {self.print_formatting(x_best)}'
              f'\n fitness: {format(fitness_best_last1, ".4e")}')

        self.mod_data = self.make_fit(x_best)
        self.best_params = x_best
        self.best_fit = fitness_best_last1

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

    def error(self, params, data):
        """
        Calculates the error of each set of parameters for the given dataset.
        :param params:
        :param data:
        :return:
        """

        t = self.data['Temperature measured']
        model_data = self.make_fit(params, data)
        errors = None

        if self.metric == 'rmsle':
            try:
                errors = np.log(self.data.Intensity) - np.log(model_data.Intensity)
            except RuntimeWarning as e:
                print(e)
                print('log(zero) or log(negative) warning')
        elif self.metric == 'rmse':
            errors = self.data.Intensity - model_data.Intensity
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

        temperatures = data['Temperatures measured']

        intensity = [self.solve_general_equation(params=params, t=t) for t in temperatures]

        data_mod_df = pd.DataFrame(columns=['Temperature', 'Counts'], dtype=np.longfloat)
        data_mod_df['Temperature'] = temperatures
        data_mod_df['Intensity'] = intensity

        return data_mod_df

    def solve_general_equation(self, params, t):
        """
        Evaluates the Single Diode Equation with the given parameters
        :param params:

        :return: Intensitu
        """
        Sdd = params['Sdd']
        b = params['b']
        E = params['E']
        n0 = params['n0']
        beta = self.beta
        kb = self.kb

        k1 = (kb * t * t) / E

        a = n0 * Sdd * np.exp(-E / (kb * t)) * k1

        b = 1 + ((b - 1) * (Sdd / beta)) * k1 * np.exp(-E / (kb * t)) \
            * ((1 - 2 * (k1 / t)) + 6 * (k1 / t) ** 2 - 24 * (k1 / t) ** 3)

        intensity = a * b ** (-b / (b - 1))
        return intensity
