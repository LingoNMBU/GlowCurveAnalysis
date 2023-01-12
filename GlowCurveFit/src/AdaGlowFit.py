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

    def __init__(self, data_df, metric, beta, g_max=100, N_pop=20, n_peaks=1):
        """
        Initialization

        """

        self.data = data_df  # Takes data with two columns, [Intensity, temperature]
        self.data.columns = ['Intensity', 'Temperature']
        self.data.reset_index(drop=True)
        self.beta = beta  # heating rate
        self.n_peaks = n_peaks

        self.mod_data = None

        # Parameter borders
        self.params_min = {'b': 1.01,
                           'n0': 10000,
                           'E': 0.1,
                           'Sdd': 10 ** (18),

                           'b2': 1.01,
                           'n02': 10000,
                           'E2': 0.1,
                           'Sdd2': 10 ** (18),

                           'b3': 1.01,
                           'n03': 10000,
                           'E3': 0.1,
                           'Sdd3': 10 ** (18)
                           }

        self.params_max = {'b': 6.0,
                           'n0': 20000000,
                           'E': 2.0,
                           'Sdd': 10 ** (21),

                           'b2': 6.0,
                           'n02': 20000000,
                           'E2': 2.0,
                           'Sdd2': 10 ** (21),

                           'b3': 6.0,
                           'n03': 20000000,
                           'E3': 2.0,
                           'Sdd3': 10 ** (21)
                           }

        self.mod_params = []

        if n_peaks == 1:
            self.params = ['b', 'n0', 'E', 'Sdd']
        elif n_peaks == 2:
            self.params = ['b', 'n0', 'E', 'Sdd',
                           'b2', 'n02', 'E2', 'Sdd2']
        elif n_peaks == 3:
            self.params = ['b', 'n0', 'E', 'Sdd',
                           'b2', 'n02', 'E2', 'Sdd2',
                           'b3', 'n03', 'E3', 'Sdd3']

        #  Algorithm parameters
        self.N_pop = N_pop  # Number of candidate solutions in each generations, usually 10-50, more=more diversity pr. g
        self.g_max = g_max  # Maximum number of iteration cycles
        print(f'max generations: {self.g_max}')
        print(f'population: {self.N_pop}')
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

        self.multi_params = []
        self.peak_fits = []

    def multipeak_fit(self):
        """
        Triies to fit several peaks by fitting each and subtrscting it from the original data.
        :return:
        """

        for i in range(self.n_peaks):
            self.algorythm()
            self.multi_params.append(self.best_params)
            self.peak_fits.append(self.mod_data.copy(deep=True))
            self.data.Intensity = self.data.Intensity.values - self.mod_data.Intensity.values

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
        print('x')
        print(x)
        # Creating initial values and constants
        a = np.log(2)  # Compression constant for adaptive crossover and mutation
        b = 1 / 2  # Compression constant for adaptive crossover and mutation
        fitness_last_200 = 100  # Initial value for comparing fitness change every 200 generations

        fitness_best_last1 = min(fitness)
        fitness_best_last2 = fitness_best_last1

        x_best_ind = fitness.index(fitness_best_last1)
        x_best = pop[x_best_ind]

        # print('x_best')
        # print(x_best)

        # Iteration
        g = 1  # start generation
        convergence = False

        print(self.g_max)

        while g <= self.g_max and not convergence:
            self.fitnesses.append(fitness_best_last1)
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

            x_best_ind = fitness.index(min(fitness))
            x_best = pop[x_best_ind]

            g = g + 1

            # Prints current values every 200 generation
            if g % 10 == 0:
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

        self.mod_data = self.make_fit(x_best, self.data)
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
        # print(params)

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
        # print(params)
        Sdd = params['Sdd']
        b = params['b']
        E = params['E']
        n0 = params['n0']
        beta = self.beta
        kb = self.kb

        def solve_single(Sdd, b, E, n0, t, beta, kb):
            p1 = n0 * Sdd * np.exp((-E) / (kb * t))

            p21 = 1
            p221 = ((b - 1) * (Sdd / beta)) * ((kb * t * t) / E) * np.exp(-E / (kb * t))
            p222 = (1 - 2 * ((kb * t) / E) + 6 * ((kb * t) / E) ** 2) - 24 * (((kb * t) / E) ** 3)
            p22 = p221 * p222

            p2 = (p21 + p22)

            if b - 1 == 0:
                p2b = p2 ** (-b / (1 * 10 ** (-20)))
            else:
                p2b = p2 ** (-b / (b - 1))

            intensity = p1 * p2b

            return intensity

        intensity1 = solve_single(Sdd, b, E, n0, t, beta, kb)

        if self.n_peaks == 1:
            return intensity1

        Sdd2 = params['Sdd2']
        b2 = params['b2']
        E2 = params['E2']
        n02 = params['n02']
        intensity2 = intensity1 + solve_single(Sdd2, b2, E2, n02, t, beta, kb)

        if self.n_peaks == 2:
            return intensity2

        Sdd3 = params['Sdd3']
        b3 = params['b3']
        E3 = params['E3']
        n03 = params['n03']
        intensity3 = intensity1 + intensity2 + solve_single(Sdd3, b3, E2, n03, t, beta, kb)

        if self.n_peaks == 3:
            return intensity3