# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import numpy as np
from scipy.optimize import leastsq


class LsGlowFit:

    def __init__(self, data_df, params0, n_peaks, beta):

        self.x = data_df['Temperature'].values
        self.y = data_df['Intensity'].values

        self.beta = beta  # heating rate
        self.n_peaks = n_peaks
        self.params = params0  # [Sdd, b, E, n0]
        self.result = None

        self.intensity_fit = None
        self.residues = []

        self.kb = 8.617333e-5



    def fit_ls(self):
        """
        fit least squares
        """

        if self.n_peaks == 1:
            params0 = [1.0e+18, 4.0, 2, 10509767]
        elif self.n_peaks == 2:
            params0 = [1.0e+18, 4.0, 2, 10509767,
                       1.0e+18, 4.0, 2, 10509767]
        else:
            params0 = [1.0e+18, 4.0, 2, 10509767]

        x = self.x
        y = self.y

        #print(x)
        #print(y)

        def evaluate_glow_curve_res(params, x, y):
            """
            evaluate residue of parameters
            """
            intensities = evaluate_glow_curve(params, x)
            res = []
            for mod_intensity, exp_intensity in zip(intensities, y):
                res.append(exp_intensity - mod_intensity)

            self.residues.append(sum(res))
            #print('params')
            #print(params)
            #print('res sum')
            #print(sum(res))
            return res

        def evaluate_glow_curve(params, x):
            """
            Evaluates the Single Diode Equation with the given parameters
            :param params:

            :return: Intensitu
            """
            # print(params)
            Sdd = params[0]
            b = params[1]
            E = params[2]
            n0 = params[3]
            beta = self.beta
            kb = self.kb

            def solve_single(Sdd, b, E, n0, t, beta, kb):
                p1 = n0 * Sdd * np.exp((-E) / (kb * t))

                p21 = 1
                p221 = ((b - 1) * (Sdd / beta)) * ((kb * t * t) / E) * np.exp(-E / (kb * t))
                p222 = (1 - 2 * ((kb * t) / E) + 6 * ((kb * t) / E) ** 2) - 24 * (
                            ((kb * t) / E) ** 3)
                p22 = p221 * p222

                p2 = (p21 + p22)

                if b - 1 == 0:
                    p2b = p2 ** (-b / (1 * 10 ** (-20)))
                else:
                    p2b = p2 ** (-b / (b - 1))

                intensity = p1 * p2b

                return intensity

            intensity1 = []
            for t in x:
                intensity1.append(solve_single(Sdd, b, E, n0, t, beta, kb))

            if self.n_peaks == 1:
                return np.array(intensity1)

            Sdd2 = params[4]
            b2 = params[5]
            E2 = params[6]
            n02 = params[7]

            intensity2 = []
            for t in x:
                intensity2.append(solve_single(Sdd2, b2, E2, n02, t, beta, kb))

            if self.n_peaks == 2:
                return np.array(intensity1) + np.array(intensity2)

            Sdd3 = params[8]
            b3 = params[9]
            E3 = params[10]
            n03 = params[11]

            intensity3 = []
            for t in x:
                intensity3.append(solve_single(Sdd3, b3, E3, n03, t, beta, kb))

            if self.n_peaks == 3:
                return np.array(intensity1) + np.array(intensity2) + np.array(intensity3)

        result = leastsq(func=evaluate_glow_curve_res, x0=params0, args=(x, y), full_output=True)

        self.params = result[0]
        self.result = result
        self.intensity_fit = evaluate_glow_curve(self.params, x)






