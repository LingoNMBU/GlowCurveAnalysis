# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import numpy as np
from scipy.optimize import leastsq
from scipy.optimize import fmin_slsqp
import lmfit


class LsGlowFit:

    def __init__(self, data_df, params0, n_peaks, beta):

        self.x = data_df['Temperature'].values.astype(np.float64)
        self.y = data_df['Intensity'].values.astype(np.float64)

        self.beta = beta  # heating rate
        self.n_peaks = n_peaks
        self.params = np.array(params0)  # [Sdd, b, E, n0]
        self.result = None

        self.intensity_fit = None
        self.peak_fits = []
        self.residues = []

        self.kb = 8.617333e-5

    def fit_lm(self):
        """
        Fitting glowcurve using Levenberg Marquard using lmfit package
        :return:
        """
        xs = self.x
        ys = self.y
        beta = self.beta
        kb = self.kb

        def evaluate_1_glowpeak(x, beta, kb, Sdd_1, b_1, E_1, n0_1):

            intensities = []
            for t in x:
                p1 = n0_1 * Sdd_1 * np.exp((-E_1) / (kb * t))

                p21 = 1
                p221 = ((b_1 - 1) * (Sdd_1 / beta)) * ((kb * t * t) / E_1) * np.exp(-E_1 / (kb * t))
                p222 = (1 - 2 * ((kb * t) / E_1) + 6 * ((kb * t) / E_1) ** 2) - 24 * (
                        ((kb * t) / E_1) ** 3)
                p22 = p221 * p222

                p2 = (p21 + p22)

                b_exp = (-b_1 / (b_1 - 1))

                p2b = np.power(p2, b_exp, dtype=np.complex)

                intensity = p1 * p2b
                intensities.append(intensity)

                if not np.isfinite(intensity):
                    print('error')

            if not np.all(np.isfinite(intensities)):
                print('error')

            return np.array(intensities)

        def evaluate_2_glowpeaks(x, beta, kb,
                                 Sdd_1, b_1, E_1, n0_1,
                                 Sdd_2, b_2, E_2, n0_2):

            intensity1 = evaluate_1_glowpeak(x, beta, kb, Sdd_1, b_1, E_1, n0_1)
            intensity2 = evaluate_1_glowpeak(x, beta, kb, Sdd_2, b_2, E_2, n0_2)
            intensity = intensity1 + intensity2

            return intensity

        def evaluate_3_glowpeaks(x, beta, kb,
                                 Sdd_1, b_1, E_1, n0_1,
                                 Sdd_2, b_2, E_2, n0_2,
                                 Sdd_3, b_3, E_3, n0_3):

            intensity1 = evaluate_1_glowpeak(x, beta, kb, Sdd_1, b_1, E_1, n0_1)
            intensity2 = evaluate_1_glowpeak(x, beta, kb, Sdd_2, b_2, E_2, n0_2)
            intensity3 = evaluate_1_glowpeak(x, beta, kb, Sdd_3, b_3, E_3, n0_3)
            intensity = intensity1 + intensity2 + intensity3
            return intensity

        def evaluate_4_glowpeaks(x, beta, kb,
                                 Sdd_1, b_1, E_1, n0_1,
                                 Sdd_2, b_2, E_2, n0_2,
                                 Sdd_3, b_3, E_3, n0_3,
                                 Sdd_4, b_4, E_4, n0_4):

            intensity1 = evaluate_1_glowpeak(x, beta, kb, Sdd_1, b_1, E_1, n0_1)
            intensity2 = evaluate_1_glowpeak(x, beta, kb, Sdd_2, b_2, E_2, n0_2)
            intensity3 = evaluate_1_glowpeak(x, beta, kb, Sdd_3, b_3, E_3, n0_3)
            intensity4 = evaluate_1_glowpeak(x, beta, kb, Sdd_4, b_4, E_4, n0_4)
            intensity = intensity1 + intensity2 + intensity3 + intensity4
            return intensity

        if self.n_peaks == 1:
            gcmodel = lmfit.Model(evaluate_1_glowpeak)
        elif self.n_peaks == 2:
            gcmodel = lmfit.Model(evaluate_2_glowpeaks)
        elif self.n_peaks == 3:
            gcmodel = lmfit.Model(evaluate_3_glowpeaks)
        elif self.n_peaks == 4:
            gcmodel = lmfit.Model(evaluate_4_glowpeaks)
        else:
            gcmodel = lmfit.Model(evaluate_1_glowpeak)

        for i in range(self.n_peaks):
            gcmodel.set_param_hint(f'Sdd_{i + 1}', value=1.0e+16, min=10**6, max=10.0**22)
            gcmodel.set_param_hint(f'b_{i + 1}', value=2.0, min=0.5, max=5.0)
            gcmodel.set_param_hint(f'E_{i + 1}', value=1.2, min=0.6, max=2.5)
            gcmodel.set_param_hint(f'n0_{i + 1}', value=1.0e+5, min=10**4, max=10**12)
            #gcmodel.set_param_hint(f'b_{i + 1}', value=1.0001, vary=False)

        gcmodel.set_param_hint(f'beta', value=beta, vary=False)
        gcmodel.set_param_hint(f'kb', value=kb, vary=False)

        #gcmodel.set_param_hint(f'n_peaks', value=self.n_peaks, vary=False)

        params = gcmodel.make_params()

        # gcmodel.nan_policy = 'propagate'

        result = gcmodel.fit(ys, params, x=xs)

        self.result = result

        for peak in range(self.n_peaks):
            Sdd = result.best_values[f'Sdd_{peak+1}']
            b = result.best_values[f'b_{peak+1}']
            E = result.best_values[f'E_{peak+1}']
            n0 = result.best_values[f'n0_{peak+1}']

            self.peak_fits.append(evaluate_1_glowpeak(xs, beta, kb, Sdd, b, E, n0))

    def fit_ls(self):
        """
        fit glowcurve nonlinear least squares using scipy
        """

        params0 = []
        for i in range(self.n_peaks):
            for param in self.params:
                params0.append(param)

        print(params0)

        x = self.x
        y = self.y

        def evaluate_glow_curve_res(params, x, y):
            """
            evaluate residue of parameters
            """
            params = params
            intensities = evaluate_glow_curve(params, x)
            res = []
            for mod_intensity, exp_intensity in zip(intensities, y):
                res.append(exp_intensity - mod_intensity)

            sumres = sum(abs(np.real(res)))

            self.residues.append(sumres)
            # print('params')
            # print(params)
            # print('res sum')
            # print(sum(res))
            return res

        def evaluate_glow_curve(params, x):
            """
            Evaluates the Single Diode Equation with the given parameters
            :param params:

            :return: Intensity
            """

            def solve_single(Sdd, b, E, n0, t, beta, kb):
                p1 = n0 * Sdd * np.exp((-E) / (kb * t))

                p221 = ((b - 1) * (Sdd / beta)) * ((kb * t * t) / E) * np.exp(-E / (kb * t))
                p222 = (1 - 2 * ((kb * t) / E) + 6 * ((kb * t) / E) ** 2) - 24 * (
                        ((kb * t) / E) ** 3)
                p22 = p221 * p222

                p2 = (1 + p22)

                p2b = np.power(p2, (-b / (b - 1)), dtype=np.complex)

                # if b - 1 == 0:
                #    p2b = p2 ** (-b / (1 * 10 ** (-20)))
                # else:
                #    p2b = p2 ** (-b / (b - 1))

                intensity = p1 * p2b

                return np.real(intensity)

            glow_peaks = []

            beta = self.beta
            kb = self.kb

            for i in range(self.n_peaks):
                startind = i * 4
                Sdd = params[startind]
                b = np.abs(params[startind + 1])
                E = params[startind + 2]
                n0 = params[startind + 3]

                intensities = []
                for t in x:
                    intensities.append(solve_single(Sdd, b, E, n0, t, beta, kb))
                glow_peaks.append(np.array(intensities))

            glow_curve = np.zeros(len(x))
            for peak in range(self.n_peaks):
                glow_curve += glow_peaks[peak]

            return glow_curve

        maxfev_default = 200 * (len(params0) + 1)

        result = leastsq(func=evaluate_glow_curve_res,
                         x0=params0,
                         args=(x, y),
                         maxfev=(maxfev_default * 3),
                         full_output=True)

        self.params = result[0]
        self.result = result
        self.intensity_fit = evaluate_glow_curve(self.params, x)

        for peak in range(self.n_peaks):
            startind = peak * 4
            endind = startind + 4
            peak_params = result[0][startind:endind]
            self.peak_fits.append(self.make_glow_peak(peak_params))

        # bounds = [(0, np.inf) for par in params0]

        # out = fmin_slsqp(func=evaluate_glow_curve_res, x0=params0, bounds=bounds, args=(x, y))

        # self.params = out

        # self.intensity_fit = evaluate_glow_curve(self.params, x)

    def make_glow_peak(self, params):
        """
        Calculates intensities for a single set of params
        :param params:
        :return:
        """

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

        beta = self.beta
        kb = self.kb

        Sdd = params[0]
        b = params[1]
        E = params[2]
        n0 = params[3]

        intensities = []
        for t in self.x:
            intensities.append(solve_single(Sdd, b, E, n0, t, beta, kb))
        return intensities
