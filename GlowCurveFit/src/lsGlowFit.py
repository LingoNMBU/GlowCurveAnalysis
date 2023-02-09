# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import numpy as np
from scipy.optimize import leastsq
import pandas as pd
from scipy.optimize import fmin_slsqp
import lmfit


class LsGlowFit:
    """
    Classs fittting glowcurves using nonlinear least squares
    """

    def __init__(self, data_df, n_peaks, beta, params0=None):

        self.data_df = data_df

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

    def fit_lm_1o(self):
        """
        Fitting glowcurve using Levenberg Marquard using lmfit package with the
        equation for general order kinetics
        :return:
        """
        xs = self.x
        ys = self.y
        beta = self.beta
        kb = self.kb

        def evaluate_1_glowpeak(x, kb, Tm1, Im1, E1):
            """

            :param Ts: Temperatures
            :param kb: Boltzmann constant
            :param beta: heating rate
            :param Tm1: Temperature of peak
            :param Im1: Intensity of peak
            :param E1: Energy gap to trap
            :return: Intensities
            """

            intensities = []
            for T in x:
                delta = (2 * kb * T) / E1
                delta_m = (2 * kb * Tm1) / E1

                E_kbT = E1 / (kb * T)
                TTm_Tm = (T - Tm1) / Tm1

                e1 = 1 + (E_kbT * TTm_Tm) - (((T / Tm1) ** 2) * np.exp(E_kbT * TTm_Tm) * (1 - delta)) - delta_m

                I_T = Im1 * np.exp(e1)

                intensities.append(I_T)

            return np.array(intensities)

        def evaluate_2_glowpeaks(x, kb,
                                 Tm1, Im1, E1,
                                 Tm2, Im2, E2):

            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2)
            intensity = intensity1 + intensity2

            return intensity

        def evaluate_3_glowpeaks(x, kb,
                                 Tm1, Im1, E1,
                                 Tm2, Im2, E2,
                                 Tm3, Im3, E3):
            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2)
            intensity3 = evaluate_1_glowpeak(x, kb, Tm3, Im3, E3)
            intensity = intensity1 + intensity2 + intensity3

            return intensity

        def evaluate_4_glowpeaks(x, kb,
                                 Tm1, Im1, E1,
                                 Tm2, Im2, E2,
                                 Tm3, Im3, E3,
                                 Tm4, Im4, E4):
            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2)
            intensity3 = evaluate_1_glowpeak(x, kb, Tm3, Im3, E3)
            intensity4 = evaluate_1_glowpeak(x, kb, Tm4, Im4, E4)
            intensity = intensity1 + intensity2 + intensity3 + intensity4

            return intensity

        def evaluate_5_glowpeaks(x, kb,
                                 Tm1, Im1, E1,
                                 Tm2, Im2, E2,
                                 Tm3, Im3, E3,
                                 Tm4, Im4, E4,
                                 Tm5, Im5, E5):
            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2)
            intensity3 = evaluate_1_glowpeak(x, kb, Tm3, Im3, E3)
            intensity4 = evaluate_1_glowpeak(x, kb, Tm4, Im4, E4)
            intensity5 = evaluate_1_glowpeak(x, kb, Tm5, Im5, E5)
            intensity = intensity1 + intensity2 + intensity3 + intensity4 + intensity5

            return intensity

        if self.n_peaks == 1:
            gcmodel = lmfit.Model(evaluate_1_glowpeak)
        elif self.n_peaks == 2:
            gcmodel = lmfit.Model(evaluate_2_glowpeaks)
        elif self.n_peaks == 3:
            gcmodel = lmfit.Model(evaluate_3_glowpeaks)
        elif self.n_peaks == 4:
            gcmodel = lmfit.Model(evaluate_4_glowpeaks)
        elif self.n_peaks == 5:
            gcmodel = lmfit.Model(evaluate_5_glowpeaks)
        else:
            gcmodel = lmfit.Model(evaluate_1_glowpeak)

        for i in range(self.n_peaks):

            gcmodel.set_param_hint(f'E{i + 1}', value=1.2, min=0.2, max=3.5)
            gcmodel.set_param_hint(f'Tm{i+1}', value=500, min=300.0, max=900)
            gcmodel.set_param_hint(f'Im{i+1}', value=10e+4, min=5*10e+3, max=10e+6)


        gcmodel.set_param_hint(f'beta', value=beta, vary=False)
        gcmodel.set_param_hint(f'kb', value=kb, vary=False)

        params = gcmodel.make_params()

        result = gcmodel.fit(ys, params, x=xs)

        self.result = result

        for peak in range(self.n_peaks):
            Tm = result.best_values[f'Tm{peak + 1}']
            Im = result.best_values[f'Im{peak + 1}']
            E = result.best_values[f'E{peak + 1}']

            self.peak_fits.append(evaluate_1_glowpeak(xs, kb, Tm, Im, E))

    def fit_lm_2o(self):
        """
        Fitting glowcurve using Levenberg Marquard using lmfit package with the
        equation for general order kinetics
        :return:
        """
        xs = self.x
        ys = self.y
        beta = self.beta
        kb = self.kb

        def evaluate_1_glowpeak(x, kb, Tm1, Im1, E1):
            """

            :param Ts: Temperatures
            :param kb: Boltzmann constant
            :param beta: heating rate
            :param Tm1: Temperature of peak
            :param Im1: Intensity of peak
            :param E1: Energy gap to trap
            :return: Intensities
            """

            intensities = []
            for T in x:
                E_kbT = E1 / (kb * T)
                TTm_Tm = (T-Tm1)/Tm1
                delta = (2*kb*T)/E1
                delta_m = (2*kb*Tm1)/E1

                a = 4.0 *Im1*np.exp(E_kbT*TTm_Tm)
                b = (((T/Tm1)**2) * (1-delta) * np.exp(E_kbT*TTm_Tm) + 1 + delta_m)**(-2)
                I_T = a * b

                intensities.append(I_T)

            return np.array(intensities)

        def evaluate_2_glowpeaks(x, kb,
                                 Tm1, Im1, E1,
                                 Tm2, Im2, E2):

            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2)
            intensity = intensity1 + intensity2

            return intensity

        def evaluate_3_glowpeaks(x, kb,
                                 Tm1, Im1, E1,
                                 Tm2, Im2, E2,
                                 Tm3, Im3, E3):
            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2)
            intensity3 = evaluate_1_glowpeak(x, kb, Tm3, Im3, E3)
            intensity = intensity1 + intensity2 + intensity3

            return intensity

        def evaluate_4_glowpeaks(x, kb,
                                 Tm1, Im1, E1,
                                 Tm2, Im2, E2,
                                 Tm3, Im3, E3,
                                 Tm4, Im4, E4):
            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2)
            intensity3 = evaluate_1_glowpeak(x, kb, Tm3, Im3, E3)
            intensity4 = evaluate_1_glowpeak(x, kb, Tm4, Im4, E4)
            intensity = intensity1 + intensity2 + intensity3 + intensity4

            return intensity

        def evaluate_5_glowpeaks(x, kb,
                                 Tm1, Im1, E1,
                                 Tm2, Im2, E2,
                                 Tm3, Im3, E3,
                                 Tm4, Im4, E4,
                                 Tm5, Im5, E5):
            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2)
            intensity3 = evaluate_1_glowpeak(x, kb, Tm3, Im3, E3)
            intensity4 = evaluate_1_glowpeak(x, kb, Tm4, Im4, E4)
            intensity5 = evaluate_1_glowpeak(x, kb, Tm5, Im5, E5)
            intensity = intensity1 + intensity2 + intensity3 + intensity4 + intensity5

            return intensity

        if self.n_peaks == 1:
            gcmodel = lmfit.Model(evaluate_1_glowpeak)
        elif self.n_peaks == 2:
            gcmodel = lmfit.Model(evaluate_2_glowpeaks)
        elif self.n_peaks == 3:
            gcmodel = lmfit.Model(evaluate_3_glowpeaks)
        elif self.n_peaks == 4:
            gcmodel = lmfit.Model(evaluate_4_glowpeaks)
        elif self.n_peaks == 5:
            gcmodel = lmfit.Model(evaluate_5_glowpeaks)
        else:
            gcmodel = lmfit.Model(evaluate_1_glowpeak)

        for i in range(self.n_peaks):

            gcmodel.set_param_hint(f'E{i + 1}', value=1.2, min=0.2, max=3)
            gcmodel.set_param_hint(f'Tm{i+1}', value=500, min=300.0, max=900)
            gcmodel.set_param_hint(f'Im{i+1}', value=10e+5, min=5*10e+3, max=10e+6)

        gcmodel.set_param_hint(f'beta', value=beta, vary=False)
        gcmodel.set_param_hint(f'kb', value=kb, vary=False)

        params = gcmodel.make_params()

        result = gcmodel.fit(ys, params, x=xs)

        self.result = result

        for peak in range(self.n_peaks):
            Tm = result.best_values[f'Tm{peak + 1}']
            Im = result.best_values[f'Im{peak + 1}']
            E = result.best_values[f'E{peak + 1}']

            self.peak_fits.append(evaluate_1_glowpeak(xs, kb, Tm, Im, E))

    def fit_lm(self):
        """
        Fitting glowcurve using Levenberg Marquard using lmfit package with the
        equation for general order kinetics
        :return:
        """
        xs = self.x
        ys = self.y
        beta = self.beta
        kb = self.kb

        def evaluate_1_glowpeak(x, kb, Tm1, Im1, E1, b1):

            intensities = []
            for T in x:

                delta = (2 * kb * T) * E1
                delta_m = (2 * kb * Tm1) * E1
                Zm = 1 + (b1 - 1) * delta_m
                E_kbT = E1 / (kb * T)
                TTm_Tm = (T - Tm1) / Tm1

                p1 = Im1 * b1 ** (b1 / (b1 - 1)) * np.exp(E_kbT * TTm_Tm)

                p2 = ((b1 - 1) * (1 - delta) * ((T / Tm1) ** 2) * np.exp(E_kbT * TTm_Tm) + Zm) ** (-b1 / (b1 - 1))

                intensity = p1 * p2
                intensities.append(intensity)

                if not np.isfinite(intensity):
                    print('error')

            if not np.all(np.isfinite(intensities)):
                print('error')

            return np.array(intensities)

        def evaluate_2_glowpeaks(x, kb,
                                 Tm1, Im1, E1, b1,
                                 Tm2, Im2, E2, b2):

            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1, b1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2, b2)
            intensity = intensity1 + intensity2

            return intensity

        def evaluate_3_glowpeaks(x, kb,
                                 Tm1, Im1, E1, b1,
                                 Tm2, Im2, E2, b2,
                                 Tm3, Im3, E3, b3):
            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1, b1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2, b2)
            intensity3 = evaluate_1_glowpeak(x, kb, Tm3, Im3, E3, b3)
            intensity = intensity1 + intensity2 + intensity3

            return intensity

        def evaluate_4_glowpeaks(x, kb,
                                 Tm1, Im1, E1,
                                 Tm2, Im2, E2,
                                 Tm3, Im3, E3,
                                 Tm4, Im4, E4):
            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2)
            intensity3 = evaluate_1_glowpeak(x, kb, Tm3, Im3, E3)
            intensity4 = evaluate_1_glowpeak(x, kb, Tm4, Im4, E4)
            intensity = intensity1 + intensity2 + intensity3 + intensity4

            return intensity

        def evaluate_5_glowpeaks(x, kb,
                                 Tm1, Im1, E1,
                                 Tm2, Im2, E2,
                                 Tm3, Im3, E3,
                                 Tm4, Im4, E4,
                                 Tm5, Im5, E5):
            intensity1 = evaluate_1_glowpeak(x, kb, Tm1, Im1, E1)
            intensity2 = evaluate_1_glowpeak(x, kb, Tm2, Im2, E2)
            intensity3 = evaluate_1_glowpeak(x, kb, Tm3, Im3, E3)
            intensity4 = evaluate_1_glowpeak(x, kb, Tm4, Im4, E4)
            intensity5 = evaluate_1_glowpeak(x, kb, Tm5, Im5, E5)
            intensity = intensity1 + intensity2 + intensity3 + intensity4 + intensity5

            return intensity

        if self.n_peaks == 1:
            gcmodel = lmfit.Model(evaluate_1_glowpeak)
        elif self.n_peaks == 2:
            gcmodel = lmfit.Model(evaluate_2_glowpeaks)
        elif self.n_peaks == 3:
            gcmodel = lmfit.Model(evaluate_3_glowpeaks)
        elif self.n_peaks == 4:
            gcmodel = lmfit.Model(evaluate_4_glowpeaks)
        elif self.n_peaks == 5:
            gcmodel = lmfit.Model(evaluate_5_glowpeaks)
        else:
            gcmodel = lmfit.Model(evaluate_1_glowpeak)

        for i in range(self.n_peaks):

            gcmodel.set_param_hint(f'E{i + 1}', value=1.2, min=0.2, max=3)
            gcmodel.set_param_hint(f'Tm{i+1}', value=500, min=300.0, max=900)
            gcmodel.set_param_hint(f'Im{i+1}', value=10e+5, min=5*10e+3, max=10e+6)
            gcmodel.set_param_hint(f'b{i + 1}', value=3, min=1.01, max=100)

        gcmodel.set_param_hint(f'beta', value=beta, vary=False)
        gcmodel.set_param_hint(f'kb', value=kb, vary=False)


        # gcmodel.set_param_hint(f'n_peaks', value=self.n_peaks, vary=False)

        params = gcmodel.make_params()

        # gcmodel.nan_policy = 'propagate'

        result = gcmodel.fit(ys, params, x=xs)

        self.result = result

        for peak in range(self.n_peaks):
            Tm = result.best_values[f'Tm{peak + 1}']
            Im = result.best_values[f'Im{peak + 1}']
            E = result.best_values[f'E{peak + 1}']
            b = result.best_values[f'b{peak + 1}']


            self.peak_fits.append(evaluate_1_glowpeak(xs, kb, Tm, Im, E, b))

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

    def make_sim_glowcurve(self, params, order):
        """
        makes dataframe of simulated glowcurve
        :param params:
        :return:
        """

        temperatures = self.x

        if order == 'first':
            intensity = self.make_first_order_glowpeak(params)

        data_mod_df = pd.DataFrame(columns=['Temperature', 'Intensity'], dtype=np.longfloat)
        data_mod_df['Temperature'] = temperatures
        data_mod_df['Intensity'] = intensity

        data_mod_df.index = data_mod_df.Temperature


        return data_mod_df

    def make_first_order_glowpeak(self, params):
        """

        :param params: params
        :return: intensities
        """
        xs = self.x
        kb = self.kb
        Tm1 = params['Tm1']
        Im1 = params['Im1']
        E1 = params['E1']

        def evaluate_1_glowpeak(x, kb, Tm1, Im1, E1):
            """

            :param Ts: Temperatures
            :param kb: Boltzmann constant
            :param beta: heating rate
            :param Tm1: Temperature of peak
            :param Im1: Intensity of peak
            :param E1: Energy gap to trap
            :return: Intensities
            """

            intensities = []
            for T in x:
                delta = (2 * kb * T) / E1
                delta_m = (2 * kb * Tm1) / E1

                E_kbT = E1 / (kb * T)
                TTm_Tm = (T - Tm1) / Tm1

                e1 = 1 + E_kbT * TTm_Tm - ((T / Tm1) ** 2) * np.exp(E_kbT * TTm_Tm) * (1 - delta) - delta_m

                I_T = Im1 * np.exp(e1)

                intensities.append(I_T)

            return np.array(intensities)

        intensities1 = evaluate_1_glowpeak(xs, kb, Tm1, Im1, E1)

        return intensities1
