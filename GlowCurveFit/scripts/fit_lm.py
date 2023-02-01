# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lmfit

from lsGlowFit import LsGlowFit

LTB_P1 = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_P1_processed.csv')
CASO4_P1 = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\CaSO4_P1_processed.csv')

LTB_2a = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_agg_2a.csv')
LTB_1a3a = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_agg_1a3a.csv')
LTB_1b = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_agg_1b.csv')
LTB_1c = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_agg_1c.csv')

exp_data = LTB_P1

#Preprocessing
exp_data.index = exp_data['Temperature measured']
exp_data = exp_data.loc[:exp_data['Temperature measured'].idxmax(), :]

exp_data = exp_data.drop(axis=1, columns=['Time', 'Temperature setpoint'])
exp_data.columns = ['Intensity', 'Temperature']
exp_data.Temperature = exp_data.Temperature + 273.15


x = exp_data['Temperature']
y = exp_data['Intensity']

#Fitting
params0 = [1.0e+18, 4.0, 2, 10509767]
ls = LsGlowFit(data_df=exp_data, params0=params0, n_peaks=2, beta=5)
print('fitting  with Levenberg-Marquardt')
ls.fit_lm_1o()

result = ls.result

plt.plot(x, y, 'o')
#plt.plot(x, result.init_fit, '--', label='initial fit')
plt.plot(x, result.best_fit, '-', label='best fit')
plt.plot(exp_data.Temperature, ls.peak_fits[0], label='model peak 1')
plt.plot(exp_data.Temperature, ls.peak_fits[1], label='model peak 2')
#plt.plot(exp_data.Temperature, ls.peak_fits[2], label='model peak 3')
#plt.plot(exp_data.Temperature, ls.peak_fits[3], label='model peak 4')
plt.legend()
plt.show()

print(result.best_values)
print('r_squared')
print(result.rsquared)
print(result.fit_report())

