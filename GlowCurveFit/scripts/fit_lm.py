# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lmfit

from lsGlowFit import LsGlowFit

exp_data = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_P1_processed.csv')

exp_data1 = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\CaSO4_P1_processed.csv')


exp_data.index = exp_data['Temperature measured']
exp_data = exp_data.loc[:exp_data['Temperature measured'].idxmax(), :]
exp_data = exp_data.drop(axis=1, columns=['Time', 'Temperature setpoint'])
exp_data.columns = ['Intensity', 'Temperature']
exp_data.Temperature = exp_data.Temperature + 273.15

x = exp_data['Temperature']
y = exp_data['Intensity']

#Fitting
params0 = [1.0e+18, 4.0, 2, 10509767]
ls = LsGlowFit(exp_data, params0, 2, 5)
ls.fit_lm()

result = ls.result

plt.plot(x, y, 'o')
#plt.plot(x, result.init_fit, '--', label='initial fit')
plt.plot(x, result.best_fit, '-', label='best fit')
plt.plot(exp_data.Temperature, ls.peak_fits[0], label='model peak 1')
plt.plot(exp_data.Temperature, ls.peak_fits[1], label='model peak 2')
#.plot(exp_data.Temperature, ls.peak_fits[2], label='model peak 3')
#plt.plot(exp_data.Temperature, ls.peak_fits[3], label='model peak 4')
plt.legend()
plt.show()

print(result.best_values)
print('r_squared')
print(result.rsquared)
print('redchi')
print(result.redchi)

