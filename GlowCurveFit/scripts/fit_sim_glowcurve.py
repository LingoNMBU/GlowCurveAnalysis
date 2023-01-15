# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from AdaGlowFit import AdaGlowFit
from lsGlowFit import LsGlowFit

exp_data = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_P1_processed.csv')

exp_data.index = exp_data['Temperature measured']
exp_data = exp_data.loc[:exp_data['Temperature measured'].idxmax(), :]
exp_data = exp_data.drop(axis=1, columns=['Time', 'Temperature setpoint'])
exp_data.columns = ['Intensity', 'Temperature']
exp_data.Temperature = exp_data.Temperature + 273.15


adaGlow0 = AdaGlowFit(exp_data, beta=5, metric='rmse', g_max=100, N_pop=50, n_peaks=1)
params1 = {'E': 1.7, 'b': 3, 'n0': 8000000, 'Sdd': 10 ** 19}
params2 = {'E': 1.5, 'b': 2.1, 'n0': 4000000, 'Sdd': 10 ** 19}
sim_curve1 = adaGlow0.make_fit(params1, exp_data)
sim_curve2 = adaGlow0.make_fit(params2, exp_data)
sim_curve2['Intensity'] = sim_curve1['Intensity'] + sim_curve2['Intensity']

sim_curve = sim_curve2


plt.plot(sim_curve['Intensity'])
plt.show()

n_peaks = 2
adaGlow1 = AdaGlowFit(exp_data, beta=5, metric='rmse', g_max=100, N_pop=50, n_peaks=n_peaks)
adaGlow1.data = sim_curve
adaGlow1.n_peaks = n_peaks
adaGlow1.algorythm()
mod_data = adaGlow1.mod_data
#plt.plot(adaGlow1.fitnesses)
#plt.show()

params0 = [1.0e+18, 4.0, 2, 10000000]
lsGlow1 = LsGlowFit(sim_curve, params0=params0, n_peaks=n_peaks, beta=5)
print('fitting lm')
lsGlow1.fit_lm()
result = lsGlow1.result
print('lm params')
print(result.best_values)
mod_data_lm = result.best_fit
print('fitting ls')
lsGlow1.fit_ls()
mod_data_ls = result.best_fit
print('lsParams')
print(lsGlow1.params)



plt.plot(exp_data['Temperature'], mod_data_lm, '-', label='best fit lm')
plt.plot(exp_data['Temperature'], mod_data_ls, '-', label='best fit ls')
plt.plot(exp_data['Temperature'], sim_curve['Intensity'], label='sim curve')
plt.plot(mod_data['Temperature'], mod_data['Intensity'], label='Ada fit')
plt.legend()
plt.show()

