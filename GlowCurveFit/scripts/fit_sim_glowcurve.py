# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from AdaGlowFit import AdaGlowFit
from lsGlowFit import LsGlowFit

# exp_data = pd.read_csv(
#     r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_P1_processed.csv')

exp_data = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\CaSO4_P1_processed.csv')

exp_data.index = exp_data['Temperature measured']
exp_data = exp_data.loc[:exp_data['Temperature measured'].idxmax(), :]
exp_data = exp_data.drop(axis=1, columns=['Time', 'Temperature setpoint'])
exp_data.columns = ['Intensity', 'Temperature']
exp_data.Temperature = exp_data.Temperature + 273.15

n_peaks = 1

adaGlow0 = AdaGlowFit(exp_data, beta=5, metric='rmse', g_max=100, N_pop=50, n_peaks=n_peaks)
lsGlow1 = LsGlowFit(exp_data, n_peaks=n_peaks, beta=5)

gen_params1 = {'E': 1.7, 'b': 3.0, 'n0': 800000, 'Sdd': 10 ** 19}
gen_params2 = {'E': 1.5, 'b': 2.1, 'n0': 4000000, 'Sdd': 10 ** 19}
gen_sim_curve1 = adaGlow0.make_fit(gen_params1, exp_data)
gen_sim_curve2 = adaGlow0.make_fit(gen_params2, exp_data)
gen_sim_curve2['Intensity'] = gen_sim_curve1['Intensity'] + gen_sim_curve2['Intensity']

o1_params1 = {'kb': 8.617333e-05, 'Tm1': 550, 'Im1': 200000, 'E1': 0.6}
o1_params2 = {'kb': 8.617333e-05, 'Tm1': 540, 'Im1': 450000, 'E1': 1.3}
o1_sim_curve1 = lsGlow1.make_sim_glowcurve(o1_params1, order='first')
o1_sim_curve2 = lsGlow1.make_sim_glowcurve(o1_params2, order='first')
o1_sim_curve3 = lsGlow1.make_sim_glowcurve(o1_params2, order='first')
o1_sim_curve3['Intensity'] = o1_sim_curve1['Intensity'].values + \
                             o1_sim_curve2['Intensity'].values

sim_curve = exp_data
n_peaks = 2
params0 = [1.0e+18, 4.0, 2, 700000]
lsGlow1 = LsGlowFit(data_df=sim_curve, n_peaks=n_peaks, beta=5, params0=params0)
print('fitting lm')
lsGlow1.fit_lm()
result = lsGlow1.result
print('lm params')
print(result.best_values)
mod_data_lm = result.best_fit

print(result.fit_report())

res = sim_curve['Intensity'].values - mod_data_lm

fig = plt.figure()
gs = GridSpec(2, 1, height_ratios=[3, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Curve plot
ax1.plot(exp_data['Temperature'], sim_curve['Intensity'], label='CaSO4 P1 agg', color='black')
ax1.plot(exp_data['Temperature'], mod_data_lm, '-', label='best fit lm', linestyle='dashdot')
# plt.plot(exp_data['Temperature'], mod_data_ls, '-', label='best fit ls')
# plt.plot(exp_data['Temperature'], o1_sim_curve1['Intensity'], label='sim curve 1')
# plt.plot(exp_data['Temperature'], o1_sim_curve2['Intensity'], label='sim curve 2')

for i in range(n_peaks):
    ax1.plot(exp_data.Temperature, lsGlow1.peak_fits[i],
             label=f'model peak {i+1}',
             linestyle='dashed')


# Res plot
ax2.scatter(exp_data['Temperature'], res, s=0.5, color='black')
ax2.plot(exp_data['Temperature'], [0 for _ in exp_data['Temperature']],
         linestyle='dashed',
         color='salmon')

# Garnityr
ax1.set_title(f'Second order curve fit, peaks : {n_peaks}')
ax1.set_xlabel('Temperature [K]')
ax1.set_ylabel('Intensity [Counts]')

ax2.set_xlabel('Temperature [K]')
ax2.set_ylabel('Residue [Counts]')

ax1.legend()
plt.show()

