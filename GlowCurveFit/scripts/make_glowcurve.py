# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from AdaGlowFit import AdaGlowFit

exp_data = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_P1_processed.csv')

exp_data.index = exp_data['Temperature measured']
exp_data = exp_data.loc[:exp_data['Temperature measured'].idxmax(), :]
exp_data = exp_data.drop(axis=1, columns=['Time', 'Temperature setpoint'])
plt.plot(exp_data['Counts measured'])
plt.show()

exp_data.columns = ['Intensity', 'Temperature']
exp_data.Temperature = exp_data.Temperature + 273.15

adaGlow1 = AdaGlowFit(exp_data, beta=5, metric='rmse')

params1 = {'b': 4.1863, 'n0': 10509767.1371, 'E': 1.9313, 'Sdd': 1.1433220782299776e+18}
params2 = {'b': 3.1184, 'n0': 12015744.0938, 'E': 4.095, 'Sdd': 4.9394834057992254e+20}
params3 = {'b': 3.1184, 'n0': 12015744.0938, 'E': 4.095, 'Sdd': 4.9394834057992254e+20}

sim_curve1 = adaGlow1.make_fit(params1, exp_data)
sim_curve2 = adaGlow1.make_fit(params2, exp_data)
sim_curve3 = adaGlow1.make_fit(params3, exp_data)

sim_curve_comb = sim_curve1 + sim_curve2

#plt.plot(exp_data.Temperature, exp_data['Intensity'], label = 'experimental')
plt.plot(sim_curve1['Intensity'], label = 'model fit')
#plt.plot(sim_curve2['Intensity'], label = 'model fit 2')
#plt.plot(sim_curve_comb['Intensity'], label = 'model fit combined')
plt.legend()
plt.show()
