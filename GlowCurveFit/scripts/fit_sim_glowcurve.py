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


exp_data.columns = ['Intensity', 'Temperature']
exp_data.Temperature = exp_data.Temperature + 273.15
#exp_data.Intensity = exp_data.Intensity.values / exp_data.Intensity.max


adaGlow1 = AdaGlowFit(exp_data, beta=5, metric='rmse', g_max=100, N_pop=50)

params1 = {'E': 1.7, 'b': 3, 'n0': 8000000, 'Sdd': 10 ** 19}
sim_curve = adaGlow1.make_fit(params1, exp_data)

plt.plot(sim_curve['Intensity'])
plt.show()

adaGlow1.data = sim_curve


adaGlow1.algorythm()

mod_data = adaGlow1.mod_data

plt.plot(sim_curve['Temperature'], sim_curve['Intensity'], label='experimental')
plt.plot(mod_data['Temperature'], mod_data['Intensity'], label='model fit')
plt.legend()
plt.show()

plt.plot(adaGlow1.fitnesses)
plt.show()