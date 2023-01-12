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

adaGlow1 = AdaGlowFit(exp_data.copy(deep=True), beta=5, metric='rmse', g_max=500, N_pop=50, n_peaks=3)

adaGlow1.algorythm()

mod_data = adaGlow1.mod_data

plt.plot(exp_data['Temperature'], exp_data['Intensity'], label='experimental')
plt.plot(mod_data['Temperature'], mod_data['Intensity'], label='model peak ')

plt.legend()
plt.show()

plt.plot(adaGlow1.fitnesses)
plt.show()