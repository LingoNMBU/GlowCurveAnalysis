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
adaGlow1 = AdaGlowFit(exp_data, beta=5, metric='rmse')

params1 = {'E': 2.05, 'b': 3, 'n0': 8000000, 'Sdd': 10 ** 19}
params2 = {'E': 1.1*2.05, 'b': 3, 'n0': 8000000, 'Sdd': 10 ** 19}
params3 = {'E': 2.05, 'b': 1.1*3.0, 'n0': 8000000, 'Sdd': 10 ** 19}
params4 = {'E': 2.05, 'b': 3, 'n0': 1.1*8000000, 'Sdd': 10 ** 19}
params5 = {'E': 2.05, 'b': 3, 'n0': 8000000, 'Sdd': 1.1*10 ** 19}
sim_curve = adaGlow1.make_fit(params1, exp_data)
adaGlow1.data = sim_curve

print('1:')
print(adaGlow1.fitness(params1))
print('2:')
print(adaGlow1.fitness(params2))
print('3:')
print(adaGlow1.fitness(params3))
print('4:')
print(adaGlow1.fitness(params4))
print('5:')
print(adaGlow1.fitness(params5))



