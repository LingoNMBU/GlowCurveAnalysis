# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lsGlowFit import LsGlowFit

exp_data = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_P1_processed.csv')

#
exp_data.index = exp_data['Temperature measured']
exp_data = exp_data.loc[:exp_data['Temperature measured'].idxmax(), :]
exp_data = exp_data.drop(axis=1, columns=['Time', 'Temperature setpoint'])
exp_data.columns = ['Intensity', 'Temperature']
exp_data.Temperature = exp_data.Temperature + 273.15




#Fitting
params0 = [1.0e+18, 4.0, 2, 10509767, 1.0e+18, 4.0, 2, 10509767]
ls = LsGlowFit(exp_data, params0, 2, 5)
ls.fit_ls()
print(ls.result[3])
plt.plot(exp_data.Temperature, exp_data.Intensity, label='experimental')
plt.plot(exp_data.Temperature, ls.intensity_fit, label='model')
plt.legend()
plt.show()

plt.plot(np.log(ls.residues))
plt.show()


