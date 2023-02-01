# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymcr.mcr import McrAR
from pymcr.regressors import OLS
from pymcr.constraints import ConstraintNonneg

exp_data = pd.read_csv(
    r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\CaSO4_P1_processed.csv')

exp_data.index = exp_data['Temperature measured']
exp_data = exp_data.loc[:exp_data['Temperature measured'].idxmax(), :]
exp_data = exp_data.drop(axis=1, columns=['Time', 'Temperature setpoint'])
exp_data.columns = ['Intensity', 'Temperature']
exp_data.Temperature = exp_data.Temperature + 273.15


mcrar = McrAR(c_regr=OLS(), st_regr=OLS(), c_constraints=[ConstraintNonneg()],
              st_constraints=[ConstraintNonneg()])
mcrar.fit(D, ST=initial_spectra)