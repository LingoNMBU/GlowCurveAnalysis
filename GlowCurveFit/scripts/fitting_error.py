# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from AdaGlowFit import AdaGlowFit

directory = r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\curve_fits\fit_info'

paths = os.listdir(directory)

heatmap_LTB = pd.DataFrame(columns=['first', 'second', 'general'],
                           index=range(1, 6),
                           dtype='float32')
heatmap_CaSO4 = pd.DataFrame(columns=['first', 'second', 'general'],
                             index=range(1, 6),
                             dtype='float32')

#heatmap_LTB.dtypes = ['float32', 'float32', 'float32']
#heatmap_LTB.dtypes = ['float32', 'float32', 'float32']

for filename in os.listdir(directory):
     path = os.path.join(directory, filename)
     file = pd.read_pickle(path)

     if filename.split('_')[0] == 'CaSO4':
          order = file['order'].values[0]
          peaks = file['n_peaks'].values[0]

          heatmap_CaSO4.loc[peaks,order] = file['rmse'].values[0]

     if filename.split('_')[0] == 'LTB':
          order = file['order'].values[0]
          peaks = file['n_peaks'].values[0]

          heatmap_LTB.loc[peaks,order] = file['rmse'].values[0]

LTB_array = np.array(heatmap_LTB)

plt.title('LTB RMSE')
sns.heatmap(heatmap_LTB, cmap="crest", vmin=0, annot=True)

plt.show()
plt.title('CaSO4 RMSE')
sns.heatmap(heatmap_CaSO4, cmap="crest", vmin=0, annot=True)


plt.show()

