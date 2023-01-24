# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

directories = [r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\data_aarhus']

from datahandling import DataHandling

dh = DataHandling(directories)
print('imported data')
dh.make_dataset()

dataset = dh.dataset
dataset.to_csv('features.csv')

plt.hist(dataset)

print('lol')
