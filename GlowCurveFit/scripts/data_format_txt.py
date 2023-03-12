# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import numpy as np
import pandas as pd

from datahandling import DataHandling

directories = [r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\data_aarhus']

target_directory = r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\Luminescence_code_files\TL_data'
dh = DataHandling(directories)

dh.format_data_txt(target_directory)

params_df = pd.DataFrame(columns=['Im, Tm, E, R', 'Min', 'Max'])

# Im, Tm, E, R
init_values = [300000, 400, 1.0, 0.5]
min_values = [0, 273.15, 0.5, 0.000001]
max_values = [2000000, 1200, 2.5, 0.9]
params_df['Im, Tm, E, R'] = init_values
params_df['Min'] = min_values
params_df['Max'] = max_values
params_df.to_csv(r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\Luminescence_code_files\TL_params\TL_params.txt',
                 index=False,
                 sep='\t')

# lowest temp = 297.05
# highest temp = 623.35

#start number 0
# end number 660
