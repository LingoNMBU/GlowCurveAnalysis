# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import os
import numpy as np
import pandas as pd

dataset_CaSO4 = pd.DataFrame()
dataset_LTB = pd.DataFrame()
directory = r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\curve_fits\3peaks2order'

samples_dict = {}
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    name_parts = filename.split('_')
    substance = name_parts[0]
    position = name_parts[1]
    sample_nr = name_parts[2]
    order = name_parts[4]
    peaks = name_parts[5][0]
    filetype = name_parts[6].split('.')[0]

    if filetype == 'info':
        info_df = pd.read_pickle(f)

        #Sort peaks, only works for 3 peaks atm
        sample_df = info_df.copy(deep=True)

        Tm1 = info_df['Tm1'].values[0]
        Tm2 = info_df['Tm2'].values[0]
        Tm3 = info_df['Tm3'].values[0]

        inds = [0,1,2]
        sorted_inds = np.argsort([Tm1, Tm2, Tm3])

        for ind in inds:
            sample_df[f'Tm{ind+1}'] = info_df[f'Tm{sorted_inds[ind]+1}']
            sample_df[f'Im{ind+1}'] = info_df[f'Im{sorted_inds[ind]+1}']
            sample_df[f'E{ind+1}'] = info_df[f'E{sorted_inds[ind]+1}']



    #    if Tm1 > Tm2:
    #        if Tm1 > Tm3:
    #            # Tm1 is largest

    #            sample_df['Tm3'] = info_df['Tm1']
    #            sample_df['Im3'] = info_df['Im1']
    #            sample_df['E3'] = info_df['E1']

    #            if Tm2 > Tm3:
    #                # Tm2 is next largest
    #                # Tm3 is smallest
    #                sample_df['Tm1'] = info_df['Tm3']
    #                sample_df['Im1'] = info_df['Im3']
    #                sample_df['E1'] = info_df['E3']

    #            else:
    #                # Tm3 is next largest
    #                # Tm2 is smallest
    #                sample_df['Tm1'] = info_df['Tm2']
    #                sample_df['Im1'] = info_df['Im2']
    #                sample_df['E1'] = info_df['E2']

    #                sample_df['Tm2'] = info_df['Tm3']
    #                sample_df['Im2'] = info_df['Im3']
    #                sample_df['E2'] = info_df['E3']
    #        else:
    #            # Tm1 is next largest
    #            # Tm3 is largest
    #            # Tm2 is smallest
    #            sample_df['Tm2'] = info_df['Tm1']
    #            sample_df['Im2'] = info_df['Im1']
    #            sample_df['E2'] = info_df['E1']

    #            sample_df['Tm1'] = info_df['Tm2']
    #            sample_df['Im1'] = info_df['Tm2']
    #            sample_df['E1'] = info_df['E2']
    #    else:
    #        if Tm2 > Tm3:

    #            sample_df['Tm3'] = info_df['Tm2']
    #            sample_df['Im3'] = info_df['Im2']
    #            sample_df['E3'] = info_df['E2']

    #            if Tm3 > Tm1:
    #                sample_df['Tm2'] = info_df['Tm3']
    #                sample_df['Im2'] = info_df['Im3']
    #                sample_df['E2'] = info_df['E3']





        sample_df[f'position'] = position
        sample_df[f'sample_nr'] = sample_nr
        column_names = []

        # Change column names to be substance-specific
        for column_name in sample_df.columns:
            column_names.append(f'{substance}_{column_name}')
        sample_df.columns = column_names

        if substance == 'CaSO4':
            dataset_CaSO4 = pd.concat([dataset_CaSO4, sample_df], axis=0)
        else:
            dataset_LTB = pd.concat([dataset_LTB, sample_df], axis=0)


dataset = pd.concat([dataset_CaSO4.reset_index(drop=True), dataset_LTB.reset_index(drop=True)], axis = 1)
dataset.to_pickle('dataset_fits.pkl')
print('yooooo')






