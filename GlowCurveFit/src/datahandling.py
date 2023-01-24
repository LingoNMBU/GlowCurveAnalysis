# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re


class DataHandling:

    def __init__(self, folder_path):

        self.directories = folder_path

        self.data_dict = self.import_data(self.directories)

        self.dataset = None

        self.depth_dict = {'2a': 2.25,
                           '1a+3a': 10.25,
                           '1b': 15.25,
                           '1c': 15.45,
                           '1C': 15.45}

    def format_data(self, path):
        df = pd.read_excel(path)

        info_data = df.T.iloc[:28, 0]
        plot_data = df.T.iloc[30:, :]
        plot_data.columns = [df.T.iloc[28, 0],
                             df.T.iloc[28, 1],
                             df.T.iloc[28, 2]]
        plot_data['Time'] = plot_data.index
        plot_data.reset_index()

        # replace nan with 0
        plot_data = plot_data.replace(np.nan, 0)

        return [pd.DataFrame(plot_data), info_data]

    def import_data(self, directories):
        data_dict = {}
        for directory in directories:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                if os.path.isfile(path):
                    data_dict[filename] = self.format_data(path)
        return data_dict

    def make_dataset(self):
        """
        processing specifically for aarhus data.
         Would need some minor change for filenames of blindern data
        :return:
        """
        # Dataset initialization
        column_names = ['substance',
                        'sample_nr',
                        'position',
                        'peak temperature',
                        'peak time',
                        'peak intensity',
                        'sum intensity',
                        'depth']
        dataset = pd.DataFrame(columns=column_names)

        data_dict = self.data_dict

        for filename in data_dict.keys():
            name_split = filename.split('_')
            data = data_dict[filename][0].copy(deep=True)

            # Test features
            substance = name_split[0]
            position = name_split[1]
            nr = name_split[2].split('.')[0]

            # Peak features
            data.index = data['Temperature measured']
            peak_temp = data['Counts measured'].idxmax()
            data.index = data['Time']
            peak_time = data['Counts measured'].idxmax()
            peak_intensity = data['Counts measured'].max()
            intensitysum = data['Counts measured'].sum()
            depth = self.depth_dict[position]

            sample_features = np.array([substance,
                                        nr,
                                        position,
                                        peak_temp,
                                        peak_time,
                                        peak_intensity,
                                        intensitysum,
                                        depth]).reshape(1, -1)

            sample_df = pd.DataFrame(sample_features, columns=column_names)

            dataset = pd.concat([dataset, sample_df], ignore_index=True)

            self.dataset = dataset

    def filter_data(self, data_dict, substances=['CaSO4'], P='P1', dose='all'):
        plot_data = {}
        for filename in data_dict.keys():
            for substance in substances:
                # substance filter
                if re.search(fr'{substance}', filename):
                    data = data_dict[filename][0].copy(deep=True)

                    # P filter
                    if P == 'all':
                        # dose filter1
                        if dose == 'all':
                            plot_data[filename] = data
                        elif re.search(fr' {dose} Gy', filename):
                            plot_data[filename] = data
                    elif re.search(fr'{P}', filename, re.IGNORECASE):
                        # dose filter2
                        if dose == 'all':
                            plot_data[filename] = data
                        elif re.search(fr'{dose}', filename):
                            plot_data[filename] = data
        return plot_data
