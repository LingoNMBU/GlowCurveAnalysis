# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, signal
import os
import re


class DataHandling:
    """
    Class handling, data imports, filtering, creation of datasets and so forth.
    """

    def __init__(self, folder_path):

        self.directories = folder_path

        self.data_dict = self.import_data(self.directories)

        self.dataset = None

        self.depth_dict = {'2a': 2.25,
                           '1a+3a': 10.25,
                           '1b': 15.25,
                           '1c': 15.45,
                           '1C': 15.45}

    def import_data(self, directories):
        """
        Imports and formats data from a list of directories
        :param directories: list of directories
        :return:
        """
        data_dict = {}
        for directory in directories:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                if os.path.isfile(path):
                    data_dict[filename] = self.format_data(path)
        return data_dict

    def format_data(self, path):
        """
        formats the data from the excel doc into a dataframe containing counts and temps,
         and another dataframe containing the info about the sample
        :param path: path of the sample file
        :return:
        """
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

        #Remove cooldown from data
        plot_data.index = plot_data['Temperature measured']
        exp_data = plot_data.loc[:plot_data['Temperature measured'].idxmax(), :]

        return [pd.DataFrame(plot_data), info_data]

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
                        'peak_temperature',
                        'peak_time',
                        'peak_intensity',
                        'peak_width_half',
                        'sum_intensity',
                        'depth',
                        'peak_intensity_1diff',
                        'minima_intensity_1diff',
                        'peak_temperature_1diff',
                        'minima_temperature_1diff',
                        'sum_intensity_1diff',
                        'skew',
                        'kurtosis']
        dataset = pd.DataFrame(columns=column_names)

        data_dict = self.data_dict

        for filename in data_dict.keys():
            name_split = filename.split('_')
            data = data_dict[filename][0].copy(deep=True)

            # Test features
            substance = name_split[0]
            position = name_split[1]
            nr = name_split[2].split('.')[0]
            depth = self.depth_dict[position]

            # Peak features
            data.index = data['Temperature measured']
            peak_temp = data['Counts measured'].idxmax()
            data.index = data['Time']
            peak_time = data['Counts measured'].idxmax()
            peak_intensity = data['Counts measured'].max()
            data.index = range(len(data.Time))
            peak_ind = [data['Counts measured'].idxmax()]
            peak_width_half = signal.peak_widths(data['Counts measured'].values, peaks=peak_ind,
                                                 rel_height=0.5)[0][0]

            # Stats features
            intensitysum = data['Counts measured'].sum()
            skew = stats.skew(data['Counts measured'].values)
            kurtosis = stats.kurtosis(data['Counts measured'].values)


            # Derivative features
            data_diff1 = np.diff(data['Counts measured'], n=1)
            peak_intensity_1diff = np.max(data_diff1)
            peak_temp_1diff = np.argmax(data_diff1)
            minima_intensity_1diff = np.min(data_diff1)
            minima_temp_1diff = np.min(data_diff1)
            sum_abs_intensity_diff1 = np.sum(data_diff1)

            sample_features = np.array([substance,
                                        nr,
                                        position,
                                        peak_temp,
                                        peak_time,
                                        peak_intensity,
                                        peak_width_half,
                                        intensitysum,
                                        depth,
                                        peak_intensity_1diff,
                                        minima_intensity_1diff,
                                        peak_temp_1diff,
                                        minima_temp_1diff,
                                        sum_abs_intensity_diff1,
                                        skew,
                                        kurtosis
                                        ]).reshape(1, -1)

            sample_df = pd.DataFrame(sample_features, columns=column_names)

            dataset = pd.concat([dataset, sample_df], ignore_index=True)

            self.dataset = dataset

    def filter_data(self, data_dict, substances=['CaSO4'], P='P1', dose='all'):
        """
        Filters data for different parameters
        :param data_dict: dictionary of datasamples created by format data
        :param substances: what substances to filter for
        :param P: what position to filter for
        :param dose: what dose to filter for
        :return: plot data, data ready for plotting, filterd
        """
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
