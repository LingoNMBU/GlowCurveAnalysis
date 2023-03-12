# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn import metrics

from AdaGlowFit import AdaGlowFit
from lsGlowFit import LsGlowFit
from datahandling import DataHandling

# Load experimental data
LTB_paths = []
CaSO4_paths = []

directory = r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\data_aarhus'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking substance
    if filename.split('_')[0] == 'LTB':
        LTB_paths.append(f)
    else:
        CaSO4_paths.append(f)


LTB = ['1a_3', '1a_6', '1b_2']

CaSO4_paths2 = [r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\data_aarhus\CaSO4_1C_4.xlsx']

LTB_paths2 = [r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\data_aarhus\LTB_1a+3a_3.xlsx']

def preprocess_data(path):
    """
    Preprocesses the datafile
    :param path: path of glowcurve file
    :return: preprocessed df of data
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
    exp_data = plot_data.replace(np.nan, 0)

    # Remove cooldown from data
    exp_data.index = exp_data['Temperature measured']
    exp_data = exp_data.loc[:exp_data['Temperature measured'].idxmax(), :]
    exp_data = exp_data.drop(axis=1, columns=['Time', 'Temperature setpoint'])
    exp_data.columns = ['Intensity', 'Temperature']

    #Change to Kelvin
    exp_data.Temperature = exp_data.Temperature + 273.15

    return exp_data


def fit_glow_curve(glowcurve, n_peaks, order, name):
    # Curve fitting

    exp_data = glowcurve

    lsGlow1 = LsGlowFit(data_df=glowcurve, n_peaks=n_peaks, beta=5)
    print(f'fitting lm for gc with {n_peaks} peaks of {order} order')
    if order == 'first':
        lsGlow1.fit_lm_1o()
    elif order == 'second':
        lsGlow1.fit_lm_2o()
    elif order == 'general':
        lsGlow1.fit_lm()
    else:
        lsGlow1.fit_lm_2o()

    result = lsGlow1.result
    mod_data_lm = result.best_fit

    print('lm params')
    print(result.best_values)

    print(result.fit_report())

    # Plot initialization
    fig = plt.figure()
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Curve plot
    ax1.plot(exp_data['Temperature'], glowcurve['Intensity'],
             label=f'{name}',
             linestyle='solid',
             color='black')
    ax1.plot(exp_data['Temperature'], mod_data_lm,
             label='best fit lm',
             linestyle='dashdot',
             color='salmon')

    # Peak plots
    for i in range(n_peaks):
        ax1.plot(exp_data.Temperature, lsGlow1.peak_fits[i],
                 label=f'model peak {i + 1}',
                 linestyle='dashed')

    # Res plot
    res = result.residual
    ax2.scatter(exp_data['Temperature'], res, s=0.5, color='salmon')
    ax2.plot(exp_data['Temperature'], [0 for _ in exp_data['Temperature']],
             linestyle='dashed',
             color='black')

    # Garnityr
    ax1.set_title(f'{order} order curve fit, peaks : {n_peaks}')
    ax1.set_xlabel('Temperature [K]')
    ax1.set_ylabel('Intensity [Counts]')

    ax2.set_xlabel('Temperature [K]')
    ax2.set_ylabel('Residue [Counts]')

    ax1.legend()
    plt.tight_layout()
    plt.savefig(f'{name}_o_{order}_{n_peaks}peaks_plot.pdf')
    plt.show()

    rmse = metrics.mean_squared_error(y_true=glowcurve['Intensity'],
                                      y_pred=mod_data_lm,
                                      squared=False)
    print(rmse)

    # Storing fit ind info
    fit_df = pd.DataFrame(columns=['Temperature', 'best_fit', 'residue'])
    fit_df['Temperature'] = exp_data['Temperature']
    fit_df['best_fit'] = result.best_fit
    fit_df['residue'] = result.residual
    for i in range(n_peaks):
        fit_df[f'peak_fit_{i + 1}'] = lsGlow1.peak_fits[i]

    info_df = pd.DataFrame(result.best_values, index=range(1))
    info_df['rmse'] = rmse
    info_df['order'] = order
    info_df['r_squared'] = result.rsquared
    info_df['n_peaks'] = n_peaks

    # Saving dataframes
    fit_df.to_pickle(f'{name}_o_{order}_{n_peaks}peaks_fit.pkl')
    info_df.to_pickle(f'{name}_o_{order}_{n_peaks}peaks_info.pkl')

paths = LTB_paths2
# Select curve
for path in paths:
    name = path.split('\\')[-1].split('.')[0]
    data_pp = preprocess_data(path)
    for order in ['second']:
        fit_glow_curve(glowcurve=data_pp,
                       order=order,
                       n_peaks=3,
                       name=name)
