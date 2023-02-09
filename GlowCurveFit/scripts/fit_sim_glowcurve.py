# !python
# -*- coding: utf-8 -*

__author__ = 'Erling Ween Eriksen'
__email__ = 'erlinge@nmbu.no'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn import metrics

from AdaGlowFit import AdaGlowFit
from lsGlowFit import LsGlowFit

# Load experimental data
path1 = r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\LTB_P1_processed.csv'
path2 = r'C:\Users\erlin\Desktop\Studie\2023\Master\GlowCurveAnalysis\data\CaSO4_P1_processed.csv'

paths = [path1, path2]



def preprocess_data(path):
    exp_data = pd.read_csv(path)

    exp_data.index = exp_data['Temperature measured']
    exp_data = exp_data.loc[:exp_data['Temperature measured'].idxmax(), :]
    exp_data = exp_data.drop(axis=1, columns=['Time', 'Temperature setpoint'])
    exp_data.columns = ['Intensity', 'Temperature']
    exp_data.Temperature = exp_data.Temperature + 273.15

    return exp_data

path = path1
name = path.split('\\')[-1].split('.')[0]
exp_data = preprocess_data(path)


# Generate sim curves
n_peaks = 1
adaGlow0 = AdaGlowFit(exp_data, beta=5, metric='rmse', g_max=100, N_pop=50, n_peaks=n_peaks)
lsGlow1 = LsGlowFit(exp_data, n_peaks=n_peaks, beta=5)

gen_params1 = {'E': 1.7, 'b': 3.0, 'n0': 800000, 'Sdd': 10 ** 19}
gen_params2 = {'E': 1.5, 'b': 2.1, 'n0': 4000000, 'Sdd': 10 ** 19}
gen_sim_curve1 = adaGlow0.make_fit(gen_params1, exp_data)
gen_sim_curve2 = adaGlow0.make_fit(gen_params2, exp_data)
gen_sim_curve2['Intensity'] = gen_sim_curve1['Intensity'] + gen_sim_curve2['Intensity']

o1_params1 = {'kb': 8.617333e-05, 'Tm1': 550, 'Im1': 200000, 'E1': 0.6}
o1_params2 = {'kb': 8.617333e-05, 'Tm1': 540, 'Im1': 450000, 'E1': 1.3}
o1_sim_curve1 = lsGlow1.make_sim_glowcurve(o1_params1, order='first')
o1_sim_curve2 = lsGlow1.make_sim_glowcurve(o1_params2, order='first')
o1_sim_curve3 = lsGlow1.make_sim_glowcurve(o1_params2, order='first')
o1_sim_curve3['Intensity'] = o1_sim_curve1['Intensity'].values + \
                             o1_sim_curve2['Intensity'].values




def fit_glow_curve(glowcurve, n_peaks, order):
    # Curve fitting
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
                 label=f'model peak {i+1}',
                 linestyle='dashed')

    # Res plot
    res = result.residual
    ax2.scatter(exp_data['Temperature'], res, s=0.5, color='salmon')
    ax2.plot(exp_data['Temperature'], [0 for _ in exp_data['Temperature']],
             linestyle='dashed',
             color='black')

    # Garnityr
    ax1.set_title(f'First order curve fit, peaks : {n_peaks}')
    ax1.set_xlabel('Temperature [K]')
    ax1.set_ylabel('Intensity [Counts]')

    ax2.set_xlabel('Temperature [K]')
    ax2.set_ylabel('Residue [Counts]')

    ax1.legend()
    plt.show()

    rmse = metrics.mean_squared_error(y_true=exp_data['Intensity'],
                                       y_pred=mod_data_lm,
                                       squared=False)
    print(rmse)

    # Storing fir ind info
    fit_df = pd.DataFrame(columns=['Temperature', 'best_fit', 'residue'])
    fit_df['Temperature'] = exp_data['Temperature']
    fit_df['best_fit'] = result.best_fit
    fit_df['residue'] = result.residual
    for i in range(n_peaks):
        fit_df[f'peak_fit_{i+1}'] = lsGlow1.peak_fits[i]

    info_df = pd.DataFrame(result.best_values, index=range(1))
    info_df['rmse'] = rmse
    info_df['order'] = order
    info_df['r_squared'] = result.rsquared
    info_df['n_peaks'] = n_peaks

    # Saving dataframes
    fit_df.to_pickle(f'{name}_o_{order}_{n_peaks}peaks_fit.pkl')
    info_df.to_pickle(f'{name}_o_{order}_{n_peaks}peaks_info.pkl')

# Select curve
for path in paths:
    exp_data = pd.read_csv(path)

    sample = path.split('\\')[-1].split('.')[0]

    exp_data.index = exp_data['Temperature measured']
    exp_data = exp_data.loc[:exp_data['Temperature measured'].idxmax(), :]
    exp_data = exp_data.drop(axis=1, columns=['Time', 'Temperature setpoint'])
    exp_data.columns = ['Intensity', 'Temperature']
    exp_data.Temperature = exp_data.Temperature + 273.15

    glowcurve = exp_data
    n_peaks = 5
    order = 'first'