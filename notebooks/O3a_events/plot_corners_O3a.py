###### works on CIT(needs posterior samples) ####

import numpy as np
import os
import sys
import glob
import json
import h5py
import argparse
import seaborn as sns
from pesummary.gw.file.read import read
import matplotlib.pylab as plt

import pandas as pd


base_dir = '/home/srashti.goyal/strong-lensing-ml/'  # CIT

O3a_events_df = pd.read_csv(
    base_dir +
    'data/O3a_events/O3a_blu.txt',
    delimiter='\t')


# In[3]:


def read_pe_samples(fname, event_name):
    data = read(fname)
    if event_name == 'GW190412' or event_name == 'GW190814':
        posterior_samples = data.samples_dict['C01:IMRPhenomPv3HM']
    elif event_name == 'GW190521':
        posterior_samples = data.samples_dict['C01:NRSur7dq4']
    else:
        posterior_samples = data.samples_dict['C01:IMRPhenomPv2']
    return posterior_samples


# In[4]:


data_dir = '/home/pe.o3/o3a_catalog/data_release/all_posterior_samples/'


def make_sns_corner_plots_new_params(
        psamples1,
        psamples2,
        fname='test.png',
        inj_pars1=None,
        inj_pars2=None,
        kde=False):
    """Plot and save superposed KDE corner plots for the two images from posteriors."""
    samples1 = pd.DataFrame()
    samples2 = pd.DataFrame()
    samples1['m1'], samples1['m2'] = psamples1['mass_1'], psamples1['mass_2']
    samples1['dec'] = np.sin(psamples1['ra'])

    samples2['m1'], samples2['m2'] = psamples2['mass_1'], psamples2['mass_2']
    samples2['dec'] = np.sin(psamples2['dec'])

    keys = [
        'ra',
        'theta_jn',
        'psi',
        'phase',
        'geocent_time',
        'luminosity_distance']
    for key in keys:
        samples1[key], samples2[key] = psamples1[key], psamples2[key]
    samples1['img'] = np.zeros(len(samples1['m1'])).astype(str)
    samples2['img'] = np.ones(len(samples2['m1'])).astype(str)
    posteriors = pd.concat([samples1, samples2])
    sns.set(font_scale=1.5)
    posteriors = posteriors.rename(
        columns={
            "m1": "$m_1$",
            "m2": "$m_2$",
            'dec': 'sin($\\delta$)',
            'ra': '$\\alpha$',
            'theta_jn': '$\\iota$',
            'psi': '$\\psi$',
            'phase': '$\\phi_0$',
            'geocent_time': '$t_c$',
            'luminosity_distance': '$d_L$'})
    fig1 = sns.PairGrid(
        data=posteriors,
        vars=[
            '$m_1$',
            '$m_2$',
            '$\\alpha$',
            'sin($\\delta$)',
            '$\\iota$',
            '$\\psi$',
            '$\\phi_0$',
            '$d_L$',
            '$t_c$'],
        height=1.5,
        corner=True,
        hue='img')
    if kde:
        fig1 = sns.PairGrid(
            data=posteriors,
            vars=[
                '$m_1$',
                '$m_2$',
                '$\\alpha$',
                'sin($\\delta$)',
                '$\\iota$',
                '$\\psi$',
                '$\\phi_0$',
                '$d_L$',
                '$t_c$'],
            height=1.5,
            hue='img')
        fig1 = fig1.map_lower(sns.kdeplot)  # ,shade=True,shade_lowest=False)

    fig1 = fig1.map_lower(sns.histplot, pthresh=0.1)

    fig1 = fig1.map_diag(plt.hist, histtype='step', lw=2)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    ndim = 9
    axes = fig1.axes

    for inj_pars in [inj_pars1, inj_pars2]:
        if inj_pars is not None:
            values = np.array([inj_pars['mass_1'],
                               inj_pars['mass_2'],
                               inj_pars['ra'],
                               np.sin(inj_pars['dec']),
                               inj_pars['theta_jn'],
                               inj_pars['psi'],
                               inj_pars['phase'],
                               inj_pars['luminosity_distance'],
                               inj_pars['geocent_time']])

            for i in range(ndim):
                ax = axes[i, i]
                ax.axvline(values[i], color="gray")
            for yi in range(ndim):
                for xi in range(yi):
                    ax = axes[yi, xi]
                    ax.axvline(values[xi], color="gray")
                    ax.axhline(values[yi], color="gray")

    plt.savefig(fname)
    plt.close()
    print('Done: ' + fname)


odir = 'corner_plots/'


# In[47]:


def plot_corners_events(i):
    posterior_file = data_dir + O3a_events_df['event1'][i] + '_comoving.h5'
    posterior_samples1 = read_pe_samples(
        posterior_file, O3a_events_df['event1'][i][:8])
    posterior_file = data_dir + O3a_events_df['event2'][i] + '_comoving.h5'
    posterior_samples2 = read_pe_samples(
        posterior_file, O3a_events_df['event2'][i][:8])
    fname = odir + O3a_events_df['event1'][i] + \
        '-' + O3a_events_df['event2'][i] + '.png'
    make_sns_corner_plots_new_params(
        posterior_samples1,
        posterior_samples2,
        fname=fname)


# In[ ]:

for i in range(5, len(O3a_events_df.event1.values)):
    print(i)
    try:
        plot_corners_events(i)
    except BaseException:
        None
