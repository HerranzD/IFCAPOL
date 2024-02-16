#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 21:20:55 2024

@author: herranz
"""

import os
import survey_model as     survey


# %% --- FILE NAMING ROUTINES

def input_radio_source_catalogue_name(chan_name):
    """
    Returns the name of the input radio source catalogue for the given
    channel.

    Parameters
    ----------
    chan_name : str
        Channel name.

    Returns
    -------
    fname : str
        File name.

    """

    return survey.cat_inp+'radio_sources_catalogue_{0}.fits'.format(chan_name)

def point_source_map_name(chan_name):
    """
    Returns the name of the radio point source map for the given channel.

    Parameters
    ----------
    chan_name : str
        Channel name.

    Returns
    -------
    fname : str
        File name.

    """

    ps_dir = survey.data_dir+'foregrounds/radio_sources/'

    source_id = ['LFT_40',
                 'LFT_50',
                 'LFT_60',
                 'LFT_68a',
                 'LFT_68b',
                 'LFT_78a',
                 'LFT_78b',
                 'LFT_89a',
                 'LFT_89b',
                 'MFT_100',
                 'LFT_100',
                 'MFT_119',
                 'LFT_119',
                 'MFT_140',
                 'LFT_140',
                 'MFT_166',
                 'HFT_195',
                 'MFT_195',
                 'HFT_235',
                 'HFT_280',
                 'HFT_337',
                 'HFT_402']

    band_str = source_id[survey.LB_channels.index(chan_name)]

    fname = ps_dir+'radio_sources_{0}_uKCMB_nside512.fits'.format(band_str)

    return fname


def simulation_name(chan_name,sim_number):
    """
    Returns the name of the file containing the run0 simulation maps for the
    given channel and simulation number.

    Parameters
    ----------
    chan_name : str
        Channel name.
    sim_number : int
        Simulation number.

    Returns
    -------
    fname : str
        File name.

    """


    str_sim = '{0:04d}'.format(sim_number)

    fname  = survey.data_dir
    fname += '{0}FT/'.format(chan_name[0])
    fname += '{0}/'.format(chan_name)
    fname += '{0}_maps/'.format(survey.sim_type)
    fname += 'LB_{0}FT_'.format(chan_name[0])
    fname += chan_name
    fname += '_{0}_'.format(survey.sim_type)
    fname += 'cmb_fg_wn_1f_{0}_{1}.fits'.format(survey.fknee,str_sim)

    return fname

def foregrounds_name(chan_name):
    """
    Returns the name of the file containing the run0 foreground maps for the
    given channel.

    Parameters
    ----------
    chan_name : str
        Channel name.

    Returns
    -------
    fname : str
        File name.

    """

    fdir = survey.data_dir+'foregrounds/'
    fname = fdir+'LB_{0}FT_{1}_fg.fits'.format(chan_name[0],chan_name)

    return fname


def detected_catalogue_name(sim_number,chan_name_inp,snrcut=3.5):
    """
    Returns the name of the blind catalogue of detected source candidates
    for a given Run0 post-PTEP simulation.

    Parameters
    ----------
    sim_number : int
        The simulation number. It must take a value between 0 and 199.
    chan_name : str
        The name of the LiteBIRD channel
    snrcut : float
        The limit signal-to-noise ratio of the catalogue


    Returns
    -------
    fname : str
        The file name of the blind catalogue.

    """

    from path_defs import cat_out

    if isinstance(chan_name_inp,str):
        chan_name = chan_name_inp
    elif isinstance(chan_name_inp,int):
        chan_name = survey.LB_channels[chan_name_inp]
    else:
        print(' Warning: wrong chan_name format in detected_catalogue_name')

    simstr         = '{0:04d}'.format(sim_number)
    output_catdir  = cat_out
    output_catdir += 'Sim_{0}/'.format(simstr)

    if not os.path.exists(output_catdir):
        os.makedirs(output_catdir)

    fname  = output_catdir
    fname += 'IFCAPOL_catalogue_postPTEP_Run0_'+chan_name+'_'
    fname += simstr+'SNR_{0}.fits'.format(snrcut)

    return fname

def cleaned_catalogue_name(sim_number,chan_name,
                           snrcut     = 3.5,
                           clean_mode = 'after'):
    """
    Returns the name of the blind catalogue of detected source candidates
    for a given Run0 post-PTEP simulation, after cleaning possible
    repetitions arising from overlapping sky patches.

    Parameters
    ----------
    sim_number : int
        The simulation number. It must take a value between 0 and 199.
    chan_name : str
        The name of the LiteBIRD channel
    snrcut : float
        The limit signal-to-noise ratio of the catalogue

    Returns
    -------
    fname_out : str
        The file name of the blind catalogue.

    """

    fname     = detected_catalogue_name(sim_number,chan_name,snrcut)
    fname_out = fname.replace('.fits',
                              '_{0}_cleaned.fits'.format(clean_mode))

    return fname_out



