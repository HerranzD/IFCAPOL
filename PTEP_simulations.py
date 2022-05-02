#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:18:07 2022

@author: herranz
"""

import survey_model      as survey

coadded_dir = survey.data_dir+'coadd_signal_maps/'
noise_dir   = survey.data_dir+'noise/'
radiops_dir = survey.data_dir+'foregrounds/radio_sources/'

def radiops_name(chan_name):
    """
    Returns the name of the PTEP radio point source simulation file for a
    given LiteBIRD channel.

    Parameters
    ----------
    chan_name : str
        The name of the LiteBIRD channel

    Returns
    -------
    str
        The file name of the radio source simulation in the NERSC LiteBIRD CFS
        system.

    """
    return radiops_dir+'radio_sources_{0}_uKcmb_nside512.fits'.format(chan_name.split('LB_')[1])

def coadd_name(sim_number,chan_name):
    """
    Returns the name of the PTEP coadded (CMB+foregrounds) simulation file for a
    given LiteBIRD channel and a given simulation number (between 0 and 99).

    Parameters
    ----------
    sim_number : int
        The simulation number. It must take a value between 0 and 99.
    chan_name : str
        The name of the LiteBIRD channel

    Returns
    -------
    fname : TYPE
        DESCRIPTION.

    """
    simstr = '{0:04d}'.format(sim_number)
    simdir = coadded_dir+simstr+'/'
    fname  = simdir+chan_name+'_coadd_signal_map_'+simstr+'_PTEP_20200915_compsep.fits'
    return fname

def noise_name(sim_number,chan_name):
    """
    Returns the name of the PTEP noise simulation file for a
    given LiteBIRD channel and a given simulation number (between 0 and 99).

    Parameters
    ----------
    sim_number : int
        The simulation number. It must take a value between 0 and 99.
    chan_name : str
        The name of the LiteBIRD channel

    Returns
    -------
    fname : TYPE
        DESCRIPTION.

    """
    simstr = '{0:04d}'.format(sim_number)
    simdir = noise_dir+simstr+'/'
    fname  = simdir+chan_name+'_noise_FULL_'+simstr+'_PTEP_20200915_compsep.fits'
    return fname

def PTEP_simulated_maps(sim_number,chan_name):
    """
    Returns a a PTEP simulation map for a given LiteBIRD channel and simulation
    number

    Parameters
    ----------
    sim_number : int
        The simulation number. It must take a value between 0 and 99.
    chan_name : str
        The name of the LiteBIRD channel

    Returns
    -------
    dict
        A dictionary containing:
            - 'TOTAL': the total simulated map (CMB+foregrounds+point sources+noise)
            - 'RADIOPS': the radio sources map
        Both maps are in micro Kelvin units.

    """

    signal  = survey.load_LiteBIRD_map(coadd_name(sim_number,chan_name),
                                       chan_name=chan_name)
    noise   = survey.load_LiteBIRD_map(noise_name(sim_number,chan_name),
                                       chan_name=chan_name)
    radiops = survey.load_LiteBIRD_map(radiops_name(chan_name),
                                       chan_name=chan_name)

    total   = signal+noise+radiops

    return {'TOTAL':total,
            'RADIOPS':radiops}