#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:18:07 2022

@author: herranz
"""

import os
import survey_model      as survey
import healpy            as hp

from astropy.table       import Table

# %% --- DEFINITIONS

coadded_dir = survey.data_dir+'coadd_signal_maps/'
noise_dir   = survey.data_dir+'noise/'
radiops_dir = survey.data_dir+'foregrounds/radio_sources/'
total_dir   = survey.map_dir+'total_maps/'

LB_channels = list(survey.IMO.keys())

if not os.path.exists(total_dir):
    os.makedirs(total_dir)

# %% --- FILE NAMING ROUTINES

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
    fname : str
        The file name of the coadded (CMB+diffuse foregrounds) simulation 
        in the NERSC LiteBIRD CFS system.

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
    fname : str
        The file name of the noise simulation 
        in the NERSC LiteBIRD CFS system.

    """
    simstr = '{0:04d}'.format(sim_number)
    simdir = noise_dir+simstr+'/'
    fname  = simdir+chan_name+'_noise_FULL_'+simstr+'_PTEP_20200915_compsep.fits'
    return fname

def total_name(sim_number,chan_name):
    """
    Returns the name of the PTEP total simulation (coadded+radiops+noise)
    file for a given LiteBIRD channel and a given simulation number 
    (between 0 and 99).

    Parameters
    ----------
    sim_number : int
        The simulation number. It must take a value between 0 and 99.
    chan_name : str
        The name of the LiteBIRD channel

    Returns
    -------
    fname : str
        The file name of the total (CMB+diffuse foregrounds+noise+radiops) 
        simulation in the NERSC scratch space.

    """
    simstr = '{0:04d}'.format(sim_number)
    totdir = total_dir+simstr+'/'
    if not os.path.exists(totdir):
        os.makedirs(totdir)
    fname = totdir+chan_name+'_total_sim_'+simstr+'_PTEP_20200915_compsep.fits'
    return fname

def mock_radio_source_catalogue_name(chan_name):
    """
    Returns the name of the file containing the radio point source catalogue
    derived from the PTEP simulation of resolved sources for the LiteBIRD
    channel `chan_name`.

    Parameters
    ----------
    chan_name : str
        The name of the LiteBIRD channel

    Returns
    -------
   fname : str
       The file name of the mock radio source catalogue in the NERSC scratch space.

    """
    
    mock_dir = survey.cat_inp
    if not os.path.exists(mock_dir):
        os.makedirs(mock_dir)
    fname = mock_dir+'radiops_catalogue_'+chan_name+'_PTEP_20200915_compsep.fits'
    return fname
    
# %% --- GENERATION OF FULL SIMULATION MAPS

def PTEP_simulated_maps(sim_number,chan_name,tofile=None):
    """
    Returns a a PTEP simulation map for a given LiteBIRD channel and simulation
    number

    Parameters
    ----------
    sim_number : int
        The simulation number. It must take a value between 0 and 99.
    chan_name : str
        The name of the LiteBIRD channel
    tofile : None or str
        If not None, the total simulated map is written to the specified
        file name. If tofile == 'default', the map is written automatically
        to the default file name corresponding the sim_number and chan_name.

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
    
    if tofile is not None:
        if tofile == 'default':
            total.write(total_name(sim_number,chan_name))
        else:
            total.write(tofile)

    return {'TOTAL':total,
            'RADIOPS':radiops}

# %% --- GENERATION OF RADIO POINT SOURCE CATALOGUES

def create_mock_point_source_catalogue(chan_name):
    """
    Creates a catalogue of point sources from a point source HEALPix map and
    stores it in the appropriate NERCS scratch filesystem

    Parameters
    ----------
    chan_name : srt
        The name of the LiteBIRD channel

    Returns
    -------
 
    """
    
    fname   = radiops_name(chan_name)
    radiops = survey.load_LiteBIRD_map(fname,chan_name=chan_name)

    sources = radiops[0].data
    minmax  = hp.hotspots(sources)
    peaks   = {'Ipix':minmax[2],
               'RA':radiops.pixel_to_coordinates(minmax[2]).icrs.ra,
               'DEC':radiops.pixel_to_coordinates(minmax[2]).icrs.dec,
               'GLON':radiops.pixel_to_coordinates(minmax[2]).galactic.l,
               'GLAT':radiops.pixel_to_coordinates(minmax[2]).galactic.b,
               'I':radiops[0].data[minmax[2]],
               'Q':radiops[1].data[minmax[2]],
               'U':radiops[2].data[minmax[2]]}
    valleys = {'Ipix':minmax[1],
               'RA':radiops.pixel_to_coordinates(minmax[1]).icrs.ra,
               'DEC':radiops.pixel_to_coordinates(minmax[1]).icrs.dec,
               'GLON':radiops.pixel_to_coordinates(minmax[1]).galactic.l,
               'GLAT':radiops.pixel_to_coordinates(minmax[1]).galactic.b,
               'I':radiops[0].data[minmax[1]],
               'Q':radiops[1].data[minmax[1]],
               'U':radiops[2].data[minmax[1]]}

    peaks   = Table(peaks)
    valleys = Table(valleys)

    peaks.write(mock_radio_source_catalogue_name(chan_name),
                overwrite=True)
    
def create_point_source_catalogues():
    """
    Generates the point source catalogues in the NERCS system

    Returns
    -------
    None.

    """
    for chan in LB_channels:
        create_mock_point_source_catalogue(chan)

    