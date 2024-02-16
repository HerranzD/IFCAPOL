#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:52:57 2023

@author: herranz
"""

import os
import survey_model as     survey
import healpy       as     hp

from time           import time
from astropy.table  import Table



# %% --- DEFINITIONS

catalogue_clean_mode = 'after'  # clean the repetitions after repatching
overwrite_existing   = True     # if True, the detect_sources routine runs
                                # even if the final catalogue exists from some
                                # previous run of the code.

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


def run0_name(chan_name,sim_number):
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

def run0_foregrounds_name(chan_name):
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

def cleaned_catalogue_name(sim_number,chan_name,snrcut=3.5):
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
                              '_{0}_cleaned.fits'.format(catalogue_clean_mode))

    return fname_out



# %% --- INPUT RADIO SOURCE CATALOGUES

def create_INPUT_point_source_catalogue(chan_name):
    """
    Creates a catalogue of point sources from a point source HEALPix map and
    stores it in the appropriate output folder

    Parameters
    ----------
    chan_name : srt
        The name of the LiteBIRD channel

    Returns
    -------
    peaks : astropy.table.Table
        The table of detected peaks
    valleys : astropy.table.Table
        The table of detected valleys

    """

    fname   = point_source_map_name(chan_name)
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

    peaks.write(input_radio_source_catalogue_name(chan_name),
                overwrite=True)

    return peaks,valleys


# %% --- SOURCE DETECTION PIPELINE

def detect_source_run0(sim_number,chan_name,snrcut=3.5,count_time=False):
    """
    Runs the IFCAPOL source detection algorithm on an individual LiteBIRD
    Run0 post-PTEP simulation in Cori at NERSC. The simulation is defined by a
    LiteBIRD detector name and a noise+CMB simulation number (from 0 to 199).

    The detection is done in three stages:
        - First a blind search is performed, using a flat sky patching with
          many overlapping areas.
        - Then, in a second run, IFCAPOL focuses on each one of the targets
          selected during the previous step. A new flat patch is projected
          around the target in order to avoid border effects.
        - Finally, the resulting catalogue is cleaned by removing repeated
          sources (appearing in the overlapping regions between flat patches)
          in descending signal-to-noise ratio. The ouput catalogues
          (both the full target set of blind detections and the catalogue
           cleaned of repetitios) are writen to file.

    Parameters
    ----------
    sim_number : int
        Simulation number.
    chan_name : str
        LiteBIRD channel name.
    snrcut : float, optional
        Signal-to-noise ratio threshold for the source detection. The default
        is 3.5.
    count_time: bool
        If True, counts the running time of catalogue creation and writes it
        to screen

    Returns
    -------
    dict
        Dictionary with the following keys:
            'overlapping':
                the blind catalogue of detected sources
            'cleaned':
                the non-blind catalogue of detected sources,
                after cleaning repetitions.

    """

    import astropy.units     as     u
    from astropy.coordinates import SkyCoord

    import IFCAPOL           as     pol
    from survey_model        import load_LiteBIRD_map
    from IFCAPOL_catalogue   import blind_survey,non_blind_survey

    # Loads the run0 simulation for the chanel CHAN_NAME
    #   and simulation number SIM_NUMBER

    fname      = run0_name(chan_name,sim_number)
    simulation = load_LiteBIRD_map(fname,chan_name=chan_name)

    # Output catalogue names

    catal_fname = detected_catalogue_name(sim_number,
                                          chan_name,
                                          snrcut)

    # Generates a mock source patch at a fiducial coordinate. This is
    #   for determining the effective FWHM (map plus pixel widths,
    #   added in quadrature) in an internally consistent way

    s           = pol.Source.from_coordinate(simulation,
                                             SkyCoord(0,0,frame='icrs',
                                                      unit=u.deg))
    fwhm        = s.fwhm

    # Blind source detection

    start_time  = time()

    blind       = blind_survey(simulation[0],
                               fwhm,
                               catal_fname,
                               verbose=False)

    end_time    = time()

    if count_time:
        print('Blind detection time: {0:.2f} seconds'.format(end_time-start_time))

    # Non-blind source detection

    start_timenb  = time()

    nonblind    = non_blind_survey(simulation,
                                   catal_fname,
                                   clean_mode = catalogue_clean_mode,
                                   snrcut     = snrcut,
                                   verbose    = False)

    end_timenb    = time()

    if count_time:
        print('Non-blind detection time: {0:.2f} seconds'.format(end_timenb-start_timenb))
        print('Total detection time: {0:.2f} seconds'.format(end_timenb-start_time))

    # Returns

    return {'overlapping':blind,
            'cleaned':nonblind}