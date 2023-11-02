#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:52:57 2023

@author: herranz
"""

import os
from time import time
import survey_model as survey


# %% --- DEFINITIONS

catalogue_clean_mode = 'after'  # clean the repetitions after repatching
overwrite_existing   = True     # if True, the detect_sources routine runs
                                # even if the final catalogue exists from some
                                # previous run of the code.

# %% --- FILE NAMING ROUTINES

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


def detected_catalogue_name(sim_number,chan_name,snrcut=3.5):
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
    for a given Run0 post-PTEP simulation, after cleaning possible repetitions arising
    from overlapping sky patches.

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