#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:52:57 2023

@author: herranz
"""

import healpy       as     hp
import survey_model as     survey

from time           import time
from astropy.table  import Table

from file_names     import input_radio_source_catalogue_name
from file_names     import point_source_map_name
from file_names     import detected_catalogue_name
from file_names     import simulation_name


# %% --- DEFINITIONS

catalogue_clean_mode = 'after'  # clean the repetitions after repatching
overwrite_existing   = True     # if True, the detect_sources routine runs
                                # even if the final catalogue exists from some
                                # previous run of the code.


# %% --- GENERATE INPUT RADIO SOURCE CATALOGUES

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

    fname      = simulation_name(chan_name,sim_number)
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