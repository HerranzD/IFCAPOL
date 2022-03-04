#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:45:31 2022

@author: herranz
"""

import survey_model as survey 
from astropy.table import Table
import healpy as hp

coadded_file = survey.data_dir+'LB_LFT_40_coadd_signal_map_0000_PTEP_20200915_compsep.fits'
noise_file   = survey.data_dir+'LB_LFT_40_noise_FULL_0000_PTEP_20200915_compsep.fits'
ps_file      = survey.data_dir+'radio_sources_LFT_40_uKcmb_nside512.fits'

chan_name    = 'LB_LFT_40'

signal  = survey.load_LiteBIRD_map(coadded_file,chan_name=chan_name)
noise   = survey.load_LiteBIRD_map(noise_file,chan_name=chan_name)
radiops = survey.load_LiteBIRD_map(ps_file,chan_name=chan_name)

total   = signal+noise+radiops


def create_mock_point_source_catalogue():
    
    sources = radiops[0].data
    minmax  = hp.hotspots(sources)
    picos   = {'Ipix':minmax[2],
               'RA':radiops.pixel_to_coordinates(minmax[2]).icrs.ra,
               'DEC':radiops.pixel_to_coordinates(minmax[2]).icrs.dec,
               'GLON':radiops.pixel_to_coordinates(minmax[2]).galactic.l,
               'GLAT':radiops.pixel_to_coordinates(minmax[2]).galactic.b,
               'I':radiops[0].data[minmax[2]],
               'Q':radiops[1].data[minmax[2]],
               'U':radiops[2].data[minmax[2]]}
    valles  = {'Ipix':minmax[1],
               'RA':radiops.pixel_to_coordinates(minmax[1]).icrs.ra,
               'DEC':radiops.pixel_to_coordinates(minmax[1]).icrs.dec,
               'GLON':radiops.pixel_to_coordinates(minmax[1]).galactic.l,
               'GLAT':radiops.pixel_to_coordinates(minmax[1]).galactic.b,
               'I':radiops[0].data[minmax[1]],
               'Q':radiops[1].data[minmax[1]],
               'U':radiops[2].data[minmax[1]]}
    
    picos  = Table(picos)
    valles = Table(valles)
    
    picos.write(survey.data_dir+'mock_ps_catalogue_LFT_40_uKcmb_nside512.fits',
                overwrite=True)
    
    return picos,valles
    