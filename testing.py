#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:45:31 2022

@author: herranz
"""

import numpy        as np
import survey_model as survey
import healpy       as hp
import os

from astropy.table  import Table
from myutils        import sigma2fwhm

coadded_file = survey.data_dir+'LB_LFT_40_coadd_signal_map_0000_PTEP_20200915_compsep.fits'
noise_file   = survey.data_dir+'LB_LFT_40_noise_FULL_0000_PTEP_20200915_compsep.fits'
ps_file      = survey.data_dir+'radio_sources_LFT_40_uKcmb_nside512.fits'

chan_name    = 'LB_LFT_40'

test_plotf   = '/Users/herranz/Desktop/testplot.jpg'

# %% --- COMPONENT MAPS

signal  = survey.load_LiteBIRD_map(coadded_file,chan_name=chan_name)
noise   = survey.load_LiteBIRD_map(noise_file,chan_name=chan_name)
radiops = survey.load_LiteBIRD_map(ps_file,chan_name=chan_name)

# %% --- TOTAL MAP

totmap_file = survey.data_dir+'LB_LFT_40_testing_map_0000_PTEP_20200915_compsep.fits'

if os.path.isfile(totmap_file):
    total = survey.load_LiteBIRD_map(totmap_file,chan_name=chan_name)
else:
    total   = signal+noise+radiops
    total.write(totmap_file,overwrite=True)

# %% --- POINT SOURCE MOCK CATALOGUE FOR TESTING

mock_catalogue_fname = survey.data_dir+'mock_ps_catalogue_LFT_40_uKcmb_nside512.fits'

def create_mock_point_source_catalogue():

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

    peaks.write(mock_catalogue_fname,
                overwrite=True)

    return peaks,valleys

if os.path.isfile(mock_catalogue_fname):
    peaks = Table.read(mock_catalogue_fname)
else:
    peaks,valleys = create_mock_point_source_catalogue()

peaks.sort(keys='I',reverse=True)
epeaks = peaks[np.abs(peaks['GLAT'])>20]

def test_fwhms(ntest=100):

    f = []
    for i in range(ntest):
        c     = radiops.pixel_to_coordinates(epeaks['Ipix'][i])
        patch = radiops[0].patch(c)
        g     = patch.central_gaussfit(return_output=True,verbose=False)
        f.append(g.sigma*sigma2fwhm)
    f = np.array(f)*patch.pixsize

    return f
