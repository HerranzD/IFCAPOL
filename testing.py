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

chan_name    = 'LB_LFT_40'
#chan_name    = 'LB_HFT_337'

coadded_file = survey.data_dir+chan_name+'_coadd_signal_map_0000_PTEP_20200915_compsep.fits'
noise_file   = survey.data_dir+chan_name+'_noise_FULL_0000_PTEP_20200915_compsep.fits'
ps_file      = survey.data_dir+'radio_sources_'+chan_name[3:]+'_uKcmb_nside512.fits'



test_plotf   = '/Users/herranz/Desktop/testplot.jpg'

testdir      = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/Tests/'

# %% --- COMPONENT MAPS

signal  = survey.load_LiteBIRD_map(coadded_file,chan_name=chan_name)
noise   = survey.load_LiteBIRD_map(noise_file,chan_name=chan_name)
radiops = survey.load_LiteBIRD_map(ps_file,chan_name=chan_name)

# %% --- TOTAL MAP

totmap_file = survey.data_dir+chan_name+'_testing_map_0000_PTEP_20200915_compsep.fits'

if os.path.isfile(totmap_file):
    total = survey.load_LiteBIRD_map(totmap_file,chan_name=chan_name)
else:
    total   = signal+noise+radiops
    total.write(totmap_file)

# %% --- POINT SOURCE MOCK CATALOGUE FOR TESTING

mock_catalogue_fname = survey.data_dir+'mock_ps_catalogue_'+chan_name[3:]+'_uKcmb_nside512.fits'

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

# %% --- PHOTOMETRY TESTING

def test_photometry(nsources,outfile):

    import IFCAPOL as pol

    lista = []

    for i in range(nsources):

        dic        = {}
        dic['RA']  = epeaks['RA'][i]
        dic['DEC'] = epeaks['DEC'][i]
        dic['I0']  = epeaks['I'][i]
        dic['Q0']  = epeaks['Q'][i]
        dic['U0']  = epeaks['U'][i]
        dic['P0']  = np.sqrt(epeaks['Q'][i]**2+epeaks['U'][i]**2)
        dic['A0']  = pol.pol_angle(epeaks['Q'][i], epeaks['U'][i])

        coord      = total.pixel_to_coordinates(epeaks['Ipix'][i])
        source     = pol.Source.from_coordinate(total, coord)

        dic['I']   = source.I.value
        dic['Q']   = source.Q.value
        dic['U']   = source.U.value
        dic['P']   = source.P.value
        dic['A']   = source.angle.value

        dic['Ie']  = source.I.error
        dic['Qe']  = source.Q.error
        dic['Ue']  = source.U.error
        dic['Pe']  = source.P.error
        dic['Ae']  = source.angle.error

        lista.append(dic)

    tabla = Table(lista)
    tabla.write(testdir+outfile,overwrite=True)

def study_test(cantidad,ntop):

    import matplotlib.pyplot as plt
    from astropy.modeling import models, fitting

    test_nopix = Table.read(testdir+'test_fotometria_nopixel_{0}.fits'.format(chan_name))
    test_sipix = Table.read(testdir+'test_fotometria_sipixel_{0}.fits'.format(chan_name))

    x  = test_nopix['{0}0'.format(cantidad)][0:ntop]
    y  = test_nopix['{0}'.format(cantidad)][0:ntop]
    s  = test_nopix['{0}e'.format(cantidad)][0:ntop]

    fit = fitting.LinearLSQFitter()
    line_init = models.Linear1D()
    fitted_line = fit(line_init, x, y, weights=1.0/s)

    plt.figure(figsize=(8,10))
    plt.errorbar(x, y, yerr=s, fmt='ko', label='Data, no pixel window')
    plt.plot(x, fitted_line(x), 'k-', label='Fitted Model, no pixel window')


    xs = test_sipix['{0}0'.format(cantidad)][0:ntop]
    ys = test_sipix['{0}'.format(cantidad)][0:ntop]
    ss = test_sipix['{0}e'.format(cantidad)][0:ntop]

    fits         = fitting.LinearLSQFitter()
    line_inits   = models.Linear1D()
    fitted_lines = fits(line_inits, xs, ys, weights=1.0/ss)

    plt.errorbar(xs, ys, yerr=ss, fmt='bo', label='Data, with pixel window')
    plt.plot(xs, fitted_lines(x), 'b-', label='Fitted Model, with pixel window')

    plt.xlabel('Input {0}'.format(cantidad))
    plt.ylabel('Estimated {0}'.format(cantidad))
    plt.legend()
    if cantidad == 'I':
        plt.loglog()
    if cantidad == 'P':
        plt.loglog()

    return fitted_line,fitted_lines
