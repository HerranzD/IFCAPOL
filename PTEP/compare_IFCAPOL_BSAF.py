#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:18:17 2023

@author: herranz
"""

import numpy             as np
import matplotlib.pyplot as plt
import astropy.units     as u

from astropy.table   import Table
from myutils         import table2skycoord
from catalogue_tools import cat_match,cat1_not_in_cat2
from linfit_errorsxy import linfit_errxy,plotfit

catdir       = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/Catalogs/Output/'
IFCAPOL_catn = catdir+'IFCAPOL_catalogue_postPTEP_LB_LFT_40_0000_after_cleaned.fits'
  #'IFCAPOL_catalogue_postPTEP_LB_LFT_40_0000_SNR3p5.fits'
IFCAPOL_cat  = Table.read(IFCAPOL_catn)

MHW_catn     = catdir+'cat_40_lb_5col.dat'
MHW_cat      = Table.read(MHW_catn,format='ascii')


def galcut_catalogue(catalogo,galcut_deg):
    c    = table2skycoord(catalogo)
    b    = c.galactic.b.deg
    absb = np.abs(b)
    return catalogo[absb>=galcut_deg]

def join_catalogues(SNRcut,galcut,trampa=False):

    if trampa:
        x = IFCAPOL_cat.copy()
        x['I err [uK_CMB]'] = x['I err [uK_CMB]']*0.75
        x['I SNR']          = x['I SNR']/0.75
        subcat1 = galcut_catalogue(x, galcut)
    else:
        subcat1 = galcut_catalogue(IFCAPOL_cat, galcut)

    subcat2 = galcut_catalogue(MHW_cat, galcut)

    subcat1 = subcat1[subcat1['I SNR']>=SNRcut]
    subcat2 = subcat2[subcat2['SNR']>=SNRcut]

    print(' ')
    print(' IFCAPOL has {0} sources above |b| >= {1} deg and SNR >= {2}'.format(len(subcat1),galcut,SNRcut))
    print(' MHW2    has {0} sources above |b| >= {1} deg and SNR >= {2}'.format(len(subcat2),galcut,SNRcut))

    jnt = cat_match(subcat1,subcat2,radius=20*u.arcmin)

    print(' There are {0} coincidences within a 20 arcmin radius '.format(len(jnt)))
    print(' ')

    plt.close('all')

    x  = jnt['FLUX']
    sx = jnt['FLUX']/jnt['SNR']
    y  = jnt['I [uK_CMB]']
    sy = jnt['I err [uK_CMB]']

    fit1,fit2 = linfit_errxy(x,y,sx,sy)

    print(fit1.beta)

    plotfit(x,y,sx,sy,fit2,
            addunit   = True,
            addfit    = True,
            logscal   = True,
            x_label   = 'MHW Flux',
            y_label   = 'IFCAPOL Flux',
            newfig    = True,
            subplotn  = 111,
            linewidth = 0.5,
            capsize   = 2)

    plt.grid()

    plt.savefig('/Users/herranz/Desktop/flux_comparison_{0}sigma.pdf'.format(SNRcut))

    plt.figure()
    plt.scatter(sx,sy)
    plt.xlabel(r'$\sigma$ MHW2')
    plt.ylabel(r'$\sigma$ IFCAPOL')
    plt.loglog()
    plt.grid()

    print(' ')
    gain = sx/sy
    print(' Sigma MHW2 / sigma IFCAPOL = {0} +- {1}'.format(gain.mean(),gain.std()))
    print(' ')

    return jnt,fit1,fit2


def study_non_matched(SNRcut,galcut):

    plt.close('all')

    subcat1 = galcut_catalogue(IFCAPOL_cat, galcut)
    subcat2 = galcut_catalogue(MHW_cat, galcut)

    subcat1 = subcat1[subcat1['I SNR']>=SNRcut]
    subcat2 = subcat2[subcat2['SNR']>=SNRcut]

    radio   = 20*u.arcmin

    print(' ')
    print(' IFCAPOL has {0} sources above |b| >= {1} deg and SNR >= {2}'.format(len(subcat1),galcut,SNRcut))
    print(' MHW2    has {0} sources above |b| >= {1} deg and SNR >= {2}'.format(len(subcat2),galcut,SNRcut))

    mias_solo = cat1_not_in_cat2(subcat1,subcat2,radio)

    print(' ')
    print(' There are {0} sources in the IFCAPOL catalogue not in MHW2'.format(len(mias_solo)))

    plt.figure()
    plt.scatter(mias_solo['GLON [deg]'],mias_solo['GLAT [deg]'])
    plt.xlabel('GLON')
    plt.ylabel('GLAT')

    marcos_solo = cat1_not_in_cat2(subcat2,subcat1,radio)
    print(' There are {0} sources in the MHW2 catalogue not in IFCAPOL'.format(len(marcos_solo)))

    plt.figure()
    plt.scatter(marcos_solo['GLON'],marcos_solo['GLAT'])
    plt.xlabel('GLON')
    plt.ylabel('GLAT')

    plt.figure()
    plt.scatter(mias_solo['GLON [deg]'],mias_solo['GLAT [deg]'],label='IFCAPOL')
    plt.scatter(marcos_solo['GLON'],marcos_solo['GLAT'],label='MHW2')
    plt.legend()
    plt.xlabel('GLON')
    plt.ylabel('GLAT')





    return mias_solo,marcos_solo

