#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 19:26:33 2022

@author: herranz
"""

import numpy
import healpy            as hp
import matplotlib.pyplot as plt

from astropy.table       import Table


catdir = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/Catalogs/Output/'

fname_40 = catdir+'Non_Matched/LB_LFT_40_non_matched.fits'
fname_50 = catdir+'Non_Matched/LB_LFT_40_non_matched.fits'

refmap_name  = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/'
refmap_name += 'Data/LB_LFT_40_testing_map_0000_PTEP_20200915_compsep.fits'

refmap       = hp.read_map(refmap_name)
catalogue    = Table.read(fname_40)

def plot_catalogue(mapa,catalogue,snr_cut=5.0,xsize=2400,title=None,tofile=None):

    hp.mollview(mapa,norm='hist',cbar=False,flip='astro',xsize=xsize)
    plt.title(' ')

    cut_catalogue = catalogue[catalogue['I SNR']>=snr_cut]

    x = cut_catalogue['GLON [deg]']
    y = cut_catalogue['GLAT [deg]']
#    c = 1.0-cut_catalogue['SIM_NUMBER']*0.01
    hp.projscatter(x, y, lonlat=True, coord='G',c='red',marker='o',alpha=0.1)
    hp.graticule()

    if title is not None:
        plt.title(title)

    # if tofile is not None:
    #     plt.savefig(tofile)
