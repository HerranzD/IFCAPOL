#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:34:27 2018

@author: herranz
"""

import numpy as np
import astropy.units as u
import healpy as hp
#import missions as mi
import os

from astropy.table import Table,hstack
from astropy.coordinates import SkyCoord,match_coordinates_sky
from myutils import table2skycoord,coord2healpix,write_blank

# %%  ----- SPECTIFIC INPUT

def load_astrocat(fname):
    t = Table.read(fname)
    for col in t.colnames:
        try:
            if 'degree' in t[col].unit.to_string().lower():
                t[col].unit = u.deg
        except AttributeError:
            pass
    return t

# %%  ----- CATALOGUE PROPERTIES

def get_typical_separations(catalogue):

    c = table2skycoord(catalogue)
    x = c.match_to_catalog_sky(c,nthneighbor=2)
    d = x[1]
    return d.min(),d.mean(),d.max()

# %%  ----- CATALOGUE MATCHING

def coord_in_catalogue(coordinate,catalogue,rsearch):

    c = table2skycoord(catalogue)
    d = c.separation(coordinate)

    if d.min() <= rsearch:
        return True
    else:
        return False

def extract_nearest_object(catalogue,coordinate):

#    Finds and extracts, in a catalogue, the nearest object to a
#    given sky coordinate

    c = table2skycoord(catalogue)
    d = c.separation(coordinate)
    t = catalogue[d==d.min()]
    t['Separation'] = d.min().to(u.deg).value
    t['Separation'].unit = u.deg

    return t



def cat_match(cat1,cat2,radius,
              coord1=None,
              coord2=None,
              return_indexes=False,
              table_names=None):

    if coord1 is not None:
        c1         = coord1
    else:
        c1         = table2skycoord(cat1)

    if coord2 is not None:
        c2         = coord2
    else:
        c2         = table2skycoord(cat2)

    idx,sep,d3 = c1.match_to_catalog_sky(c2)
    mask       = sep<radius

    match1     = cat1[mask]
    match2     = cat2[idx[mask]]

    t          = hstack([match1,match2],table_names=table_names)
    t['Separation'] = sep[mask].to(u.arcsec)

    if return_indexes:
        return t,np.where(mask)[0],idx[mask]
    else:
        return t

def cat1_in_cat2(cat1,cat2,radius,
                 coord1=None,
                 coord2=None,
                 return_indexes=False,
                 table_names=None):

    if coord1 is not None:
        c1         = coord1
    else:
        c1         = table2skycoord(cat1)

    if coord2 is not None:
        c2         = coord2
    else:
        c2         = table2skycoord(cat2)

    idx,sep,d3 = c1.match_to_catalog_sky(c2)
    mask       = sep<=radius

    match1     = cat1[mask]

    if return_indexes:
        return match1,np.where(mask)[0]
    else:
        return match1

def cat1_not_in_cat2(cat1,cat2,radius,
                     coord1=None,
                     coord2=None,
                     return_indexes=False,
                     table_names=None):

    if coord1 is not None:
        c1         = coord1
    else:
        c1         = table2skycoord(cat1)

    if coord2 is not None:
        c2         = coord2
    else:
        c2         = table2skycoord(cat2)

    idx,sep,d3 = c1.match_to_catalog_sky(c2)
    mask       = sep>radius

    match1     = cat1[mask]

    if return_indexes:
        return match1,np.where(mask)[0]
    else:
        return match1

# %%  ----- CLEAN REPEATED POSITIONS

def clean_first_repetition(catalogue,dist):

    c           = table2skycoord(catalogue)
    idx,d2d,d3d = match_coordinates_sky(c,c,nthneighbor=2)

    keeps       = np.ones(len(catalogue),dtype=bool)

    for i in range(len(catalogue)):
        if keeps[i] and d2d[i]<=dist:
            if idx[i] > i:
                keeps[idx[i]] = False

    return catalogue[keeps]

def clean_repetitions(catalogue,dist,verbose=False,return_cleaned=False):

    newcat = catalogue.copy()
    l0     = len(newcat)
    newcat = clean_first_repetition(newcat,dist)
    l1     = len(newcat)

    if verbose:
        print(l0,l1)

    while l0>l1:
        l0     = len(newcat)
        newcat = clean_first_repetition(newcat,dist)
        l1     = len(newcat)
        if verbose:
            print(l0,l1)

    if return_cleaned:
        eliminadas = cat1_not_in_cat2(catalogue.copy(), newcat.copy(), 1*u.arcmin)
        return newcat,eliminadas
    else:
        return newcat




# %%  ----- EFFECTIVE SKY AREA AND HEALPIX PATCHING

def effective_area(table,method='conservative',verbose=True,remove_borders=True):

    dmin,dmean,dmax = get_typical_separations(table)
    if method == 'conservative':
        maxsep = dmax
    else:
        maxsep = dmean
    write_blank(verbose)

    nside  = 2**16
    while hp.nside2resol(nside)*u.rad < maxsep:
        nside = nside//2
    pixres = (hp.nside2resol(nside)*u.rad).to(u.arcmin)

    coord = table2skycoord(table).galactic
    ipix  = coord2healpix(nside,coord)
    ipix  = np.unique(ipix)

    mapa       = np.zeros(hp.nside2npix(nside))
    mapa[ipix] = 1

    if remove_borders:

        nside      = 2*nside
        mapa       = hp.ud_grade(mapa,nside_out=nside,power=0)
        ipix       = np.where(mapa>0)[0]
        neighbors  = mapa[hp.get_all_neighbours(nside,ipix)]
        sums       = neighbors.sum(axis=0)
        ipix       = np.array([ipix[i] for i in range(ipix.size) if sums[i]>=6.0])
        mapa       = np.zeros(hp.nside2npix(nside))
        mapa[ipix] = 1.0

    area  = ipix.size * hp.nside2pixarea(nside,degrees=True) * u.deg * u.deg

    if verbose:
        print(' --- Typical separation between elements = {0} arcmin'.format(maxsep.to(u.arcmin).value))
        print('                        Optimal NSIDE    = {0}'.format(nside))
        print('                        pixel resolution = {0} arcmin'.format(pixres.value))

    write_blank(verbose)
    if verbose:
        print(' --- Total effective area   = {0} deg2'.format(area.value))
        print(' --- Effective sky fraction = {0}'.format((area/(4*np.pi*u.sr)).si.value))
    write_blank(verbose)

    return {'IPIX':ipix,'NSIDE':nside,'AREA':area,'MAP':mapa}


def find_common_area(cat1,cat2,
                     method='conservative',
                     verbose=True,
                     remove_borders=True):
    if verbose:
        write_blank(verbose)
        print(' ---- CATALOGUE 1 ')
        write_blank(verbose)
    dict1 = effective_area(cat1,
                           method=method,
                           verbose=verbose,
                           remove_borders=remove_borders)
    if verbose:
        write_blank(verbose)
        print(' ---- CATALOGUE 2 ')
        write_blank(verbose)
    dict2 = effective_area(cat2,
                           method=method,
                           verbose=verbose,
                           remove_borders=remove_borders)

    nside1 = dict1['NSIDE']
    nside2 = dict2['NSIDE']

    if nside1 == nside2:
        nside = nside1
        map1  = dict1['MAP']
        map2  = dict2['MAP']
    elif nside1 > nside2:
        nside = nside1
        map1  = dict1['MAP']
        map2  = hp.ud_grade(dict2['MAP'],nside_out=nside,power=0)
    else:
        nside = nside2
        map2  = dict2['MAP']
        map1  = hp.ud_grade(dict1['MAP'],nside_out=nside,power=0)

    mask = map1 > 0
    mask = mask & (map2 > 0)

    mapa       = np.zeros(hp.nside2npix(nside))
    mapa[mask] = 1
    ipix       = np.where(mapa>0)[0]
    area       = ipix.size * hp.nside2pixarea(nside,degrees=True) * u.deg * u.deg

    write_blank(verbose)
    if verbose:
        print(' --- Joint effective area   = {0} deg2'.format(area.value))
        print(' --- Effective sky fraction = {0}'.format((area/(4*np.pi*u.sr)).si.value))
    write_blank(verbose)

    return {'IPIX':ipix,'NSIDE':nside,'AREA':area,'MAP':mapa}

# %%  ----- Planck PCCS2

# def load_PCCS2():

#     proot  = '/Users/herranz/Trabajo/Planck/Non_Thermal_Catalogue/Results/PCCS2/'
#     nfreqs = [str(int(x)).rjust(3,'0') for x in mi.Planck.freq_GHz]
#     list0  = os.listdir(proot)
#     pnames = [proot+x for x in list0]

#     PCCS2  = {}

#     for key in nfreqs:
#         nu    = int(key)
#         i     = np.where(mi.Planck.freq_GHz == nu)[0][0]
#         fname = [x for x in pnames if key in x][0]
#         cat0  = Table.read(fname)
#         for ncol in cat0.colnames:
#             try:
#                 if cat0[ncol].unit.to_string() == 'degrees':
#                     cat0[ncol].unit = u.deg
#             except AttributeError:
#                 pass
#             if 'FLUX' in ncol:
#                 cat0[ncol].meta['Freq'] = mi.Planck.freq[i]
#                 cat0[ncol].meta['FWHM'] = mi.Planck.fwhm[i]
#                 cat0[ncol].meta['AREA'] = mi.Planck.beam_area[i]
#                 for nnu in nfreqs:
#                     if nnu in ncol:
#                         j = np.where(mi.Planck.freq_GHz == int(nnu))[0][0]
#                         cat0[ncol].meta['Freq'] = mi.Planck.freq[j]
#                         cat0[ncol].meta['FWHM'] = mi.Planck.fwhm[j]
#                         cat0[ncol].meta['AREA'] = mi.Planck.beam_area[j]

#         PCCS2[key] = cat0

#     return PCCS2

# %%  ----- Planck PCNT

# PCNT_file = '/Users/herranz/Trabajo/Planck/Non_Thermal_Catalogue/Paper/SVN/PIP_127_Herranz/Catalogue/PCNT.fits'

# def load_PCNT():
#     return load_astrocat(PCNT_file)