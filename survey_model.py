#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:27:32 2022

    SURVEY MODEL v1.0
    
    This module includes both local (paths, folders, etc) 
    and global (i.e. instrument, data format, etc) definitions 
    and utilities concerning the configuration of the LiteBIRD
    mission and its data products.


@author: Diego Herranz
"""

# %% --- IMPORTS

import numpy         as np
import astropy.units as u
from fits_maps import Fitsmap
from unit_conversions import Kcmb,mKcmb,uKcmb

# %% --- LOCAL PATH DEFINITIONS

LBdir    = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/'    # main LiteBIRD directory (local)
data_dir = LBdir+'Data/'               # data folder
src_dir  = LBdir+'Src/'                # code folder 
cat_inp  = LBdir+'Catalogs/Input/'     # input point source catalogues
cat_out  = LBdir+'Catalogs/Output/'    # output poit source catalogues

# %% --- LITEBIRD IMO   

IMO = np.load(src_dir+'instrument_LB_IMOv1.npy', allow_pickle=True).item()

for k in IMO.keys():    # add physical units to the IMO dictionary. Beams will be asumed to be FWHM values
    IMO[k]['freq']      = IMO[k]['freq']*u.GHz
    IMO[k]['freq_band'] = IMO[k]['freq_band']*u.GHz
    IMO[k]['beam']      = IMO[k]['beam']*u.arcmin
    IMO[k]['P_sens']    = IMO[k]['P_sens']*uKcmb*u.arcmin
    
# %% --- BASIC MAP INPUT  

def load_LiteBIRD_map(fname,chan_name=None,fwhm=None,freq=None,units_string='mK_cmb'):
    
    """
    
    Reads a LiteBIRD healpix map from a file FNAME. Units are assumed to be
    thermodynamic mK except otherwise said. FWHM and central frequency are provided
    either using the corresponding argument (using physical quantities, e.g. 
    fwhm = 70*u.arcmin) or, preferably, are obtained from the IMO if the channel name is
    provided in the CHAN_NAME argument. CHAN_NAME takes preference over the FWHM and
    FREQ arguments, so if wanted to use different values from the IMO ones the
    CHAN_NAME should not be declared.
    
    """    
    
    if chan_name is None:
        maps = Fitsmap.from_file(fname,freq=freq,fwhm=fwhm)
    else:
        maps = Fitsmap.from_file(fname,
                                 freq=IMO[chan_name]['freq'],
                                 fwhm=IMO[chan_name]['beam'])
        
    maps.set_units(units_string)
    
    return maps
