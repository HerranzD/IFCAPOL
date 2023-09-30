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

import os
import numpy         as np
import astropy.units as u
from fits_maps import Fitsmap
from unit_conversions import uKcmb

from path_defs import LBdir,homed,scratchd
from path_defs import data_dir,src_dir,cat_inp,cat_out
from path_defs import scriptd,map_dir


# %% --- LITEBIRD IMO

IMo_version = 'V2'

if IMo_version == 'PTEP':

    IMO = np.load(src_dir+'instrument_LB_IMOv1.npy', allow_pickle=True).item()

    for k in IMO.keys():    # add physical units to the IMO dictionary.
                            # Beams will be asumed to be FWHM values.

        IMO[k]['freq']      = IMO[k]['freq']*u.GHz
        IMO[k]['freq_band'] = IMO[k]['freq_band']*u.GHz
        IMO[k]['beam']      = IMO[k]['beam']*u.arcmin
        IMO[k]['P_sens']    = IMO[k]['P_sens']*uKcmb*u.arcmin

else:

    from LiteBIRD_IMo import read_IMo
    IMO = read_IMo(IMo_version)

# %% --- BASIC MAP INPUT

def load_LiteBIRD_map(fname,chan_name=None,fwhm=None,freq=None,units_string='uK_CMB'):

    """

    Reads a LiteBIRD healpix map from a file FNAME. Units are assumed to be
    thermodynamic mK except otherwise said. FWHM and central frequency are provided
    either using the corresponding argument (using physical quantities, e.g.
    fwhm = 70*u.arcmin) or, preferably, are obtained from the IMO if the channel name is
    provided in the CHAN_NAME argument. CHAN_NAME takes preference over the FWHM and
    FREQ arguments, so if wanted to use different values from the IMO ones the
    CHAN_NAME should not be declared.

    Parameters
    ----------
    fname : string
        The name of the file containing the map to be read.

    chan_name : string
        The standard name of one of the LiteBIRD channels. If None, the routine
        will look for the FWHM and central frequency FREQ in the corresponding
        parameters.

    fwhm: astropy.units.quantity.Quantity
        The FWHM of the observing beam. It is overrided by the IMO data if
        CHAN_NAME is set to one of the LiteBIRD frequencies.

    freq: astropy.units.quantity.Quantity
        The central frequency at which the image has been observed.
        It is overrided by the IMO data if CHAN_NAME is set to one of the
        LiteBIRD frequencies.

    units_string : string
        The units of the map in string format. 'K_CMB', "mK_CMB" and "uK_CMB"
        stand for thermodynamic temperature in Kelvin, milli Kelvin or micro
        Kelvin, respectively.

    Returns
    -------
    Image object containing I,Q,U healpix sky map

    """

    if chan_name is None:
        maps = Fitsmap.from_file(fname,freq=freq,fwhm=fwhm)
    else:
        maps = Fitsmap.from_file(fname,
                                 freq=IMO[chan_name]['freq'],
                                 fwhm=IMO[chan_name]['beam'])

    maps.set_units(units_string)

    return maps
