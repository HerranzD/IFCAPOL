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
         get_lb_instrument() and IMoV2 interpreter by Giuseppe Puglisi
"""

# %% --- IMPORTS

import numpy         as np
import astropy.units as u
import json

from astropy.table    import QTable
from fits_maps        import Fitsmap
from unit_conversions import uKcmb

from path_defs import LBdir,homed,scratchd
from path_defs import data_dir,src_dir,cat_inp,cat_out
from path_defs import IMos_dir
from path_defs import scriptd,map_dir


# %% --- Simulation specific definitions

fknee    = '100mHz'
sim_type = 'binned'


# %% --- LITEBIRD IMO

IMo_version = 'V2'   # The currently used version of IMo is V2


def read_IMo(IMOversion,IMo_from='tbl'):
    '''
    Reads the Instrument Model parameters.

    Parameters
    ----------
    IMOversion : str
        The version of the LiteBIRD IMo. Form example, 'IMo.v1.json'
    IMo_from : str, optional
        Chooses between limited input from a table or full input
        from a json file (for this, set IMO_from ='json').
        The default is 'tbl'.

    Returns
    -------
    dict
        A nested dictionary containing the IMo model parameters. The
        main level keys are the names of the detectors (channels)

    '''

    if 'ptep' in IMOversion.lower():

        IMo_dir = IMos_dir+'iMo.v1/'
        IMO = np.load(IMo_dir+'instrument_LB_IMOv1.npy', allow_pickle=True).item()

        for k in IMO.keys():    # add physical units to the IMO dictionary.
                                # Beams will be asumed to be FWHM values.
            IMO[k]['freq']      = IMO[k]['freq']*u.GHz
            IMO[k]['freq_band'] = IMO[k]['freq_band']*u.GHz
            IMO[k]['beam']      = IMO[k]['beam']*u.arcmin
            IMO[k]['P_sens']    = IMO[k]['P_sens']*uKcmb*u.arcmin

    if 'v1.3' in IMOversion.lower():

        IMo_dir      = IMos_dir
        IMov1p3_dir  = IMo_dir+'iMo.V1.3/'
        IMo_file     = IMov1p3_dir+'schema_131022.json'
        IMO = {}

    if 'v2' in IMOversion.lower():

        IMo_dir      = IMos_dir+'IMoV2-14June/'
        IMo_textfile = IMo_dir+'litebird_instrument_model.tbl'
        IMo_jsonfile = IMo_dir+'IMoV2-14June.json'

        if IMo_from.lower() == 'tbl':

            imo = QTable.read(IMo_textfile, format='ascii.ipac' )
            imo.add_index('tag')

        else:

            def get_lb_instrument(data ):
                '''
                Code by G. Puglisi.
                This routine interprets the IMoV2 json file

                Parameters
                ----------
                data : json
                    A json structure cointaining the data to be extracted.

                Returns
                -------
                strings : list of strings
                    Instrumen names.
                freqs : list of floats
                    Central frequencies.
                fwhms : list of floats
                    Intrument FWHMs.
                sensitivities : list of floats
                    Instrument sensitivities.
                bandwidth : list of floats
                    Bandwidths.

                '''

                freqs=[]
                fwhms = []
                sensitivities  = []
                bandwidth  = []
                strings= []

                offset = 1
                for i in range(24)   :
                    try :
                        ndets =data['data_files'][offset]['metadata']['number_of_detectors']

                        offset+=1
                        channelname =data['data_files'][offset]['metadata']['channel']
                        strings.append(channelname)
                        f = data['data_files'][offset]['metadata']['bandcenter_ghz']

                        if f  in freqs:
                            f+=1
                            freqs.append( f   )
                        else:
                            freqs.append( f  )
                        bandwidth.append(data['data_files'][offset]['metadata']['bandwidth_ghz'])
                        fwhms.append(data['data_files'][offset]['metadata']['fwhm_arcmin'])
                        sensitivities.append(data['data_files'][offset]['metadata']['net_ukrts']/np.sqrt(ndets))
                        offset +=ndets
                        #print(channelname , f, bandwidth[-1] )
                    except KeyError:
                        offset +=1

                    i+=1

                return strings,freqs,fwhms,sensitivities, bandwidth

            file = IMo_jsonfile

            with open(file) as json_file:
                    hw  = json.load(json_file)

            strings,freqs,fwhms,sensitivities, bandwidth = get_lb_instrument(hw)

            imo = QTable(data=[strings,
                               freqs*u.GHz,
                               fwhms*u.arcmin,
                               sensitivities*u.uK*u.second**.5,
                               bandwidth*u.GHz],
                         names=['tag','center_frequency',
                                'fwhm','NET', 'bandwidth'])

            for i, field in enumerate(['telescope', 'band']):
                imo[field] = [t.split('-')[i] for t in imo['tag']]

            for r in imo:
                band = float(r['band']) * u.GHz
            if not u.isclose(band, r['center_frequency']):
                print(r['tag'], 'replace', r['center_frequency'], 'with', band)
                r['center_frequency'] = band

            high_freq = imo['center_frequency'] > 200 * u.GHz
            imo['nside'] = 512
            imo['nside'][high_freq] = 1024

        IMO = {}
        for i in range(len(imo)):
            k = imo['tag'][i]
            d = {'freq':imo['center_frequency'][i],
                 'freq_band':imo['bandwidth'][i],
                 'beam':imo['fwhm'][i],
                 'band':imo['band'][i],
                 'NET':imo['NET'][i],
                 'telescope':imo['telescope'][i],
                 'nside':imo['nside'][i]}
            IMO[k] = d

    return IMO

IMO         = read_IMo(IMo_version)

LBc         = [x for x in IMO.keys()]

if IMo_version.lower() != 'ptep':
    LB_channels = sorted(LBc, key=lambda x: int(x[-3:]))
else:
    LB_channels = LBc.copy()

# %% --- BASIC MAP INPUT

def load_LiteBIRD_map(fname,chan_name=None,fwhm=None,freq=None,units_string='uK_CMB'):

    """

    Reads a LiteBIRD healpix map from a file FNAME. Units are assumed to be
    thermodynamic mK except otherwise said. FWHM and central frequency are
    provided either using the corresponding argument
    (using physical quantities, e.g. fwhm = 70*u.arcmin) or, preferably,
    are obtained from the IMO if the channel name is provided in the CHAN_NAME
    argument. CHAN_NAME takes preference over the FWHM and FREQ arguments,
    so if wanted to use different values from the IMO ones the CHAN_NAME
    should not be declared.

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
