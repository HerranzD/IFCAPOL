#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Sep 19 15:17:08 2023

@author: herranz
'''

import numpy as np
import json
from astropy.table import QTable
import astropy.units as u


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

    if 'v1.3' in IMOversion.lower():

        IMo_dir      = '/Users/herranz/Dropbox/Trabajo/LiteBird/IMo/'
        IMov1p3_dir  = IMo_dir+'iMo.V1.3/'
        IMo_file     = IMov1p3_dir+'schema_131022.json'
        IMO = {}

    if 'v2' in IMOversion.lower():

        IMo_dir      = '/Users/herranz/Dropbox/Trabajo/LiteBird/IMo/IMoV2-14June/'
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


