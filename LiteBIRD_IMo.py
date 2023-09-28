#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:17:08 2023

@author: herranz
"""

import numpy as np
import os
import json
from astropy.table import QTable
import matplotlib.pyplot as plt

IMo_from = 'tbl'   # chooses between limited input from a table
                   # or full input from json file (for this, set
                   # IMO_from ='json')

def read_IMo(IMOversion):

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



            imo = QTable.read(IMo_textfile, format="ascii.ipac" )
            imo.add_index('tag')

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

        else:

            def get_lb_instrument(data ):
                """
                Code by G. Puglisi.
                This routine interprets the IMoV2 json file

                Parameters
                ----------
                data : TYPE
                    DESCRIPTION.

                Returns
                -------
                strings : TYPE
                    DESCRIPTION.
                freqs : TYPE
                    DESCRIPTION.
                fwhms : TYPE
                    DESCRIPTION.
                sensitivities : TYPE
                    DESCRIPTION.
                bandwidth : TYPE
                    DESCRIPTION.

                """

                freqs=[]
                fwhms = []
                sensitivities  = []
                bandwidth  = []
                strings= []


                offset = 1
                for i in range(24)   :
                    try :
                        ndets =data["data_files"][offset]['metadata']['number_of_detectors']

                        offset+=1
                        channelname =data["data_files"][offset]['metadata']['channel']
                        strings.append(channelname)
                        f = data["data_files"][offset]['metadata']['bandcenter_ghz']

                        if f  in freqs:
                            f+=1
                            freqs.append( f   )
                        else:
                            freqs.append( f  )
                        bandwidth.append(data["data_files"][offset]['metadata']['bandwidth_ghz'])
                        fwhms.append(data["data_files"][offset]['metadata']['fwhm_arcmin'])
                        sensitivities.append(data["data_files"][offset]['metadata']['net_ukrts']/np.sqrt(ndets))
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

            IMO = {'strings':strings,
                   'freqs':freqs,
                   'fwhms':fwhms,
                   'sensitivities':sensitivities}

    return IMO


