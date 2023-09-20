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

    return IMO


