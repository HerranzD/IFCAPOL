#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:03:17 2024

@author: herranz
"""

from astropy.coordinates import SkyCoord

import IFCAPOL           as     pol
from survey_model        import load_LiteBIRD_map
from file_names          import simulation_name
from survey_model        import LB_channels

fname = simulation_name(LB_channels[0],0)
coord = SkyCoord.from_name('Crab')

smap  = load_LiteBIRD_map(fname,chan_name=LB_channels[0])
s     = pol.Source.from_coordinate(smap, coord)

print(' Hola, he sacado una fuente en la coordenada {0}'.format(coord))
