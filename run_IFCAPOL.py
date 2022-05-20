#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:40:47 2022

@author: herranz
"""

import sys
import PTEP_simulations as PTEP

chan_number = sys.argv[0]
sim_number  = sys.argv[1]

nmax        = len(PTEP.LB_channels)

if chan_number <= nmax:
    chan_name = PTEP.LB_channels[chan_number-1]
    if sim_number < 100:
        PTEP.detect_sources(sim_number,chan_name)
    else:
        print('Wrong simulation number')
else:
    print('Wrong channel number')