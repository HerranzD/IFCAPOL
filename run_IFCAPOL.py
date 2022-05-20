#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:40:47 2022

@author: herranz
"""

import sys
import PTEP_simulations as PTEP

args        = sys.argv[1]
L           = len(args)
p1          = L-2
p2          = L-1

chan_number = int(sys.argv[p1])
sim_number  = int(sys.argv[p2])

nmax        = len(PTEP.LB_channels)

if chan_number < nmax:
    chan_name = PTEP.LB_channels[chan_number]
    if sim_number < 100:
        PTEP.detect_sources(sim_number,chan_name)
    else:
        print('Wrong simulation number')
else:
    print('Wrong channel number')