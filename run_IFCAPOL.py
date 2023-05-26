#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for calling PTEP_simulations.detect_sources from command line

@author: herranz
"""

import os
import sys
import PTEP_simulations as PTEP

args        = sys.argv
L           = len(args)
p1          = L-2
p2          = L-1

chan_number = int(sys.argv[p1])
sim_number  = int(sys.argv[p2])

nmax        = len(PTEP.LB_channels)

if chan_number < nmax:
    chan_name = PTEP.LB_channels[chan_number]
    if sim_number < 100:
        fname = PTEP.cleaned_catalogue_name(sim_number,chan_name)
        if os.path.isfile(fname):
            print(' Output catalogue already exists for sim {0} in {1}'.format(sim_number,
                                                                               chan_name))
        else:
            print(' IFCAPOL detecting in sim {0} in {1}'.format(sim_number,
                                                                chan_name))
            dicia = PTEP.detect_sources(sim_number,chan_name)
    else:
        print('Wrong simulation number')
else:
    print('Wrong channel number')