#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:23:55 2022

@author: herranz
"""

import os
import survey_model     as survey
import PTEP_simulations as simuls

chan_name    = 'LB_LFT_40'


simnum = 10
diccio = simuls.PTEP_simulated_maps(simnum,chan_name)

simstr = '{0:04d}'.format(simnum)

outdir = survey.map_dir+simstr+'/'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

diccio['TOTAL'].write(outdir+'total_'+chan_name+'_'+simstr+'.fits')

