#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:45:31 2022

@author: herranz
"""

import survey_model as survey 

coadded_file = survey.data_dir+'LB_LFT_40_coadd_signal_map_0000_PTEP_20200915_compsep.fits'
noise_file   = survey.data_dir+'LB_LFT_40_noise_FULL_0000_PTEP_20200915_compsep.fits'
ps_file      = survey.data_dir+'radio_sources_LFT_40_uKcmb_nside512.fits'

chan_name    = 'LB_LFT_40'

signal  = survey.load_LiteBIRD_map(coadded_file,chan_name=chan_name)
noise   = survey.load_LiteBIRD_map(noise_file,chan_name=chan_name)
radiops = survey.load_LiteBIRD_map(ps_file,chan_name=chan_name)

total   = signal+noise+radiops