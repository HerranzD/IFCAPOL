#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 17:57:08 2023

@author: herranz
"""

import os

running_system = 'local'     # it can be 'local' or 'NERSC'

# %% --- PATH DEFINITIONS

if running_system == 'local':


    LBdir    = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/'    # main LiteBIRD directory (local)
    homed    = ''
    scratchd = ''

    data_dir = LBdir+'Data/Run0/'               # data folder
    src_dir  = LBdir+'Src/'                # code folder
    cat_inp  = LBdir+'Catalogs/Input/'     # input point source catalogues
    cat_out  = LBdir+'Catalogs/Output/'    # output poit source catalogues

    IMos_dir = '/Users/herranz/Dropbox/Trabajo/LiteBird/IMo/'

    scriptd  = ''
    map_dir  = ''

elif running_system.upper() == 'NERSC':

    LBdir    = ''
    homed    = os.getenv('HOME')+'/'
    scratchd = os.getenv('SCRATCH')+'/'

    data_dir = '/global/cfs/projectdirs/litebird/simulations/LB_e2e_simulations/e2e_ns512/'
    src_dir  = homed+'LiteBIRD/src/'
    cat_inp  = scratchd+'LiteBIRD/Results/Catalogues/Input/'
    cat_out  = scratchd+'LiteBIRD/Results/Catalogues/Output/'

    IMos_dir = ''

    scriptd  = homed+'LiteBIRD/scripts/'
    map_dir  = scratchd+'LiteBIRD/Data/Maps/'

    # Ensure that cat_in, cat_out, scriptd and map_dir exist. If not, create them (only in NERSC)

    if not os.path.exists(cat_inp):
        os.makedirs(cat_inp)

    if not os.path.exists(cat_out):
        os.makedirs(cat_out)

    if not os.path.exists(scriptd):
        os.makedirs(scriptd)

    if not os.path.exists(map_dir):
        os.makedirs(map_dir)

else:

    print(' WARNING: unknown running system')
