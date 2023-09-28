#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:04:18 2023

@author: herranz
"""

import os
import pysftp

version = 'Run0'

nersc_dir = '/global/cfs/projectdirs/litebird/simulations/LB_e2e_simulations/e2e_ns512/'

instr_list = ['LFT', 'MFT', 'HFT']

chans_list = {'LFT':['L1-040','L1-078',
                     'L2-068','L3-068',
                     'L3-119','L4-100',
                     'L1-060','L2-050',
                     'L2-089','L3-089',
                     'L4-078','L4-140'],
              'MFT':['M1-100','M1-140',
                     'M1-195','M2-119',
                     'M2-166'],
              'HFT':['H1-195','H1-280',
                     'H2-235','H2-337',
                     'H3-402']}

fnkee_list = ['030','100']

typ_list   = ['binned','destriped']

def fname_sim(instr,chan,fknee,typ,nsim):
    fname =  nersc_dir
    fname += instr+'/'
    fname += chan+'/'
    fname += typ+'_maps/'
    fname += 'LB_{0}_{1}_{2}_cmb_fg_wn_1f_{3}mHz_{4}.fits'.format(instr,
                                                                  chan,
                                                                  typ,
                                                                  fknee,
                                                                  nsim)
    return fname

def list_all_cases(nsim,typ,fknee):
    """
    Returns a list with all the files to be downloaded
    """
    l = []
    for inst in instr_list:
        for chan in chans_list[inst]:
            l.append(fname_sim(inst,chan,fknee,typ,nsim))
    return l


# write a file with a get command for each element in the list
def write_get_file(l):
 
    with open('get_files.txt','w') as f:
        for fname in l:
            f.write('get '+fname+'\n')
    return



