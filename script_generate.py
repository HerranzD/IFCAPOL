#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:20:39 2022

@author: herranz
"""

from myutils import save_ascii_list
import PTEP_simulations as PTEP

nchans = len(PTEP.LB_channels)
nsims  = 100

command_list = []

for ichan in range(nchans):
    for isim in range(nsims):

        lsta = []
        lsta.append('#!/bin/bash')
        lsta.append('#SBATCH -N 1')
        lsta.append('#SBATCH -C haswell')
        lsta.append('#SBATCH -q regular')
        lsta.append('#SBATCH -J IFCAPOL_{0}_{1}'.format(ichan,isim))
        lsta.append('#SBATCH --mail-user=herranz@ifca.unican.es')
        lsta.append('#SBATCH --mail-type=ALL')
        lsta.append('#SBATCH --account=mp107')
        lsta.append('#SBATCH -t 01:30:00')
        lsta.append('#SBATCH --output={0}Output_Logs/IFCAPOL_nchan{1}_nsim{2}.out'.format(PTEP.survey.scriptd,ichan,isim))
        lsta.append('#SBATCH --chdir={0}'.format(PTEP.survey.scriptd))
        lsta.append(' ')
        lsta.append('#run the application:')
        lsta.append('module load python')
        lsta.append('source activate pycmb')
        lsta.append('srun -n 1 -c 1 python3 $HOME/LiteBIRD/src/run_IFCAPOL.py {0} {1}'.format(ichan,isim))
        lsta.append('conda deactivate')

        macro_name = PTEP.survey.scriptd+'submit_nchan{0}_nsim{1}.sh'.format(ichan,isim)
        save_ascii_list(lsta,macro_name)
        command_list.append('sbatch {0}'.format(macro_name))

fname = PTEP.survey.scriptd+'send.sh'
save_ascii_list(command_list,fname)




