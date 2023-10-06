#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:20:39 2022

@author: herranz
"""

from myutils import save_ascii_list
import run0_simulations as postPTEP
import os
import sys

nchans = len(postPTEP.survey.LB_channels)
nsims  = 200
Njobs  = nsims

# %% --- SLURM SCRIPT GENERATION

def make_script(isim,ichan,hours=1,minutes=30):
    """
    This routine generates automatically a Slurm script needed to run
    the IFCAPOL source extraction over one of the LiteBIRD postPTEP Run0
    simulations at NERSC.

    Parameters
    ----------
    isim : int
        The simulation number.
    ichan : int
        The channel number.
    hours : int, optional
        The number of hours to run the simulation for. Default is 1.
    minutes : int, optional
        The number of minutes to run the simulation for. Default is 30.

    Returns
    -------
    None.

    """
    time_str = '0{0}:{1}:00'.format(hours,minutes)

    lsta = []
    lsta.append('#!/bin/bash')
    lsta.append('#SBATCH -N 1')
    lsta.append('#SBATCH -C haswell')
    lsta.append('#SBATCH -q regular')
    lsta.append('#SBATCH -J IFCAPOL_{0}_{1}'.format(ichan,isim))
    lsta.append('#SBATCH --mail-user=herranz@ifca.unican.es')
    lsta.append('#SBATCH --mail-type=ALL')
    lsta.append('#SBATCH --account=mp107')
    lsta.append('#SBATCH -t {0}'.format(time_str))
    lsta.append('#SBATCH --output={0}Output_Logs/IFCAPOL_nchan{1}_nsim{2}.out'.format(postPTEP.survey.scriptd,ichan,isim))
    lsta.append('#SBATCH --chdir={0}'.format(postPTEP.survey.scriptd))
    lsta.append(' ')
    lsta.append('#run the application:')
    lsta.append('module load python')
    lsta.append('source activate pycmb')
    lsta.append('srun -n 1 -c 1 -t {0} python3 $HOME/LiteBIRD/src/run_IFCAPOL_on_postPTEP.py {1} {2}'.format(time_str,ichan,isim))
    lsta.append('conda deactivate')

    macro_name = postPTEP.survey.scriptd+'Run_scripts/run_nchan{0}_nsim{1}.slurm'.format(ichan,isim)
    save_ascii_list(lsta,macro_name)

def generate_slurm_scripts():
    """
    This routine generates automatically the Slurm scripts needed to run
    the IFCAPOL source extraction over all the LiteBIRD postPTEP Run0
    simulations at NERSC. The number of hours to run the simulations increases
    by one hour for each five simulations, starting with 1 for simulation 0.

    Returns
    -------
    None.

    """
    for ichan in range(nchans):
        for isim in range(nsims):
            hours   = 1 + isim//5
            minutes = 30
            make_script(isim,ichan,hours,minutes)

# %% --- SUBMISSION SCRIPTS

def generate_submission_script(ichan,sim_list=None):
    """
    This routine generates automatically a shell script needed to submit
    the IFCAPOL source extraction over a list of LiteBIRD postPTEP Run0
    simulations at NERSC. The list refer to a set of simulations for a
    given channel.

    Parameters
    ----------
    ichan : int
        The channel number. Can take values from 0 to the number of
        LiteBIRD channels (22) minus one.
    sim_list : list, optional
        The list of simulation numbers to run. If None, all the simulations
        for the given channel are run. The default is None. The list can be
        a subset of the simulations for the given channel.

    Returns
    -------
    None.

    """

    if sim_list is not None:
        lista = sim_list.copy()
    else:
        lista = [i for i in range(nsims)]

    fname = postPTEP.survey.scriptd+'Submit_scripts/submit_nchan{0}.sh'.format(ichan)

    lsubmits = []

    for isim in lista:
        macro_name = postPTEP.survey.scriptd+'Run_scripts/run_nchan{0}_nsim{1}.slurm'.format(ichan,
                                                                                 isim)
        lsubmits.append('sbatch '+macro_name)

    save_ascii_list(lsubmits,fname)

# Test the routine with a single simulation for channel 0 and sim_list = [0,1,2]

def test_generate_submission_script():
    """
    This routine tests the generation of a submission script for a single
    channel and a given list of simulations.

    Returns
    -------
    None.

    """
    ichan = 0
    sim_list = [0,1,2]
    generate_submission_script(ichan,sim_list)


# %% --- SUBMISSION SCRIPTS FOR ALL CHANNELS

def generate_all_submission_scripts():
    """
    This routine generates automatically the shell scripts needed to submit
    the IFCAPOL source extraction over all the LiteBIRD postPTEP Run0
    simulations at NERSC. The number of hours to run the simulations increases
    by one hour for each five simulations, starting with 1 for simulation 0.

    Returns
    -------
    None.

    """
    for ichan in range(nchans):
        generate_submission_script(ichan)

# %% --- CHECKING STATUS UF SUBMISSIONS

def check_submission_status(ichan):
    """
    This routine checks the status of the submissions for a given channel.
    The routine checks if the output log file exists for each simulation.
    It also checks if the output catalogues, with names defined by
    postPTEP.detected_catalogue_name() and
    postPTEP.cleaned_catalogue_name(), exist for each simulation.
    For every simulation that does not have an output log file and does not
    have both output catalogues, the routine creates a list and generates a
    new submission script for the simulations in the list using
    generate_submission_script().

    Parameters
    ----------
    ichan : int
        The channel number. Can take values from 0 to the number of
        LiteBIRD channels (22) minus one.

    Returns
    -------
    None.

    """

    # check if the output log file exists for each simulation
    # check if the output catalogues exist for each simulation
    # if not, create a list of simulations and generate a new submission script

    # check if the output log file exists for each simulation
    lsim = []
    for isim in range(nsims):
        fname = postPTEP.survey.scriptd+'Output_Logs/IFCAPOL_nchan{0}_nsim{1}.out'.format(ichan,isim)
        if not os.path.isfile(fname):
            lsim.append(isim)

    # check if the output catalogues exist for each simulation
    lsim2 = []
    for isim in lsim:
        fname = postPTEP.detected_catalogue_name(isim,ichan) # the order of the arguments in this routine is different from the order in the other routines
        if not os.path.isfile(fname):
            lsim2.append(isim)
        else:
            fname = postPTEP.cleaned_catalogue_name(isim,ichan) # the order of the arguments in this routine is different from the order in the other routines
            if not os.path.isfile(fname):
                lsim2.append(isim)

    # merges lsim and lsim2 and removes duplicates
    lsim2 = list(set(lsim2 + lsim))

    # generate a new submission script for the simulations in the list
    generate_submission_script(ichan,lsim2)

    # write a log of the stauts of the submissions to a file named 'status_{0}.txt'.format(ichan) located
    # in the directory postPTEP.survey.scriptd

    fname = postPTEP.survey.scriptd+'status_{0}.txt'.format(ichan)
    lstatus = []
    lstatus.append('Channel {0}'.format(ichan))
    lstatus.append(' ')
    lstatus.append('Number of simulations: {0}'.format(nsims))
    lstatus.append(' ')
    lstatus.append('Number of simulations without output log file: {0}'.format(len(lsim)))
    lstatus.append(' ')
    lstatus.append('Number of simulations without output catalogues: {0}'.format(len(lsim2)))
    lstatus.append(' ')
    lstatus.append('List of simulations without output log file:')
    lstatus.append(' ')
    lstatus.append(str(lsim))
    lstatus.append(' ')
    lstatus.append('List of simulations without output catalogues:')
    lstatus.append(' ')
    lstatus.append(str(lsim2))
    lstatus.append(' ')
    save_ascii_list(lstatus,fname)


# %% --- RUN THE ROUTINES FROM THE COMMAND LINE

if __name__ == '__main__':
    """
    This routine runs the routines in this file from the command line. It
    can take  four arguments:
        generate_slurm_scripts
        generate_all_submission_scripts
        check_submission_status
        test_generate_submission_script
    The first three arguments run the routines with the same name. The last
    argument runs the routine test_generate_submission_script().
    """

    if len(sys.argv) > 1:
        if sys.argv[1] == 'generate_slurm_scripts':
            generate_slurm_scripts()
        elif sys.argv[1] == 'generate_all_submission_scripts':
            generate_all_submission_scripts()
        elif sys.argv[1] == 'check_submission_status':
            if len(sys.argv) > 2:
                ichan = int(sys.argv[2])
                check_submission_status(ichan)
            else:
                print('ERROR: check_submission_status requires a channel number as argument')
        elif sys.argv[1] == 'test_generate_submission_script':
            test_generate_submission_script()
        else:
            print('ERROR: invalid argument')
            print(' ')
            print(' Valid arguments:')
            print(' ')
            print('     generate_slurm_scripts')
            print('     generate_all_submission_scripts')
            print('     check_submission_status')
            print('     test_generate_submission_script')
            print(' ')

    else:
        print('ERROR: no argument given')
        print(' ')
        print(' Valid arguments:')
        print(' ')
        print('     generate_slurm_scripts')
        print('     generate_all_submission_scripts')
        print('     check_submission_status')
        print('     test_generate_submission_script')
        print(' ')