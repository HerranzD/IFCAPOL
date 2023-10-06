"""
This script generates Slurm scripts needed to run the IFCAPOL source extraction over all the LiteBIRD postPTEP Run0 simulations at NERSC. It also generates shell scripts needed to submit the IFCAPOL source extraction over a list of LiteBIRD postPTEP Run0 simulations at NERSC. The script checks the status of the submissions for a given channel and generates a new submission script for the simulations that do not have an output log file and do not have both output catalogues. 

Functions:
----------
make_script(isim,ichan,hours=1,minutes=30)
    Generates a Slurm script needed to run the IFCAPOL source extraction over one of the LiteBIRD postPTEP Run0 simulations at NERSC.
    
generate_slurm_scripts()
    Generates the Slurm scripts needed to run the IFCAPOL source extraction over all the LiteBIRD postPTEP Run0 simulations at NERSC.
    
generate_submission_script(ichan,sim_list=None)
    Generates a shell script needed to submit the IFCAPOL source extraction over a list of LiteBIRD postPTEP Run0 simulations at NERSC.
    
test_generate_submission_script()
    Tests the generation of a submission script for a single channel and a given list of simulations.
    
generate_all_submission_scripts()
    Generates the shell scripts needed to submit the IFCAPOL source extraction over all the LiteBIRD postPTEP Run0 simulations at NERSC.
    
check_submission_status(ichan)
    Checks the status of the submissions for a given channel and generates a new submission script for the simulations that do not have an output log file and do not have both output catalogues.
"""

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
    This routine runs the routines from the command line. The argument is
    given when the script is called from the command line. The argument can
    be 'generate_slurm_scripts', 'generate_all_submission_scripts',
    'check_submission_status' or 'test_generate_submission_script'.

    Returns
    -------
    None.

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
    else:
        print('ERROR: no argument given')