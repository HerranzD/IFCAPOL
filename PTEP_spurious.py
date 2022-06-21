#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:59:55 2022

@author: herranz
"""

import os

import numpy              as np
import PTEP_simulations   as PTEP
import quality_assessment as QA
import survey_model       as Table,survey
import astropy.units      as u

from astropy.table import vstack


nchans = len(PTEP.LB_channels)
nsims  = 100

def get_catalogues(isim,ichan):
    """
    Given a simulation index and a channel index, returns (if possible),
    the PTEP reference point source catalogue for that channel and the
    IFCAPOL catalogue of detections. If the IFCAPOL catalogue does not exist,
    the routine returns the boolean False value.

    Parameters
    ----------
    isim : int
        Simulation index (between 0 and 99).
    ichan : TYPE
        Channel index (between 0 and **nchans**-1).

    Returns
    -------
    dict or bool
        If there an IFCAPOL detection catalogue exists, the routine returns
        a dictionary containing the `reference' and the `IFCAPOL` catalogues.
        It not, the routine returns FALSE.

    """

    chan_name       = PTEP.LB_channels[ichan]
    fname_reference = PTEP.mock_radio_source_catalogue_name(chan_name)
    fname_IFCAPOL   = PTEP.cleaned_catalogue_name(isim,chan_name)

    if os.path.isfile(fname_IFCAPOL):

        ref_cat     = Table.read(fname_reference)
        IFCAPOL_cat = Table.read(fname_IFCAPOL)

        return {'reference':ref_cat,
                'IFCAPOL':IFCAPOL_cat}

    else:

        return False

def get_spurious_list(isim,ichan):
    """
    Returns a table containing the spurious (that is, source candidates that have
    no counterpart in the reference point source catalogue) sources for a given
    simulation number and channel index. If the IFCAPOL catalogue for such simulation
    and channel indexes does not exist, the routine returns **False**.

    Parameters
    ----------
    isim : int
        Simulation index (between 0 and 99).
    ichan : TYPE
        Channel index (between 0 and **nchans**-1).
    Returns
    -------
    Table or bool
        If the IFCAPOL catalogue of source candidates existed for the given isim and
        ichan, the routine returns a table with the positions and photometry of spurious
        sources. If it does not exist, it returns the boolean `False` .

    """

    rsult = get_catalogues(isim,ichan)

    if rsult is False:
        return False
    else:
        dicta = QA.catalogue_assessment(input_catalogue       = rsult['IFCAPOL'],
                                        reference_catalogue   = rsult['reference'],
                                        match_radius          = 30*u.arcmin,
                                        input_ra              = 'RA [deg]',
                                        input_dec             = 'DEC [deg]',
                                        ref_ra                = 'RA',
                                        ref_dec               = 'DEC',
                                        input_flux            = 'I [uK_CMB]',
                                        input_snr             = 'I SNR',
                                        ref_flux              = 'I',
                                        input_snrcut          = 4.0,
                                        ref_fluxcut           = 100.0,
                                        galcut_deg            = 5.0)

        sp               = dicta['spurious']
        sp['SIM_NUMBER'] = isim*np.ones(len(sp),dtype=int)

        return sp


def run_spurious():

    for ichan in range(nchans):

        tblist = []
        counta = 0

        for isim in range(nsims):

            spur = get_spurious_list(isim,ichan)
            if spur is not False:
                counta += 1
                tblist.append(spur)

        print('Channel {0} had {1} valid catalogues to be studied'.format(PTEP.LB_channels[ichan],counta))

        if counta>0:

            tabla = vstack(tblist)
            fname = survey.cat_out+'{0}_not_matched.fits'.format(PTEP.LB_channels[ichan])
            tabla.write(fname,overwrite=True)

run_spurious()