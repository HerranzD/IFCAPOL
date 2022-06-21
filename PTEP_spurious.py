#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:59:55 2022

@author: herranz
"""

import os

import numpy              as np
import PTEP_simulations   as PTEP
import survey_model       as survey
import astropy.units      as u

from astropy.table import Table,vstack


nchans = len(PTEP.LB_channels)
nsims  = 100

def catalogue_assessment(input_catalogue,
                         reference_catalogue,
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
                         galcut_deg            = 20.0):
    """


    Parameters
    ----------
    input_catalogue : `~astropy.Table`, optional
        The catalogue whose quality is going to be assessed.
        The default is ref_catalogue.
    reference_catalogue : `~astropy.Table`, optional
        The groundtruth catalogue to which the input catalogue is
        compared. The default is ref_catalogue.
    match_radius : `~astropy.Quantity`, optional
        The search radius used for cross-matching the input and the
        reference catalogues. The default is 30*u.arcmin.
    input_ra : str, optional
        The name of the column from `input_catalogue` containing the
        Right Ascension coordinate. The values inside this column should be
        given in degrees. The default is 'RA [deg]'.
    input_dec : str, optional
        The name of the column from `input_catalogue` containing the
        Declination coordinate. The values inside this column should be
        given in degrees. The default is 'DEC [deg]'.
    ref_ra : str, optional
        The name of the column from `reference_catalogue` containing the
        Right Ascension coordinate. The values inside this column should be
        given in degrees. The default is 'RA'.
    ref_dec : str, optional
        The name of the column from `reference_catalogue` containing the
        Declination coordinate. The values inside this column should be
        given in degrees. The default is 'DEC'.
    input_flux : str, optional
        The name of the column from `input_catalogue` containing the
        flux density of the sources. The default is 'I [uK_CMB]'.
    input_snr : str, optional
        The name of the column from `input_catalogue` containing the
        detection signal-to-noise ratio of the sources. The default is 'I SNR'.
    ref_flux : str, optional
        The name of the column from `reference_catalogue` containing the
        flux density of the sources. The default is'I'.
    input_snrcut : float, optional
        The signal-to-noise ratio threshold over which the `input_catalogue` is
        cut. The default is 4.0.
    ref_fluxcut : float, optional
        The flux density threshold in the `reference_catalogue` over which
        the comparison is done. The default is 100.0.
    galcut_deg : float, optional
        The Galactic band cut applied to both catalogues, in degrees.
        The default is 20.0.

    Returns
    -------
    dict
        A dictionary containing:
            - **'matched'**: a Table with the sources from the `input_catalogue` found in the `reference_catalogue`, after the different thresholds and Galactic cut have been applied.
            - **'spurious'**: a Table with the sources from the `input_catalogue` that are not in the `reference_catalogue`, after the different thresholds and Galactic cut have been applied.
            - **'missing'**: a Table with the sources from the `reference_catalogue` that have not  been detected in the `input_catalogue`, after the different thresholds and Galactic cut have been applied.
            - **'completeness'**: the completeness of the `input_catalogue`, computed as number of matched objects over number of sources in the reference catalogue (above the `ref_fluxcut` value).
            - **'purity'**: the purity of the `input_catalogue`, computed as 1 - number of spurious detections over total number of objects found in the `input_catalogue` (above the `ref_fluxcut` value).
            - **'unit conversion'**: for the kind of tests done on the Foregrounds JWG simulations, the factor that converts uK_CMB to Jy.

    """

    from catalogue_tools import cat1_not_in_cat2,cat_match
    from myutils         import table2skycoord

    # Cutting input catalogue by SNR
    cat1 = input_catalogue[input_catalogue[input_snr]>=input_snrcut].copy()
    cat1.sort(keys=input_snr,reverse=True)

    # Reference catalogue
    cat2 = reference_catalogue.copy()


    # Adding formatted coordinates
    if 'RA' not in cat1:
        cat1['RA']  = cat1[input_ra].copy()
        cat1['DEC'] = cat1[input_dec].copy()

    if 'RA' not in cat2:
        cat2['RA']  = cat2[ref_ra].copy()
        cat2['DEC'] = cat2[ref_dec].copy()

    # Galactic band cut
    # c1   = table2skycoord(cat1)
    # cat1 = cat1[np.abs(c1.galactic.b.deg)>=galcut_deg]
    # c2   = table2skycoord(cat2)
    # cat2 = cat2[np.abs(c2.galactic.b.deg)>=galcut_deg]

    # Spurious sources:
    spurious = cat1_not_in_cat2(cat1,cat2,match_radius)
    csp      = table2skycoord(spurious)
    spurious = spurious[np.abs(csp.galactic.b.deg)>=galcut_deg]

    # Missing sources:
    missing  = cat1_not_in_cat2(cat2,cat1,match_radius)
    cms      = table2skycoord(missing)
    missing  = missing[np.abs(cms.galactic.b.deg)>=galcut_deg]

    # Matched sources
    matched  = cat_match(cat2,cat1,match_radius)
    cmt      = table2skycoord(matched)
    matched  = matched[np.abs(cmt.galactic.b.deg)>=galcut_deg]

    # Completeness
    mthr     = np.count_nonzero(matched[ref_flux]>=ref_fluxcut)
    c2       = table2skycoord(cat2)
    cat2     = cat2[np.abs(c2.galactic.b.deg)>=galcut_deg]
    tthr     = np.count_nonzero(cat2[ref_flux]>=ref_fluxcut)
    try:
        compl = mthr/tthr
    except ZeroDivisionError:
        compl = np.nan

    # Purity
    sthr      = np.count_nonzero(spurious[input_flux]>=ref_fluxcut)
    try:
        purit = 1.0-sthr/(mthr+sthr)
    except ZeroDivisionError:
        purit = np.nan

    # Unit conversion
    if 'I [uK_CMB]' in cat1.colnames:
        if 'I [Jy]' in cat1.colnames:
            r         = cat1['I [Jy]']/cat1['I [uK_CMB]']
            unit_conv = r.mean()
        else:
            unit_conv = 1.0
    else:
        unit_conv = 1.0

    return {'matched':matched[matched[ref_flux]>=ref_fluxcut],
            'spurious':spurious[spurious[input_flux]>=ref_fluxcut],
            'missing':missing[missing[ref_flux]>=ref_fluxcut],
            'completeness':compl,
            'purity':purit,
            'unit conversion':unit_conv}


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

        # print('   Reference catalogue = ',fname_reference)
        # print('   IFCAPOL   catalogue = ',fname_IFCAPOL)

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
        dicta = catalogue_assessment(rsult['IFCAPOL'],
                                     rsult['reference'],
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