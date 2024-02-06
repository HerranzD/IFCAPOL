#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:58:01 2024

@author: herranz
"""

import os
import numpy               as     np
import healpy              as     hp
import astropy.units       as     u
import matplotlib.pyplot   as     plt

from run0_simulations      import input_radio_source_catalogue_name
from run0_simulations      import run0_name
from run0_simulations      import run0_foregrounds_name
from run0_simulations      import cleaned_catalogue_name

from astropy.table         import Table
from astropy.modeling      import models, fitting
from astropy.visualization import hist

# %%    CATALOGUE INPUT

def load_cleaned_catalogue(chan_name, sim_number,conversion_factor=1.e6):
    """
    Load the cleaned catalogue for a given channel and simulation number,
    and the input radio source catalogue for the corresponding channel.

    Parameters
    ----------
    chan_name : str
        Name of the channel.
    sim_number : int
        Number of the simulation.
    conversion_factor: float
        Factor needed to convert run0 simulations to uK_CMB units.
        Default is 1.e6

    Returns
    -------
    cleaned_catalogue : astropy.table.Table
        cleaned catalogue for the given channel and simulation number.
    radio_source_catalogue : astropy.table.Table
        Input radio source catalogue for the given channel.

    """

    cleaned_catalogue_path = cleaned_catalogue_name(sim_number,chan_name)
    fname                  = input_radio_source_catalogue_name(chan_name)

    if os.path.exists(cleaned_catalogue_path):
        cleaned_catalogue      = Table.read(cleaned_catalogue_path)
        radio_source_catalogue = Table.read(fname)
        radio_source_catalogue.sort('I', reverse=True)
    else:
        print('Warning: cleaned catalogue not found for channel',
              chan_name,
              'and simulation number', sim_number)
        cleaned_catalogue = None
        radio_source_catalogue = None

    if conversion_factor != 1.0:
        photometry_columns = ['I [uK_CMB]','I err [uK_CMB]',
                              'I [Jy]','I err [Jy]',
                              'P [uK_CMB]','P err [uK_CMB]',
                              'P [Jy]','P err [Jy]']
        for col in photometry_columns:
            cleaned_catalogue[col] = conversion_factor*cleaned_catalogue[col]

    return cleaned_catalogue, radio_source_catalogue




# %%    CATALOGUE ASSESSMENT

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
    Assesses the quality of a given catalogue by comparing it to a reference
    catalogue. The assessment is done by computing the completeness and
    purity of the input catalogue, and by identifying the matched, spurious
    and missing sources. The input and reference catalogues are cross-matched
    using a given search radius, and the sources are cut by a given signal-to-
    noise ratio and flux density thresholds. A Galactic band cut is also
    applied to both catalogues. The completeness is computed as the number of
    matched objects over the number of sources in the reference catalogue
    (above the `ref_fluxcut` value). The purity is computed as 1 minus the
    number of spurious detections over the total number of objects found in
    the input catalogue (above the `ref_fluxcut` value). The unit conversion
    factor from uK_CMB to Jy is also computed. The results are returned as a
    dictionary.

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
            - **'matched'**: a Table with the sources from the `input_catalogue`
                found in the `reference_catalogue`, after the different
                thresholds and Galactic cut have been applied.
            - **'spurious'**: a Table with the sources from the `input_catalogue`
                that are not in the `reference_catalogue`, after the different
                thresholds and Galactic cut have been applied.
            - **'missing'**: a Table with the sources from the
                `reference_catalogue` that have not  been cleaned in the
                `input_catalogue`, after the different thresholds and Galactic
                cut have been applied.
            - **'completeness'**: the completeness of the `input_catalogue`,
                computed as number of matched objects over number of
                sources in the reference catalogue
                (above the `ref_fluxcut` value).
            - **'purity'**: the purity of the `input_catalogue`, computed
                as 1 - number of spurious detections over total number of
                objects found in the `input_catalogue`
                (above the `ref_fluxcut` value).
            - **'unit conversion'**: for the kind of tests done on
                the Foregrounds JWG simulations,
                the factor that converts uK_CMB to Jy.

    """

    from catalogue_tools import cat1_not_in_cat2,cat_match
    from myutils         import table2skycoord

    # Cutting input catalogue by SNR
    cat1 = input_catalogue[input_catalogue[input_snr]>=input_snrcut].copy()
    cat1.sort(keys=input_snr,reverse=True)

    # Reference catalogue
    cat2 = reference_catalogue.copy()


    # Adding formatted coordinates
    if 'RA' not in cat1.colnames:
        cat1['RA']  = cat1[input_ra].copy()
        cat1['DEC'] = cat1[input_dec].copy()

    if 'RA' not in cat2.colnames:
        cat2['RA']  = cat2[ref_ra].copy()
        cat2['DEC'] = cat2[ref_dec].copy()

    # Galactic band cut
    c1   = table2skycoord(cat1)
    cat1 = cat1[np.abs(c1.galactic.b.deg)>=galcut_deg]
    c2   = table2skycoord(cat2)
    cat2 = cat2[np.abs(c2.galactic.b.deg)>=galcut_deg]

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

    thematch = matched[matched[ref_flux]>=ref_fluxcut]
    thematch = reformat_matched_catalogue(thematch,unit_conv)

    return {'matched':thematch,
            'spurious':spurious[spurious[input_flux]>=ref_fluxcut],
            'missing':missing[missing[ref_flux]>=ref_fluxcut],
            'completeness':compl,
            'purity':purit,
            'unit conversion':unit_conv}


def reformat_matched_catalogue(matched,unit_conversion):
    """
    Reformats the matched catalogue, converting the input and detection
    photometry from uK_CMB to Jy, and renaming the columns accordingly.
    The input and detection photometry are also converted to polarization
    fraction and angle, and the Galactic and Equatorial coordinates are
    renamed accordingly. The output is a Table with the reformatted
    columns.

    Parameters
    ----------
    matched : `~astropy.Table`
        The matched catalogue.
    unit_conversion : float
        The factor that converts uK_CMB to Jy.

    Returns
    -------
    `~astropy.Table`
        The reformatted matched catalogue.

    """

    for c in ['I','Q','U']:
        matched.rename_column(c,'input '+c+' [uK_CMB]')

    qq = matched['input Q [uK_CMB]']
    uu = matched['input U [uK_CMB]']
    p  = np.sqrt(qq*qq+uu*uu)

    matched['input P [uK_CMB]'] = p

    for c in ['I','Q','U','P']:
        old = 'input '+c+' [uK_CMB]'
        new = 'input '+c+' [Jy]'
        matched[new] = unit_conversion*matched[old]

    matched.rename_column('RA_1','RA')
    matched.rename_column('DEC_1','DEC')
    matched.rename_column('RA_2','detection RA')
    matched.rename_column('DEC_2','detection DEC')

    nuisance_columns = ['RA [deg]','DEC [deg]','GLON [deg]','GLAT [deg]']
    matched.remove_columns(nuisance_columns)

    output_order = ['ID',
                    'Ipix',
                    'RA',
                    'DEC',
                    'GLON',
                    'GLAT',
                    'input I [uK_CMB]',
                    'input Q [uK_CMB]',
                    'input U [uK_CMB]',
                    'input P [uK_CMB]',
                    'input I [Jy]',
                    'input Q [Jy]',
                    'input U [Jy]',
                    'input P [Jy]',
                    'detection RA',
                    'detection DEC',
                    'I [uK_CMB]',
                    'I err [uK_CMB]',
                    'I [Jy]',
                    'I err [Jy]',
                    'I SNR',
                    'P [uK_CMB]',
                    'P err [uK_CMB]',
                    'P [Jy]',
                    'P err [Jy]',
                    'Angle [deg]',
                    'Angle err [deg]',
                    'P significance',
                    'Polarization fraction [%]',
                    'Polarization fraction error [%]',
                    'Extended flag',
                    'Photometry flag',
                    'Separation from centre [deg]',
                    'Separation']

    return matched[output_order]

# %%    PLOTTING

def plot_catalogue(reference_map,
                   catalogue,
                   snr_cut = 5.0,
                   xsize   = 2400,
                   color   = 'red',
                   title   = None,
                   tofile  = None):
    """
    Plots the given catalogue over a given reference map.

    Parameters
    ----------
    reference_map : array
        The reference map over which the catalogue is going to be plotted.
    catalogue : astropy.table.Table
        The catalogue to be plotted.
    snr_cut : float, optional
        The signal-to-noise ratio threshold over which the `catalogue` is cut.
        The default is 5.0.
    xsize : int, optional
        The size of the plot. The default is 2400.
    color : str, optional
        The color of the markers. The default is 'red'.
    title : str, optional
        The title of the plot. The default is None.
    tofile : str, optional
        The name of the file where the plot is going to be saved.
        The default is None.

    Returns
    -------
    None.

    """

    hp.mollview(reference_map,norm='hist',cbar=False,flip='astro',xsize=xsize)
    plt.title(' ')

    cut_catalogue = catalogue[catalogue['I SNR']>=snr_cut]

    x = cut_catalogue['GLON [deg]']
    y = cut_catalogue['GLAT [deg]']

    hp.projscatter(x,y,lonlat=True,coord='G',c=color,marker='o',alpha=0.5)
    hp.graticule()

    if title is not None:
        plt.title(title)

def compare_fluxes(matched_catalog,
                   quantity = 'I',
                   units    = 'Jy',
                   title    = None,
                   loglog   = True,
                   tofile   = None):
    """
    Compares the input and estimated photometry for a given quantity
    (intensity, polarization fraction or angle) in a given matched catalogue.
    The comparison is done by plotting the input photometry against the
    estimated photometry, and fitting a linear model to the data. The
    results are returned as a Table with the best fit parameters.

    Parameters
    ----------
    matched_catalog : `~astropy.Table`
        The matched catalogue.
    quantity : str, optional
        The quantity to be compared. The default is 'I'.
    units : str, optional
        The units of the quantity to be compared. The default is 'Jy'.
    title : str, optional
        The title of the plot. The default is None.
    loglog : bool, optional
        Whether to plot the data in log-log scale. The default is True.
    tofile : str, optional
        The name of the file where the plot is going to be saved.
        The default is None.

    Returns
    -------
    `~astropy.modeling.fitting.LinearLSQFitter`
        The best fit parameters.

    """


    x = matched_catalog['input {0} [{1}]'.format(quantity,units)]
    y = matched_catalog['{0} [{1}]'.format(quantity,units)]
    s = matched_catalog['{0} err [{1}]'.format(quantity,units)]

    plt.figure()
    plt.errorbar(x,y,yerr=s,capsize=2,fmt='o')
    if loglog:
        plt.loglog()
    plt.xlabel('input {0} [{1}]'.format(quantity,units))
    plt.ylabel('estimated {0} [{1}]'.format(quantity,units))
    if title is not None:
        plt.title(title)

    # initialize a linear fitter
    fit = fitting.LinearLSQFitter()

    # initialize a linear model
    line_init = models.Linear1D()

    # fit the data with the fitter
    fitted_line = fit(line_init, x, y, weights=1.0/s)

    xx = np.linspace(0.5*x.min(),1.1*x.max(),1000)
    plt.plot(xx,xx,label='y=x')
    plt.plot(xx,fitted_line(xx),label='best fit')
    plt.legend()

    if tofile is not None:
        plt.savefig(tofile)

    return fitted_line


def plot_intensity_histograms(matched,spurious,missing,conversion_factor,
                              loglog   = True,
                              title    = None,
                              tofile   = None):

    """
    Plots the histograms of the input and estimated photometry in Jansky in
    the matched, spurious and missing sources. The histograms are plotted using
    astropy.visualization.hist with knuth bins. The histograms are normalized
    to the total number of sources in each category. The histograms have
    different colors and a degree of transparency so the overlapping areas
    are clearly visualized. The figure is saved to a file if tofile is not None.

    Parameters
    ----------
    matched : `~astropy.Table`
        The matched catalogue.
    spurious : `~astropy.Table`
        The spurious catalogue.
    missing : `~astropy.Table`
        The missing catalogue.
    conversion_factor : float
        The factor that converts uK_CMB to Jy.
    title : str, optional
        The title of the plot. The default is None.
    tofile : str, optional
        The name of the file where the plot is going to be saved.
        The default is None.

    Returns
    -------
    None.

    """

    fig, ax = plt.subplots()

    flux_matched  = matched['I [Jy]'].data
    flux_spurious = spurious['I [Jy]'].data
    flux_missing  = missing['I'].data*conversion_factor

    # if any of the three previous arrays contain less than 3 elements, the
    # array is duplicated to avoid a ValueError in the hist function

    if flux_matched.size < 3:
        flux_matched = np.concatenate([flux_matched,flux_matched])
    if flux_spurious.size < 3:
        flux_spurious = np.concatenate([flux_spurious,flux_spurious])
    if flux_missing.size < 3:
        flux_missing = np.concatenate([flux_missing,flux_missing])

    if flux_matched.size > 2:
        hist(flux_matched,
             bins='knuth',
             histtype='stepfilled',
             alpha=0.5,
             density=True,
             label='matched')

    if flux_spurious.size > 2:
        hist(flux_spurious,
             bins='knuth',
             histtype='stepfilled',
             alpha=0.5,
             density=True,
             label='spurious')

    if flux_missing.size > 2:
        hist(flux_missing,
             bins='knuth',
             histtype='stepfilled',
             alpha=0.5,
             density=True,
             label='missing')

    plt.xlabel('I [Jy]')
    plt.ylabel('Number density of sources')
    plt.legend()

    if loglog:
        plt.loglog()

    if title is not None:
        plt.title(title)

    if tofile is not None:
        plt.savefig(tofile)


def plot_matched_spurious_missing(reference_map,
                                  matched,
                                  spurious,
                                  missing,
                                  tofile = None):
    """
    Plots the matched, spurious and missing sources over a given reference map.
    The sources are plotted following the steps in the plot_catalogue function,
    with different colors for each category and a legend. The figure is saved
    to a file if tofile is not None.

    Parameters
    ----------
    reference_map : Fitsmap
        The reference map over which the sources are going to be plotted.
    matched : `~astropy.Table`
        The matched catalogue.
    spurious : `~astropy.Table`
        The spurious catalogue.
    missing : `~astropy.Table`
        The missing catalogue.
    tofile : str, optional
        The name of the file where the plot is going to be saved.
        The default is None.

    Returns
    -------
    None.

    """

    hp.mollview(reference_map.data,norm='hist',cbar=False,flip='astro',xsize=2400)
    plt.title(' ')

    x = spurious['GLON [deg]']
    y = spurious['GLAT [deg]']
    hp.projscatter(x,y,lonlat=True,coord='G',c='yellow',marker='o',alpha=0.5)

    x = missing['GLON']
    y = missing['GLAT']
    hp.projscatter(x,y,lonlat=True,coord='G',c='blue',marker='o',alpha=0.5)

    x = matched['GLON']
    y = matched['GLAT']
    hp.projscatter(x,y,lonlat=True,coord='G',c='red',marker='o',alpha=1)

    hp.graticule()
    plt.legend(['spurious','missing','matched'])

    if tofile is not None:
        plt.savefig(tofile)

