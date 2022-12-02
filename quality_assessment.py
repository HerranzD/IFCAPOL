#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:46:01 2022

@author: herranz
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units     as u
import survey_model      as survey

from astropy.table import Table


chan_name  = 'LB_LFT_40'  # The channel over which we have done the testing
LB_dir     = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/'
cat_dir    = LB_dir+'Catalogs/Output/'  # where we placed the catalogues
                                        # extracted from the JWG simulations

subdirs    = [f'{i:04d}' for i in range(100)]

# %% --- CATALOGUE OF REFERENCE

ref_catalogue_fname  = survey.data_dir+'mock_ps_catalogue_'
ref_catalogue_fname += chan_name[3:]+'_uKcmb_nside512.fits'
ref_catalogue        = Table.read(ref_catalogue_fname)

ref_catalogue.sort(keys='I',reverse=True)

# %% --- BASIC CATALOGUE ASSESSMENT

def catalogue_assessment(input_catalogue       = ref_catalogue,
                         reference_catalogue   = ref_catalogue,
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
    if 'RA' not in cat1.colnames:
        cat1['RA']  = cat1[input_ra].copy()
        cat1['DEC'] = cat1[input_dec].copy()

    if 'RA' not in cat2.colnames:
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

# %% --- CHECK A SIMULATION:

def plot_overlay_catalogue(catalogue,mapa,title=''):

    from myutils import table2skycoord
    import healpy as hp

    plt.figure()
    mapa.moll(norm='hist',cbar=False,flip='astro')
    c = table2skycoord(catalogue)
    x = c.galactic.l.deg
    y = c.galactic.b.deg
    hp.projscatter(x, y, lonlat=True, coord='G',color='r',marker='o')
    plt.title(title)

def check_simulation(nsim):

    from myutils import table2skycoord
    from testing import radiops
    import IFCAPOL as pol

    simstr = '{0:04d}'.format(nsim)

    fname_cat    = cat_dir+simstr+'/'+chan_name
    fname_cat   += '_{0}_catalogue_after_IFCAPOL.fits'.format(simstr)
    catalogue    = Table.read(fname_cat)

    localdir     = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/Data/'
    total_dir    = localdir+'total_sims/'
    coadd_dir    = localdir+'coadd_signal_maps/'
    total_subdir = total_dir+simstr+'/'
    coadd_subdir = coadd_dir+simstr+'/'
    total_fname  = total_subdir+chan_name+'_{0}_'.format(simstr)+'full_sim.fits'
    coadd_fname  = coadd_subdir+chan_name+'_coadd_signal_map_{0}_PTEP_20200915_compsep.fits'.format(simstr)
    maps         = survey.load_LiteBIRD_map(total_fname,chan_name=chan_name)
    comaps       = survey.load_LiteBIRD_map(coadd_fname,chan_name=chan_name)

    d            = catalogue_assessment(input_catalogue = catalogue,
                                        match_radius    = 1*u.deg,
                                        galcut_deg      = 40.0)

    mapz = maps.copy()
    plt.close('all')

    plt.figure();
    plot_overlay_catalogue(d['missing'], mapz[0]);
    plt.title('Missing sources');

    plt.figure();
    plot_overlay_catalogue(d['spurious'], mapz[0]);
    plt.title('Spurious sources');

    lostf    = []
    lostreal = []
    coor     = table2skycoord(d['missing'])
    for c in coor:
        lostf.append(pol.Source.from_coordinate(maps, c))
        lostreal.append(pol.Source.from_coordinate(radiops, c))

    espf     = []
    espfreal = []
    espcoadd = []
    coor     = table2skycoord(d['spurious'])
    for c in coor:
        espf.append(pol.Source.from_coordinate(maps, c))
        espfreal.append(pol.Source.from_coordinate(radiops, c))
        espcoadd.append(pol.Source.from_coordinate(comaps,c))

    return maps,d,lostf,lostreal,espf,espfreal,espcoadd



# %% --- COMPLETENESS/PURITY TABLES:

def completeness_purity(input_catalogue,
                        reference_catalogue,
                        smin   = 10,
                        smax   = 1000,
                        rmatch = 30*u.arcmin,
                        galcut = 20.0,
                        toplot = False):
    """
    Returns a table with the completeness and purity of a catalogue, with respect
    to another reference catalogue, as a function of the flux density.

    Parameters
    ----------
    input_catalogue : `~astropy.Table`
        The catalogue from which completeness and purity are to be assessed.
    reference_catalogue : `~astropy.Table`
        The reference (groundtruth) catalogue.
    smin : float, optional
        The minimum flux density (n uK_CMB units) for the table. The default is 10.
    smax : float, optional
        The maximum flux density (n uK_CMB units) for the table. The default is 1000.
    rmatch : `~astropy.Quantity`, optional
        The search radius used for cross-matching the input and the
        reference catalogues. The default is 30*u.arcmin.
    toplot : bool, optional
        If True, the purity and completeness of the `input_catalogue` as
        a function of the flux density (in janskys) is plotted. The default is False.

    Returns
    -------
    dict
        A dictionary containing:
            - **'flux'**: flux density, in Jy.
            - **'completeness'**: completeness for that flux density.
            - **'purity'**: purity for that flux density.

    """


    S0   = np.linspace(smin,smax,100)
    tout = {}

    tout['completeness'] = []
    tout['purity']       = []
    tout['flux']         = []

    for S in S0:
        d = catalogue_assessment(input_catalogue     = input_catalogue,
                                 reference_catalogue = reference_catalogue,
                                 match_radius        = rmatch,
                                 ref_fluxcut         = S)

        tout['flux'].append(S*d['unit conversion'])
        tout['completeness'].append(d['completeness'])
        tout['purity'].append(d['purity'])

    if toplot:
        plt.figure()
        plt.plot(tout['flux'],tout['completeness'],label='Completeness')
        plt.plot(tout['flux'],tout['purity'],label='Purity')
        plt.xlabel('Flux density [Jy]')
        plt.legend()

    return Table(tout)

def make_tables(galcut=20,rmatch=1.0*u.deg):

    tables = []

    for d in subdirs:

        print(' Test '+d)

        fname  = cat_dir+d+'/'+chan_name
        fname += '_{0}_catalogue_after_IFCAPOL.fits'.format(d)

        try:
            input_catalogue = Table.read(fname)
            tables.append(completeness_purity(input_catalogue,
                                              ref_catalogue,
                                              rmatch=rmatch,
                                              galcut=galcut))
        except FileNotFoundError:
            pass

    fname = cat_dir+'QA.pkl'

    file = open(fname,'wb')
    pickle.dump(tables, file)
    file.close()

    return tables

def process_table():

    fname = cat_dir+'QA.pkl'
    file  = open(fname,'rb')
    tabs  = pickle.load(file)
    file.close()

    L = len(tabs)

    F = np.asarray([tabs[i]['flux'] for i in range(L)])
    C = np.asarray([tabs[i]['completeness'] for i in range(L)])
    P = np.asarray([tabs[i]['purity'] for i in range(L)])

    flux  = F.mean(axis=0)
    comp  = C.mean(axis=0)
    comps = C.std(axis=0)
    puri  = P.mean(axis=0)
    puris = P.std(axis=0)

    plt.figure(figsize=(8,8))
    plt.errorbar(flux, comp, yerr=comps,label='completeness',capsize=2)
    plt.errorbar(flux, puri, yerr=puris,label='purity',capsize=2)
    plt.xlabel('Flux density [Jy]')
    plt.legend()

    plt.grid()






