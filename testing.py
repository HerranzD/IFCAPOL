#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:45:31 2022

@author: herranz
"""

import numpy             as np
import survey_model      as survey
import healpy            as hp
import astropy.units     as u
import matplotlib.pyplot as plt
import os

from astropy.table  import Table
from myutils        import sigma2fwhm,table2skycoord

chan_name    = 'LB_LFT_40'
#chan_name    = 'LB_HFT_337'

coadded_file = survey.data_dir+chan_name+'_coadd_signal_map_0000_PTEP_20200915_compsep.fits'
noise_file   = survey.data_dir+chan_name+'_noise_FULL_0000_PTEP_20200915_compsep.fits'
ps_file      = survey.data_dir+'radio_sources_'+chan_name[3:]+'_uKcmb_nside512.fits'



test_plotf   = '/Users/herranz/Desktop/testplot.jpg'

testdir      = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/Tests/'

# %% --- COMPONENT MAPS

signal  = survey.load_LiteBIRD_map(coadded_file,chan_name=chan_name)
noise   = survey.load_LiteBIRD_map(noise_file,chan_name=chan_name)
radiops = survey.load_LiteBIRD_map(ps_file,chan_name=chan_name)

# %% --- TOTAL MAP

totmap_file = survey.data_dir+chan_name+'_testing_map_0000_PTEP_20200915_compsep.fits'

if os.path.isfile(totmap_file):
    total = survey.load_LiteBIRD_map(totmap_file,chan_name=chan_name)
else:
    total   = signal+noise+radiops
    total.write(totmap_file)

# %% --- POINT SOURCE MOCK CATALOGUE FOR TESTING

mock_catalogue_fname = survey.data_dir+'mock_ps_catalogue_'+chan_name[3:]+'_uKcmb_nside512.fits'

def create_mock_point_source_catalogue():

    sources = radiops[0].data
    minmax  = hp.hotspots(sources)
    peaks   = {'Ipix':minmax[2],
               'RA':radiops.pixel_to_coordinates(minmax[2]).icrs.ra,
               'DEC':radiops.pixel_to_coordinates(minmax[2]).icrs.dec,
               'GLON':radiops.pixel_to_coordinates(minmax[2]).galactic.l,
               'GLAT':radiops.pixel_to_coordinates(minmax[2]).galactic.b,
               'I':radiops[0].data[minmax[2]],
               'Q':radiops[1].data[minmax[2]],
               'U':radiops[2].data[minmax[2]]}
    valleys = {'Ipix':minmax[1],
               'RA':radiops.pixel_to_coordinates(minmax[1]).icrs.ra,
               'DEC':radiops.pixel_to_coordinates(minmax[1]).icrs.dec,
               'GLON':radiops.pixel_to_coordinates(minmax[1]).galactic.l,
               'GLAT':radiops.pixel_to_coordinates(minmax[1]).galactic.b,
               'I':radiops[0].data[minmax[1]],
               'Q':radiops[1].data[minmax[1]],
               'U':radiops[2].data[minmax[1]]}

    peaks   = Table(peaks)
    valleys = Table(valleys)

    peaks.write(mock_catalogue_fname,
                overwrite=True)

    return peaks,valleys

if os.path.isfile(mock_catalogue_fname):
    peaks = Table.read(mock_catalogue_fname)
else:
    peaks,valleys = create_mock_point_source_catalogue()

peaks.sort(keys='I',reverse=True)
epeaks = peaks[np.abs(peaks['GLAT'])>20]

def test_fwhms(ntest=100):

    f = []
    for i in range(ntest):
        c     = radiops.pixel_to_coordinates(epeaks['Ipix'][i])
        patch = radiops[0].patch(c)
        g     = patch.central_gaussfit(return_output=True,verbose=False)
        f.append(g.sigma*sigma2fwhm)
    f = np.array(f)*patch.pixsize

    return f

# %% --- PHOTOMETRY TESTING

def test_photometry(nsources,outfile):

    import IFCAPOL as pol

    lista = []

    for i in range(nsources):

        dic        = {}
        dic['RA']  = epeaks['RA'][i]
        dic['DEC'] = epeaks['DEC'][i]
        dic['I0']  = epeaks['I'][i]
        dic['Q0']  = epeaks['Q'][i]
        dic['U0']  = epeaks['U'][i]
        dic['P0']  = np.sqrt(epeaks['Q'][i]**2+epeaks['U'][i]**2)
        dic['A0']  = pol.pol_angle(epeaks['Q'][i], epeaks['U'][i])

        coord      = total.pixel_to_coordinates(epeaks['Ipix'][i])
        source     = pol.Source.from_coordinate(total, coord)

        dic['I']   = source.I.value
        dic['Q']   = source.Q.value
        dic['U']   = source.U.value
        dic['P']   = source.P.value
        dic['A']   = source.angle.value

        dic['Ie']  = source.I.error
        dic['Qe']  = source.Q.error
        dic['Ue']  = source.U.error
        dic['Pe']  = source.P.error
        dic['Ae']  = source.angle.error

        lista.append(dic)

    tabla = Table(lista)
    tabla.write(testdir+outfile,overwrite=True)

def study_test(cantidad,snrcut=5):

    import matplotlib.pyplot as plt
    from astropy.modeling import models, fitting
    from astropy.stats import sigma_clip

    test_nopix = Table.read(testdir+'test_fotometria_nopixel_{0}.fits'.format(chan_name))
    test_sipix = Table.read(testdir+'test_fotometria_sipixel_{0}.fits'.format(chan_name))

    test_sipix['I SNR'] = test_sipix['I']/test_sipix['Ie']
    mask = test_sipix['I SNR'] >= snrcut
    print(' ')
    print(' {0} sources with SNR >= {1}'.format(np.count_nonzero(mask),snrcut))

    x  = test_nopix['{0}0'.format(cantidad)][mask]
    y  = test_nopix['{0}'.format(cantidad)][mask]
    s  = test_nopix['{0}e'.format(cantidad)][mask]

    fit = fitting.LinearLSQFitter()
    sfit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)
    line_init = models.Linear1D(fixed={'intercept':True})
    fitted_line,m = sfit(line_init, x, y, weights=1.0/s)

    plt.figure(figsize=(8,10))
    plt.errorbar(x, y, yerr=s, fmt='ko', label='Data, no pixel window')
    plt.plot(x, fitted_line(x), 'k-', label='Fitted Model, no pixel window')


    xs = test_sipix['{0}0'.format(cantidad)][mask]
    ys = test_sipix['{0}'.format(cantidad)][mask]
    ss = test_sipix['{0}e'.format(cantidad)][mask]

    fit          = fitting.LinearLSQFitter()
    sfits        = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)
    line_inits   = models.Linear1D(fixed={'intercept':True})
    fitted_lines,sm = sfits(line_inits, xs, ys, weights=1.0/ss)

    plt.errorbar(xs, ys, yerr=ss, fmt='bo', label='Data, with pixel window')
    plt.plot(xs, fitted_lines(x), 'b-', label='Fitted Model, with pixel window')

    plt.xlabel('Input {0}'.format(cantidad))
    plt.ylabel('Estimated {0}'.format(cantidad))
    plt.legend()
    if cantidad == 'I':
        plt.loglog()
    if cantidad == 'P':
        plt.loglog()

    return fitted_line,fitted_lines

# %% --- PATCHING SCHEME

def nparches():
    """
    Calculates how many patches (in the default LiteBIRD configuration)
    are needed to cover at least once the whole sky.

    Returns
    -------
    N : int
        The minimum number of patches that would cover the sphere, if the
        patching was 100% efficient.

    """

    import IFCAPOL as pol
    import astropy.units as u

    sky = 4*np.pi*u.sr
    s   = pol.Source.from_coordinate(total,
                                 total.pixel_to_coordinates(epeaks['Ipix'][0]))

    parea = (s.diccio['Patch I'].size[0]*s.diccio['Patch I'].pixsize)**2

    N = (sky/parea).si.value

    return N

def study_coverage(nside):
    """
    Write a hits map for a given nside basis of sky patching

    Parameters
    ----------
    nside : int
        The NSIDE parameter of the patching scheme.

    Returns
    -------
    mapa : Fitsmap
        Hits map in the default LiteBIRD nside

    """

    from fits_maps import Fitsmap

    mapa = total[0].copy()
    vac  = Fitsmap.empty(nside)

    lpix = []

    for ic in range(vac.npix):
        coord = vac.pixel_to_coordinates(ic)
        patch = mapa.patch(coord)
        size  = patch.size
        i,j   = np.meshgrid(np.arange(size[0]),np.arange(size[1]))
        pcoo  = patch.pixel_coordinate(i,j)
        ipix  = mapa.coordinates_to_pixel(pcoo)
        lpix  = lpix + list(ipix.flatten())

    listap = np.array(lpix)
    cuenta = np.zeros(mapa.npix)
    for pixel in range(mapa.npix):
        cuenta[pixel] = np.count_nonzero(listap==pixel)

    mapa.data = cuenta

    mapa.write(testdir+'counts_nside{0}.fits'.format(nside))

    return mapa

# %% --- NON-BLIND AND BLIND CATALOGUES:

fname_nonblind = testdir+chan_name+'_full_catalogue.fits'

def make_catalogue_test():

    from IFCAPOL_catalogue   import blind_survey,non_blind_survey
    from astropy.coordinates import SkyCoord
    import IFCAPOL           as     pol
    import astropy.units     as     u

    s         = pol.Source.from_coordinate(total, SkyCoord(0,0,frame='icrs',unit=u.deg))
    fwhm      = s.fwhm

    blind     = blind_survey(total[0], fwhm, fname_nonblind, verbose=True)

    print(' ')

    nonblinda = non_blind_survey(total, fname_nonblind,
                                clean_mode = 'after',
                                verbose=True)

    nonblindb = non_blind_survey(total, fname_nonblind,
                                clean_mode = 'before',
                                verbose=True)


    return blind,nonblindb,nonblinda

fname_testing     = '/Users/herranz/Dropbox/Trabajo/LiteBird/'
fname_testing    += 'Source_Extractor/Catalogs/Output/'
fname_testing    += '40GHz_output_catalogue_IFCAPOL.fits'
testing_catalogue = Table.read(fname_testing)

from quality_assessment import catalogue_assessment as catalogue_assesment

def plot_overlay_catalogue(catalogue,mapa,title=''):

    plt.figure()
    mapa.moll(norm='hist',cbar=False,flip='astro')
    c = table2skycoord(catalogue)
    x = c.galactic.l.deg
    y = c.galactic.b.deg
    hp.projscatter(x, y, lonlat=True, coord='G',color='r',marker='o')
    plt.title(title)

def plots_quality():

    S0   = np.linspace(10,1000,100)
    comp = []
    pur  = []
    flux = []

    for S in S0:
        d = catalogue_assesment(ref_fluxcut=S)
        flux.append(S*d['unit conversion'])
        comp.append(d['completeness'])
        pur.append(d['purity'])

    plt.figure()
    plt.plot(flux,comp,label='Completeness')
    plt.plot(flux,pur,label='Purity')
    plt.xlabel('Flux density [Jy]')
    plt.legend()

    return flux,comp,pur

def flux_comparison(dicc):

    from astropy.modeling import models, fitting
    from astropy.stats import sigma_clip

    x  = dicc['matched']['I']*dicc['unit conversion']
    y  = dicc['matched']['I [uK_CMB]']*dicc['unit conversion']
    sy = dicc['matched']['I err [uK_CMB]']*dicc['unit conversion']

    plt.figure()
    plt.errorbar(x, y, yerr=sy, fmt='o')

    fit         = fitting.LinearLSQFitter()
    or_fit      = fitting.FittingWithOutlierRemoval(fit,
                                                    sigma_clip,
                                                    niter=3,
                                                    sigma=3.0)

    line_init   = models.Linear1D()
    line_initf   = models.Linear1D(intercept=0,fixed={'intercept':True})
    fitted_line = fit(line_init, x, y, weights=1.0/sy)
    fitted_line2, mask = or_fit(line_init, x, y, weights=1.0/sy)
    fitted_line3 = fit(line_initf, x, y, weights=1.0/sy)
    line_orig   = models.Linear1D(slope=1.0, intercept=0.0)

    plt.plot(x, line_orig(x), 'b-', label='y=x')
    plt.plot(x, fitted_line(x), 'k-', label='Fitted Model')
    plt.plot(x, fitted_line2(x), 'r:', label='Fitted Model, sigma clipping')
    plt.plot(x, fitted_line3(x), 'g-.', label='Fitted Model, no intercept')

    plt.loglog()
    plt.xlabel('Flux density [Jy]')
    plt.ylabel('Estimated flux density [Jy]')
    plt.legend()

    return fitted_line,fitted_line2,fitted_line3

# %% --- FULL RUN OF THE CATALOGUES OVER THE 100 SIMULATIONS AT 40 GHz:

def run_blinds_40GHz_simulations(starts=0,ends=100):

    from IFCAPOL_catalogue   import blind_survey,non_blind_survey
    from astropy.coordinates import SkyCoord
    import IFCAPOL           as     pol
    import astropy.units     as     u
    import gc

    s         = pol.Source.from_coordinate(total, SkyCoord(0,0,frame='icrs',unit=u.deg))
    fwhm      = s.fwhm

    subdirs   = [f'{i:04d}' for i in range(100)]
    localdir  = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/Data/'
    catsdir   = '/Users/herranz/Dropbox/Trabajo/LiteBird/Source_Extractor/Catalogs/Output/'
    keyfile   = 'LB_LFT_40'
    total_dir = localdir+'total_sims/'

    for isubd in range(starts,ends):

        subdir = subdirs[isubd]

        total_subdir = total_dir+subdir+'/'
        total_fname  = total_subdir+keyfile+'_{0}_'.format(subdir)+'full_sim.fits'

        catal_subdir = catsdir+subdir+'/'
        if not os.path.isdir(catal_subdir):
            os.mkdir(catal_subdir)
        catal_fname  = catal_subdir+keyfile+'_{0}_'.format(subdir)+'catalogue.fits'

        print(' Extracting catalogue from simulation number {0}...'.format(subdir))

        simulation = survey.load_LiteBIRD_map(total_fname,chan_name=keyfile)

        print('     Extracting blind catalogue...')
        blind      = blind_survey(simulation[0],
                                  fwhm,
                                  catal_fname,
                                  verbose=False)
        print('         {0} targets in the blind catalogue.'.format(len(blind)))


        print('     Obtaining non-blind catalogue...')
        nonblind   = non_blind_survey(simulation,
                                      catal_fname,
                                      clean_mode = 'after',
                                      verbose=False)
        print('         {0} sources in the non-blind catalogue.'.format(len(nonblind)))

        print(' ')

        gc.collect()


def do_it_old_chum():
    for i in range(12):
        imin = 51+i*4
        imax = 51+(i+1)*4
        print(imin,imax)
        run_blinds_40GHz_simulations(starts=imin,ends=imax)
    run_blinds_40GHz_simulations(starts=99,ends=100)
