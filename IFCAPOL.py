#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:18:15 2021

    IFCAPOL code for detection/estimation of polarized compact sources

    Version 2.0 for LiteBIRD specifications

@author: herranz
"""

import os
import pickle
import numpy                    as     np
import matplotlib.pyplot        as     plt
import astropy.units            as     u
import healpy                   as     hp

from astropy.coordinates        import SkyCoord
from myutils                    import coord2healpix
from myutils                    import fwhm2sigma,sigma2fwhm
from sky_images                 import Imagen
from gauss2dfit                 import fit_single_peak
from unit_conversions           import parse_unit,convert_factor
from astropy.table              import QTable
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling           import fitting

# %%  Inherited experiment characteristics (liteBIRD)

import survey_model       as     survey

# %%  Ignore warnings:

import warnings
warnings.filterwarnings("ignore")

# %%  IFCAPOL default parameters:

stokes           = ['I','Q','U']   # Names of the Stokes parameters
use_ideal_beams  = True            # Use or not ideal Gaussian beam PSF
use_pixel_window = False           # Add or not the pixel effective beam to
                                   #    the beam FWHM for filtering effects
signif_border    = 10              # Number of border pixels to exclude
                                   #    for the calculation of P significance
signif_fraction  = 0.025           # Graction of britht pixels to mask
                                   #    for the calculation of P significance
iterative_MF     = True            # Iterative matched filtering
image_resampling = 1               # Resampling factor when projecting patches

# %%  SKY PATCHING

def define_npix(nside):
    return int(64*hp.nside2resol(256)/hp.nside2resol(nside))

def get_patches(sky_map,coord):

    """
    Gnomic projection from HEALPix maps to square images around a coordinate

    Parameters
    ----------
    sky_map : Fitsmap object (see the Fitsmap class in fits_maps.py)
        The input LiteBIRD image, as a Fitsmap object.

    coord : astropy.SkyCoord
        Coordinate around which the sky is being projected.

    Returns
    -------
    A dictionary containing:

        - Three flat sky patches, one for each Stokes
            parameters. By default, each patch is a Imagen object (see
            the Imagen class in sky_images.py) containing a flat patch
            14.658 x 14.658 degrees wide. The pixel area is equal to
            the parent HEALPix map pixel area, unless the image_resampling
            parameter is larger than one (in that case the pixel area is
            rescaled accordingly).
        - The FWHM of the patch beam. This FWHM is the same as the HEALPix
            parent map, unless the 'use_pixel_window' parameter is set to
            True. In that case, the HEALPix pixel effective window is
            quadratically added to the nominal FWHM
        - The central frequency of observation. It is the same as the parent
            HEALPIX map.
        - The physical units of the patches. It is the same as the parent
            HEALPIX map.

    """

    if use_pixel_window:
        fwhm    = np.sqrt(sky_map.fwhm[0]**2+sky_map.pixel_fwhm**2)
    else:
        fwhm    = sky_map.fwhm[0]
    freq    = sky_map.obs_frequency
    nside   = sky_map.nside
    unid    = sky_map.physical_units[0]
    npix    = define_npix(nside)

    mapa    = sky_map.copy()

    plt.ioff()
    patch_I = mapa[0].patch(coord,npix,resampling=image_resampling)
    patch_Q = mapa[1].patch(coord,npix,resampling=image_resampling)
    patch_U = mapa[2].patch(coord,npix,resampling=image_resampling)
    plt.ion()

    out_dict = {'I':patch_I,
                'Q':patch_Q,
                'U':patch_U,
                'FWHM':fwhm,
                'FREQ':freq,
                'UNIT':unid}

    return out_dict


# %%  PATCH OPERATIONS

def arc_min(d):
    return d.to(u.arcmin).value

def d2pix(d,patch):
    return (d/patch.pixsize).si.value

def stats_central(patch,fwhm,clip=None,verbose=False):

    """
    Performs basic statistics around the center of a given patch of the sky.
    By default, the radii of the rings used for statistics are:
        - A central circle or radius = 1sigma, for estimating the peak and
            valley values.
        - An outer ring for computing mean value and stddev, with
            - inner radius = 7sigma and
            - outer radius = image half size - 2*FWHM


    Parameters
    ----------
    patch : Imagen object (see the Imagen class in sky_images.py)
        An input patch (either I, Q or U).

    fwhm : astropy.units.quantity.Quantity
        Beam FWHM. Its value is used to compute the inner an outer
        radii of the rings used for the analysis.

    clip : bool
        If True, sigma clipping is used for the statistics

    verbose: bool
        If True, some control messages are written on screen


    Returns
    -------
    A dictionary containing:

        - 'MIN' and 'MAX': The minimum and maximum values inside the central
            circle, once the average value of the outer ring has been removed.
        - 'MEAN': Average of the pixels within the outer ring
        - 'STD': Standard deviation of the pixels within the outer ring
        - 'PEAK', 'BOTTOM': The minimum and maximum of values inside the
            central circle

    """

    rmin_central = 1*fwhm2sigma*fwhm
    rmax_central = 3*fwhm2sigma*fwhm
    rmin_stats   = 7*fwhm2sigma*fwhm
    rmax_stats   = patch.size[0]*patch.pixsize/2-2*fwhm

    if verbose:
        print(' Inner radius = {0} arcmin = {1} pixels'.format(arc_min(rmin_central),
                                                               d2pix(rmin_central,patch)))
        print(' Inner stat radius = {0} arcmin = {1} pixels'.format(arc_min(rmin_stats),
                                                                    d2pix(rmin_stats,patch)))
        print(' Outer stat radius = {0} arcmin = {1} pixels'.format(arc_min(rmax_stats),
                                                               d2pix(rmax_stats,patch)))

    st1          = patch.stats_in_rings(rmin_central,rmax_central,clip=clip)
    st2          = patch.stats_in_rings(rmin_stats,rmax_stats,clip=clip)

    return {'MIN'   :st1['min_inside']-st2['mean'],
            'MAX'   :st1['max_inside']-st2['mean'],
            'MEAN'  :st2['mean'],
            'STD'   :st2['std'],
            'PEAK'  :st1['max_inside'],
            'BOTTOM':st1['min_inside']}

def peak_info(patch,
              fwhm,
              keyname       = 'I',
              take_positive = False,
              x             = None,
              y             = None,
              clip          = None):

    """
    Returns a dictionary containing the intensity, rms and position of a
    local extremum near the center of a patch. If the input parameters X and Y
    are not None, then the ruotine returns the values for that pixel instead.
    If take_positive is set to True, then the routine returns either the peak
    or valley values, depending on which one has the larger absolute value.


    Parameters
    ----------
    patch : Imagen object (see the Imagen class in sky_images.py)
        An input patch (either I, Q or U).

    fwhm : astropy.units.quantity.Quantity
        Beam FWHM. Its value is used to compute the inner an outer
        radii of the rings used for the analysis.

    keyname : string
        The name of the quantity that is going to be extracted. It normally
        will be 'I', 'Q' or 'U'

    take_positive: bool
        If True, the output will be forced to return the value of the maximum.
        If False, the output will return either the maximum or the minimum,
        depending on which one has the largest absolute value

    x,y : integers
        If None, the routine will find the location of the extremum and
        return values at that position. If not none, the routine will return
        values at the pixel [X,Y]


    Returns
    -------
    A dictionary containing:

        - '<keyname> peak' : The value of the maximum.
        - '<keyname> err'  : The estimated error of the peak value.
        - '<keyname> X'    : The X position (pixel) where the peak is measured.
        - '<keyname> Y'    : The Y position (pixel) where the peak is measured.
        - '<keyname> coord': The coordinate on the sky where the peak is located.

    """


    stats    = stats_central(patch,fwhm,clip=clip)
    out_dict = {}

    out_dict['{0} err'.format(keyname)]   = stats['STD']

    if x is not None:

        out_dict[keyname]                     = patch.datos[x,y]-stats['MEAN']
        out_dict['{0} peak'.format(keyname)]  = patch.datos[x,y]
        out_dict['{0} X'.format(keyname)]     = x
        out_dict['{0} Y'.format(keyname)]     = y
        out_dict['{0} coord'.format(keyname)] = patch.pixel_coordinate(x,y)

    else:

        condplus = np.abs(stats['MAX']) >= np.abs(stats['MIN'])

        if take_positive or condplus:
            pos = np.where(patch.datos  == stats['PEAK'])
            out_dict[keyname]                     = stats['MAX']
            out_dict['{0} peak'.format(keyname)]  = stats['PEAK']
        else:
            pos = np.where(patch.datos  == stats['BOTTOM'])
            out_dict[keyname]                     = stats['MIN']
            out_dict['{0} peak'.format(keyname)]  = stats['BOTTOM']

        i   = pos[0][0]
        j   = pos[1][0]
        out_dict['{0} X'.format(keyname)]     = i
        out_dict['{0} Y'.format(keyname)]     = j
        out_dict['{0} coord'.format(keyname)] = patch.pixel_coordinate(i,j)

    return out_dict

# %%  MATCHED FILTER AROUND A GIVEN COORDINATE

def filter_coordinate(sky_map,coord,beam='default'):

    """
    Matched filtered sky patches from Gnomic projection patches from
    HEALPix maps around a coordinate

    Parameters
    ----------
    sky_map : Fitsmap object (see the Fitsmap class in fits_maps.py)
        The input LiteBIRD image, as a Fitsmap object.

    coord : astropy.SkyCoord
        Coordinate around which the sky is being projected.

    beam : string or FHWM (as an astropy.units.quantity.Quantity) or
        two-dimensional beam image array (of the same dimensions as
        the patches that will be generated):
            - If beam='default' the patches will be filtered with an
                ideal Gaussian beam whose FWHM is the one read from
                the sky_map (and, if use_pixel_window is set to True, also
                the pixel effective window FWHM).
            - If beam is an astropy.units.quantity.Quantity, the patches
                will be filtered with an ideal Gaussian beam whose FWHM
                is the one here specified.
            - If beam is a two-dimensional array, the patches will be
                filtered using the beam function defined in the array
                (assumed to be its real space expression). In this case,
                the iterative matched filtering option is deactivated.


    Returns
    -------
    A dictionary containing:

        - Three flat sky patches, one for each Stokes
            parameters. By default, each patch is a Imagen object (see
            the Imagen class in sky_images.py) containing a flat patch
            14.658 x 14.658 degrees wide. The pixel area is equal to
            the parent HEALPix map pixel area, unless the image_resampling
            parameter is larger than one (in that case the pixel area is
            rescaled accordingly).
        - Three matched-filtered flat sky patches, one for each Stokes
            parameters. The size and number of pixels are the same as those
            of the corresponding non-filtered image
        - The FWHM of the patch beam. This FWHM is the same as the HEALPix
            parent map, unless the 'use_pixel_window' parameter is set to
            True. In that case, the HEALPix pixel effective window is
            quadratically added to the nominal FWHM
        - The central frequency of observation. It is the same as the parent
            HEALPIX map.
        - The physical units of the patches. It is the same as the parent
            HEALPIX map.

    """

    from astropy.units.quantity import Quantity as quant

    patches_dict = get_patches(sky_map,coord)

    if beam == 'default':
        fwhm   = patches_dict['FWHM']
        iterat = iterative_MF
    elif type(beam) == quant:
        fwhm   = beam
        iterat = iterative_MF
    else:
        fwhm   = beam
        iterat = False

    for stokes in ['I','Q','U']:
        patch = patches_dict[stokes]
        if iterat:
            r,fp = patch.iter_matched(fwhm=fwhm)
        else:
            fp   = patch.matched(fwhm=fwhm)
        h = fp.header
        patches_dict['MF {0}'.format(stokes)] = fp

    return patches_dict

# %%  GAUSSIAN FITTING

def peak_fit(patch,
             fwhm = None,
             x    = None,
             y    = None):

    """
    Fits a 2D Gaussian plus a planar baseline to the central region of a patch.
    If the FWHM or either one of X,Y keywords are not NONE,
    then the fit is constrained to that width and/or position.

    Parameters
    ----------
    patch : Imagen object (see the Imagen class in sky_images.py)
        An input patch (either I, Q or U).

    fwhm : astropy.units.quantity.Quantity
        Beam FWHM. If None, the fitting routine will try to estimate the FWHM
        of the central peak. If FWHM is not note, the routine will force the
        fitting to a Gaussian profile with that precise FWHM.

     x,y : integer
        If set to None, the routine will try to fit the location of the
        Gaussian peak. If not NONE, the fitting will be forced to happen around
        the X,Y pixel.


    Returns
    -------
    fit: Ajuste object (see gauss2dfit.py)
        An instance of the Ajuste class, it contains not only the fitted
        parameters, but also copies of the residual, fitted model and
        synthetic (Gaussian+linear baseline) model.

    coord: astropy.SkyCoord
        Sky coordinate of the fitted peak.

    """

    lsize        = 4*u.deg
    lsize_pix    = int((lsize/patch.pixsize).si.value)
    patch_center = patch.stamp_central_region(lsize_pix)
    image        = patch_center.datos

    if fwhm is None:

        if x is None:
            source_fit = fit_single_peak(image)
        else:
            source_fit = fit_single_peak(image,
                                         fixcenter=True,center=(x,y))

    else:

        sigma = ((fwhm2sigma*fwhm)/patch.pixsize).si.value
        if x is None:
            source_fit = fit_single_peak(image,
                                         fixwidth=True,fixed_sigma=sigma)
        else:
            source_fit = fit_single_peak(image,
                                         fixwidth=True,fixed_sigma=sigma,
                                         fixcenter=True,center=(x,y))

    return source_fit,patch_center.pixel_coordinate(source_fit.x,source_fit.y)


# %%  POLARIMETRIC ESTIMATOR

def significance_level(image,value):

    n = image.datos.count()
    x = image.datos[image.datos.mask==False].flatten()
    m = np.count_nonzero(x>=value)

    sig = 1.0-m/n
    if sig == 1.0:
        sig = 1.0-1.0/n

    return sig

def pol_angle(Q,U):
    phi = 0.5*np.arctan2(-U,Q)*u.rad
    return phi.to(u.deg)

def pol_angle_error(Q,U,sQ,sU):
    nume  = Q*Q*sU*sU + U*U*sQ*sQ
    deno  = 4*(Q*Q+U*U)**2
    sigma = np.sqrt(nume/deno)*u.rad
    return sigma.to(u.deg)

def polfrac_error(I,P,sI,sP):
    nume = I*I*sP*sP + P*P*sI*sI
    deno = np.power(I,4)
    return np.sqrt(np.divide(nume,deno))

def P_from_dict(dicc):

    Q    = dicc['Q']
    U    = dicc['U']
    sQ   = dicc['Q err']
    sU   = dicc['U err']

    deno = np.sqrt(Q*Q+U*U)
    nume = Q*Q*sQ*sQ + U*U*sU*sU

    P    = np.sqrt(Q**2+U**2)
    sP   = np.sqrt(nume)/deno
    if (P**2-sP**2) >= 0.0:
        debP = np.sqrt(P**2-sP**2)
    else:
        debP = P

    Pmap = dicc['Patch MF P'].copy()
    Pmap.mask_border(signif_border)
    Pmap.mask_brightest_fraction(signif_fraction)
    sgn  = significance_level(Pmap,P)

    polfrac     = 100*debP/dicc['I']
    polfrac_err = 100*polfrac_error(dicc['I'],P,dicc['I err'],sP)

    return {'P':P,'debiased P':debP,'P err':sP,'P significance level':sgn,
            'pol frac [%]':polfrac,
            'pol frac err [%]':polfrac_err,
            'pol angle':pol_angle(Q,U),
            'pol angle err':pol_angle_error(Q,U,sQ,sU),
            'pol angle fit':pol_angle(dicc['Gaussian fit Q'].amplitude,
                                      dicc['Gaussian fit U'].amplitude)}

# %%  I,Q,U,P ESTIMATION AROUND A GIVEN COORDINATE

def get_IQUP(sky_map,
             coord,
             return_abbrv  = False,
             QU_mode       = 'intensity'):

    """
    Returns the estimated photometry of a source candidate located at
    a given coordinate of the sky.

    Parameters
    ----------
    sky_map : Fitsmap object (see the Fitsmap class in fits_maps.py)
        The input LiteBIRD image, as a Fitsmap object.

    coord : astropy.SkyCoord
        Coordinate around which the sky is being projected.

    return_abbrv : bool
        If True, the output dictionary is much abbreviated, keeping only
        essential info.

    QU_mode : string
        If set to 'intensity', then Q and U are measured exactly at the
        position where the maximum of the intensity map has been measured. If not,
        they are measured on the local maxima (or minima) around the centre
        of the filtered patch.



    Returns
    -------
    A dictionary containing:
        - Parent map info: units, central frequency, FWHM, pixel size.
        - Configuration parameters: image resampling factor, iterative matched
            filtering, QU mode, ideal beams.
        - Central coordinate of the patch.
        - For Stokes' parameters I,Q,U:
            - The original patch.
            - The matched filtered patch.
            - Value, position and coordinate of the peak.
            - Rms calculated in a ring around the peak.
            - Separation of the peak with respect to the geometrical center
                of the patch.
            - Gaussian fit to the peak.
        - For polarization P: the same information as for I,Q,U plus:
            - Debiased estimation of P.
            - Significance level of the P detection.

    """

    patches = filter_coordinate(sky_map,coord)

    fwhm                                = patches['FWHM']
    out_dict                            = {}
    out_dict['FWHM']                    = fwhm
    out_dict['Beam area']               = sky_map.beam_area[0]
    out_dict['Coord']                   = coord
    out_dict['Ideal beams']             = use_ideal_beams
    out_dict['QU mode']                 = QU_mode
    out_dict['Iterative MF']            = iterative_MF
    out_dict['Image resampling factor'] = image_resampling
    out_dict['UNIT']                    = patches['UNIT']
    out_dict['Freq']                    = patches['FREQ']

    dictI     = peak_info(patches['MF I'],fwhm,keyname = 'I',take_positive=True)

    if QU_mode == 'intensity':
        dictQ = peak_info(patches['MF Q'],fwhm,keyname = 'Q',
                          x=dictI['I X'],y=dictI['I Y'])
        dictU = peak_info(patches['MF U'],fwhm,keyname = 'U',
                          x=dictI['I X'],y=dictI['I Y'])
    else:
        dictQ = peak_info(patches['MF Q'],fwhm,keyname = 'Q')
        dictU = peak_info(patches['MF U'],fwhm,keyname = 'U')

    out_dict.update(dictI)
    out_dict.update(dictQ)
    out_dict.update(dictU)

    psize    = sky_map.pixel_size

    out_dict['Map pixel size']   = psize.to(u.arcmin)
    out_dict['Patch pixel size'] = patches['I'].pixsize.to(u.arcmin)

    for s in stokes:
        c = out_dict['{0} coord'.format(s)]
        out_dict['{0} separation (arcmin)'.format(s)] = coord.separation(c).to(u.arcmin).value
        out_dict['{0} separation (pixels)'.format(s)] = (coord.separation(c)/psize).si.value

    fitI,cI       = peak_fit(patches['I'],fwhm=fwhm)
    fitQ,cI       = peak_fit(patches['Q'],fwhm=fwhm,x=fitI.x,y=fitI.y)
    fitU,cI       = peak_fit(patches['U'],fwhm=fwhm,x=fitI.x,y=fitI.y)

    out_dict['Gaussian fit I']      = fitI
    out_dict['Gaussian fit Q']      = fitQ
    out_dict['Gaussian fit U']      = fitU
    out_dict['Gaussian fit coord']  = cI

    free_fitI,cIf = peak_fit(patches['I'])
    fwhmIfit      = sigma2fwhm*free_fitI.sigma*patches['I'].pixsize
    free_fitQ,cIf = peak_fit(patches['Q'],fwhm=fwhmIfit,x=free_fitI.x,y=free_fitI.y)
    free_fitU,cIf = peak_fit(patches['U'],fwhm=fwhmIfit,x=free_fitI.x,y=free_fitI.y)

    out_dict['Free Gaussian fit I'] = free_fitI
    out_dict['Free Gaussian fit Q'] = free_fitQ
    out_dict['Free Gaussian fit U'] = free_fitU
    out_dict['Free Gaussian fit coord'] = cIf

    out_dict['Patch I']    = patches['I'].copy()
    out_dict['Patch Q']    = patches['Q'].copy()
    out_dict['Patch U']    = patches['U'].copy()
    out_dict['Patch MF I'] = patches['MF I'].copy()
    out_dict['Patch MF Q'] = patches['MF Q'].copy()
    out_dict['Patch MF U'] = patches['MF U'].copy()

#      P from the matched filtered Q and U maps

    out_dict['Patch MF P'] = (out_dict['Patch MF U']**2 + out_dict['Patch MF Q']**2)**(1/2)
    out_dict['Patch MF P'].image_header['TTYPE1'] = 'P POLARIZATION'
    diccP = P_from_dict(out_dict.copy())
    out_dict.update(diccP)

#      P from a Gaussian fit in the unfiltered maps

    out_dict['Patch P']        = (out_dict['Patch U']**2 + out_dict['Patch Q']**2)**(1/2)
    out_dict['Patch P'].image_header['TTYPE1'] = 'P POLARIZATION'
    fitP,cP                    = peak_fit(out_dict['Patch P'],fwhm=fwhm,x=fitI.x,y=fitI.y)
    out_dict['Gaussian fit P'] = fitP
    out_dict['Gaussian fit P significance level'] = (1 -
        np.count_nonzero(fitP.residual>=fitP.amplitude)/fitP.residual.size)

    summary = {'I':out_dict['I'],
               'fit I':fitI.amplitude,
               'ffit I':free_fitI.amplitude,
               'Q':out_dict['Q'],
               'fit Q':fitQ.amplitude,
               'ffit Q':free_fitQ.amplitude,
               'U':out_dict['U'],
               'fit U':fitU.amplitude,
               'ffit U':free_fitU.amplitude,
               'FWHM':fwhm.to(u.arcmin).value,
               'fit FWHM':(fitI.sigma*sigma2fwhm*patches['I'].pixsize).to(u.arcmin).value,
               'ffit FWHM':(free_fitI.sigma*sigma2fwhm*patches['I'].pixsize).to(u.arcmin).value,
               'P':out_dict['P'],
               'unbiased P':out_dict['debiased P'],
               'polfrac [%]':out_dict['pol frac [%]'],
               'polfrac err [%]':out_dict['pol frac err [%]'],
               'P fit':out_dict['Gaussian fit P'].amplitude,
               'P significance':out_dict['P significance level'],
               'P fit significance':out_dict['Gaussian fit P significance level'],
               'pol angle [deg]':out_dict['pol angle'].value,
               'pol angle err [deg]':out_dict['pol angle err'].value,
               'pol angle fit [deg]':out_dict['pol angle fit'].value}

    if return_abbrv:
        return out_dict,summary
    else:
        return out_dict

# %%  PHOTOMETRY CLASS  ----------------------------

class Photometry:

    def __init__(self,value,error,significance,outer):
        self.value        = value
        self.error        = error
        self.significance = significance
        self.outer        = outer

    def copy(self):
        x = Photometry(self.value,
                       self.error,
                       self.significance,
                       self.outer)
        return x

    @property
    def snr(self):
        return self.value/self.error

    @property
    def Jy(self):
        f = self.outer.to_Jy
        return Photometry(f*self.value,
                          f*self.error,
                          self.significance,
                          self.outer)

# %%  SOURCE CLASS  --------------------------------

class Source:

    def __init__(self,diccio):
        self.diccio = diccio

    def copy(self):
        x = self.diccio.copy()
        return Source(x)

# %% -- properties

    @property
    def coord(self):
        return self.diccio['Coord']

    @property
    def fwhm(self):
        return self.diccio['FWHM'].to(u.deg)

    @property
    def area(self):
        return self.diccio['Beam area'].to(u.sr)

    @property
    def nu(self):
        return self.diccio['Freq'].to(u.GHz)


    @property
    def unit(self):
        return parse_unit(self.diccio['UNIT'])

    @property
    def to_Jy(self):
        return convert_factor(self.unit,u.Jy,nu=self.nu,beam_area=self.area)

# %% -- comparisons

    def has_better_SNR(self,other):
        if self.I.snr >= other.I.snr:
            return True
        else:
            return False

    def has_better_significance(self,other):
        if self.P.significance >= other.P.significance:
            return True
        else:
            return False


# %% -- plotting

    def draw(self,tofile=None):

        plt.figure(figsize=(16,12))
        plt.subplot(221)
        self.diccio['Patch I'].draw()
        plt.title('I [{0}]'.format(self.unit))
        plt.subplot(222)
        self.diccio['Patch Q'].draw()
        plt.title('Q [{0}]'.format(self.unit))
        plt.subplot(223)
        self.diccio['Patch U'].draw()
        plt.title('U [{0}]'.format(self.unit))
        if tofile is not None:
            plt.savefig(tofile)

    def mfdraw(self,tofile=None):

        plt.figure(figsize=(16,12))
        plt.subplot(221)
        self.diccio['Patch MF I'].draw()
        plt.title('MF I [{0}]'.format(self.unit))
        plt.subplot(222)
        self.diccio['Patch MF Q'].draw()
        plt.title('MF Q [{0}]'.format(self.unit))
        plt.subplot(223)
        self.diccio['Patch MF U'].draw()
        plt.title('MF U [{0}]'.format(self.unit))
        if tofile is not None:
            plt.savefig(tofile)


# %% -- photometry

    @property
    def I(self):
        return Photometry(self.diccio['I'],
                          self.diccio['I err'],
                          0,
                          self.copy())

    @property
    def Q(self):
        return Photometry(self.diccio['Q'],
                          self.diccio['Q err'],
                          0,
                          self.copy())

    @property
    def U(self):
        return Photometry(self.diccio['U'],
                          self.diccio['U err'],
                          0,
                          self.copy())

    @property
    def P(self):
        return Photometry(self.diccio['debiased P'],
                          self.diccio['P err'],
                          self.diccio['P significance level'],
                          self.copy())

    @property
    def polfrac(self):
        return Photometry(self.diccio['pol frac [%]'],
                          self.diccio['pol frac err [%]'],
                          self.diccio['P significance level'],
                          self.copy())

    @property
    def angle(self):
        return Photometry(self.diccio['pol angle'],
                          self.diccio['pol angle err'],
                          0,
                          self.copy())


    @property
    def Ifit(self):
        return Photometry(self.diccio['Gaussian fit I'].amplitude,
                          self.diccio['Gaussian fit I'].residual.std(),
                          0,
                          self.copy())

    @property
    def Qfit(self):
        return Photometry(self.diccio['Gaussian fit Q'].amplitude,
                          self.diccio['Gaussian fit Q'].residual.std(),
                          0,
                          self.copy())

    @property
    def Ufit(self):
        return Photometry(self.diccio['Gaussian fit U'].amplitude,
                          self.diccio['Gaussian fit U'].residual.std(),
                          0,
                          self.copy())

    @property
    def Pfit(self):
        return Photometry(self.diccio['Gaussian fit P'].amplitude,
                          self.diccio['Gaussian fit P'].residual.std(),
                          0,
                          self.copy())

    @property
    def polfrac_fit(self):
        return Photometry(100*self.Pfit.value/self.Ifit.value,
                          100*polfrac_error(self.Ifit.value,
                                            self.Pfit.value,
                                            self.Ifit.error,
                                            self.Pfit.error),
                          self.diccio['P significance level'],
                          self.copy())

    @property
    def angle_fit(self):
        return Photometry(self.diccio['pol angle fit'],
                          pol_angle_error(self.Qfit.value,
                                          self.Ufit.value,
                                          self.Qfit.error,
                                          self.Ufit.error),
                          0,
                          self.copy())


# %% -- input/output

    @classmethod
    def from_coordinate(self,sky_map,coordinate):
        d = get_IQUP(sky_map,coordinate)
        return Source(d)

    @classmethod
    def from_name(self,sky_map,name):
        coordinate = SkyCoord.from_name(name)
        return self.from_coordinate(sky_map,coordinate)




