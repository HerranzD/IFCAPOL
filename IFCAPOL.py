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
import tarfile
import shutil
import datetime
import numpy                    as     np
import matplotlib.pyplot        as     plt
import astropy.units            as     u
import healpy                   as     hp

from astropy.coordinates        import SkyCoord
from myutils                    import fwhm2sigma,sigma2fwhm
from sky_images                 import Imagen
from gauss2dfit                 import fit_single_peak
from unit_conversions           import parse_unit,convert_factor

# %%  Inherited experiment characteristics (liteBIRD)

import survey_model       as     survey

# %%  Ignore warnings:

import warnings
warnings.filterwarnings("ignore")

# %%  IFCAPOL default parameters:

stokes           = ['I','Q','U']   # Names of the Stokes parameters
use_ideal_beams  = True            # Use or not ideal Gaussian beam PSF
use_pixel_window = True            # Add or not the pixel effective beam to
                                   #    the beam FWHM for filtering effects
signif_border    = 10              # Number of border pixels to exclude
                                   #    for the calculation of P significance
signif_fraction  = 0.025           # Graction of britht pixels to mask
                                   #    for the calculation of P significance
iterative_MF     = True            # Iterative matched filtering
image_resampling = 1               # Resampling factor when projecting patches

sigmaclipping    = 2.5             # Sigma clipping for rms estimation in rings

# %%  SKY PATCHING

def define_npix(nside):
    """
    The finbes the number of pixels of a patch 14.658 x 14.658 degrees wide,
    depending on the nside parameter of the HEALPix parent map. This number
    of pixels is set so that the flat patch pixel area is roughly equal to
    the HEALPix pixel size.

    Parameters
    ----------
    nside : int
        HEALpix nside parameter.

    Returns
    -------
    int
        Number of pixels (per axis) of the square projected patch.

    """
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
    """
    Converts an angular Quantity into its value in arcmin

    Parameters
    ----------
    d : astropy.units.quantity.Quantity
        Angle.

    Returns
    -------
    float
        The value of the angle in arcmin.

    """
    return d.to(u.arcmin).value

def d2pix(d,patch):
    """
    Express an angular Quantity as a number of pixels in an IFCAPOL flat patch

    Parameters
    ----------
    d : astropy.units.quantity.Quantity
        Angle.
    patch : Imagen object (see the Imagen class in sky_images.py)
        Imagen object containing the patch

    Returns
    -------
    TYPE
        The angle expressed as a number of pixels.

    """
    return (d/patch.pixsize).si.value

def stats_central(patch,fwhm,clip=None,verbose=False,toplot=False):

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

    toplot: bool, optional
        If True, makes a plot showing the rings used for statistcal analysis


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

    if toplot:

        r0 = (rmin_central/patch.pixsize).si.value
        r1 = (rmin_stats/patch.pixsize).si.value
        r2 = (rmax_stats/patch.pixsize).si.value
        ll = patch.size[0]//2
        fig,ax = plt.subplots(1)
        ax.imshow(patch.datos)
        circle0 = plt.Circle((ll,ll),r0,color='green',alpha=0.2)
        circle1 = plt.Circle((ll,ll),r1,color='red',alpha=0.2)
        circle2 = plt.Circle((ll,ll),r2,color='yellow',alpha=0.1)
        ax.add_patch(circle0)
        ax.add_patch(circle1)
        ax.add_patch(circle2)

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
              clip          = sigmaclipping):

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

# %%  MEXICAN HAT WAVELET AROUND A GIVEN COORDINATE

def mhw_coordinate(sky_map,coord,beam='default',order=2):

    """
    MHW2 filtered sky patches from Gnomic projection patches from
    HEALPix maps around a coordinate.

    Parameters
    ----------
    sky_map : Fitsmap object (see the Fitsmap class in fits_maps.py)
        The input LiteBIRD image, as a Fitsmap object.

    coord : astropy.SkyCoord
        Coordinate around which the sky is being projected.

    beam : string or FHWM (as an astropy.units.quantity.Quantity):
            - If beam='default' the patches will be filtered with an
                ideal Gaussian beam whose FWHM is the one read from
                the sky_map (and, if use_pixel_window is set to True, also
                the pixel effective window FWHM).
            - If beam is an astropy.units.quantity.Quantity, the patches
                will be filtered with an ideal Gaussian beam whose FWHM
                is the one here specified.
    order : int
        The MHW family order. The default value is 2 for MHW2.

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
        - Three MHW2-filtered flat sky patches, one for each Stokes
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

    patches_dict = get_patches(sky_map,coord)

    if beam == 'default':
        fwhm   = patches_dict['FWHM']
    else:
        fwhm   = beam

    for stokes in ['I','Q','U']:
        patch = patches_dict[stokes]
        fp    = patch.mhw(fwhm=fwhm,order=order)
        patches_dict['MHW2 {0}'.format(stokes)] = fp

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
    """
    Returns the significance level of a given value in a given image

    Parameters
    ----------
    image : Imagen object (see the Imagen class in sky_images.py)
        An input patch (either I, Q or U).
    value : float
        Value to be tested.

    Returns
    -------
    float
        Significance level of the value in the image.

    """

    n = image.datos.count()
    x = image.datos[image.datos.mask==False].flatten()
    m = np.count_nonzero(x>=value)

    sig = 1.0-m/n
    if sig == 1.0:
        sig = 1.0-1.0/n

    return sig

def pol_angle(Q,U):
    """
    Returns the polarization angle phi, given the Stokes parameters Q,U.

    Parameters
    ----------
    Q : float
        Stokes Q.
    U : float
        Stokes U.

    Returns
    -------
    astropy quantity
        The polarization angle phi in degrees

    """
    phi = 0.5*np.arctan2(-U,Q)*u.rad
    return phi.to(u.deg)

def pol_angle_error(Q,U,sQ,sU):
    """
    Returns the error of the polarization angle phi, given the Stokes
    parameters Q,U and their associated errors.

    Parameters
    ----------
    Q : float
        Stokes Q.
    U : float
        Stokes U.
    sQ : float
        Error of Stokes Q.
    sU : float
        Error of Stokes U.

    Returns
    -------
    astropy quantity
        The error of the polarization angle phi in degrees

    """
    nume  = Q*Q*sU*sU + U*U*sQ*sQ
    deno  = 4*(Q*Q+U*U)**2
    sigma = np.sqrt(nume/deno)*u.rad
    return sigma.to(u.deg)

def polfrac_error(I,P,sI,sP):
    """
    Returns the error of the polarization fraction P, given the Stokes
    parameters I,Q,U and their associated errors.

    Parameters
    ----------
    I : float
        Stokes I.
    P : float
        Polarization P.
    sI : float
        Error of Stokes I.
    sP : float
        Error of polarization P.

    Returns
    -------
    float
        The error of the polarization fraction P.

    """

    nume = I*I*sP*sP + P*P*sI*sI
    deno = np.power(I,4)
    return np.sqrt(np.divide(nume,deno))

def P_from_dict(dicc):
    """
    Returns a dictionary containing the polarization parameters estimated
    from the Stokes parameters I,Q,U. The Stokes parameters are assumed to
    be contained in the input dictionary.

    Parameters
    ----------
    dicc : dictionary
        Input dictionary containing the Stokes parameters I,Q,U and their associated errors,
        plus the Gaussian fits to the Stokes parameters. The dictionary must also contain
        the matched filtered Stokes parameters and the matched filtered polarization P map.
        This dictionary can be obtained from the output of the get_IQUP function.

    Returns
    -------
    A dictionary containing:
        - Polarization P and its error.
        - Debiased polarization P and its error.
        - Polarization significance level.
        - Polarization fraction and its error.
        - Polarization angle and its error.
        - Polarization angle from the Gaussian fit to Q and U.
        - Polarization angle error from the Gaussian fit to Q and U.

    """

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
        """
        Creates a Photometry instance

        Parameters
        ----------
        value : astropy Quantity
            A photometric value.
        error : astropy Quantity
            Associated error.
        significance : float
            Significance or signal to noise ratio.
        outer : Photometry
            Outer instance to interface with the parent Souce class.

        Methods summary
        ---------------
        copy()
            Makes a copy of the Photometry instance.

        snr
            Returns the signal to noise ratio of the Photometry object.

        Jy
            Returns a Photometry instance converted to jansky units.

        """
        self.value        = value
        self.error        = error
        self.significance = significance
        self.outer        = outer

    def copy(self):
        """
        Makes a copy of the Photometry instance
        """
        x = Photometry(self.value,
                       self.error,
                       self.significance,
                       self.outer)
        return x

    @property
    def snr(self):
        """
        Returns the signal to noise ratio of the Photometry object
        """

        return self.value/self.error

    @property
    def Jy(self):
        """
        Returns a Photometry instance converted to jansky units
        """
        f = self.outer.to_Jy
        return Photometry(f*self.value,
                          f*self.error,
                          self.significance,
                          self.outer)

# %%  SOURCE CLASS  --------------------------------

class Source:

    """
    High-level object providing a flexible interface for compact
    source representation, querying and visualization.


    Parameters
    ----------
    diccio : Dictionary
        A dictionary of the type generated by the get_IQUP routine


    Methods summary
    ---------------
    copy()
        Makes a copy of the Source instance.

    coord
        Returns the central coordinate of the sky patch where the source is
        located.

    fwhm
        Returns the FWHM used for source detection and photometry.

    estimated_fwhm
        Returns the fwhm estimated by a Gaussian fitting
        near the center of the image.

    area()
        Returns the area of the beam assumed for the source.

    nu
        Returns the observing frequency.

    unit
        Returns the units of the maps used for detecting the source.

    to_Jy
        Returns the conversion factor from the units of the maps to Jy.

    print_date
        Writes the date of creation of this Source instance.

    has_better_SNR(other)
        Checks if this source has better signal to noise ratio in intensity
        than Other.

    has_better_significance(other)
        Checks if this source has better signification in polarization
        than Other.

    draw()
        Plots I,Q,U in three subplots.

    mfdraw()
        Plots the matched filtered I,Q,U,P in four subplots.

    I,Q,U,P,angle,polfrac
        Return a Photometry of the intensity por polarization of the source.
        The value used for P is the unbiased P estimator.

    Ifit,Qfit,Ufit,Pfit,angle_fit,polfrac_fit
        Return a Photometry of the intensity por polarization of the source,
        estimated by a Gaussian fitting. The value used for P is the unbiased
        P estimator.

    from_coordinate(sky_map,coord)
        Returns a Source object from the sky_map, around coord.

    from_object_name(sky_map,coord)
        Returns a Source object from the sky_map, around a source with
        a known name.

    from_tgz(name)
        Reads a Source object from a saved .tgz file.

    write_tgz(name)
        Writes a Source object to a .tgz file.

    """

    def __init__(self,diccio):
        """
        Creates a Source instance from a dictionary of the type generated by
        the get_IQUP routine.
        """
        self.diccio = diccio

    def copy(self):
        """
        Makes a copy of the Source instance.
        """
        x = self.diccio.copy()
        return Source(x)

# %% -- properties

    @property
    def coord(self):
        """
        Returns the central coordinate of the sky patch where the source is
        located.
        """
        return self.diccio['Coord']

    @property
    def fwhm(self):
        """
        Returns the FWHM used for source detection and photometry.
        """
        return self.diccio['FWHM'].to(u.arcmin)

    @property
    def area(self):
        """
        Returns the area of the beam assumed for the source.
        """
        return self.diccio['Beam area'].to(u.sr)

    @property
    def nu(self):
        """
        Returns the observing frequency.
        """
        return self.diccio['Freq'].to(u.GHz)

    @property
    def unit(self):
        """
        Returns the units of the maps used for detecting the source.
        """
        return parse_unit(self.diccio['UNIT'])

    @property
    def to_Jy(self):
        """
        Returns the conversion factor from the units of the maps to Jy.
        """
        return convert_factor(self.unit,u.Jy,nu=self.nu,beam_area=self.area)

    @property
    def estimated_fwhm(self):
        """
        Returns the fwhm estimated by the Gaussian fitting
        associated to the Source.
        """
        return self.diccio['Free Gaussian fit I'].sigma*sigma2fwhm*self.diccio['Patch I'].pixsize.to(u.arcmin)

    @property
    def print_date(self):
        """
        Writes the date of creation of this Source instance.
        """
        try:
            t = self.diccio['Creation time']
            print(t)
        except KeyError:
            pass

# %% -- comparisons

    def has_better_SNR(self,other):
        """
        Checks if this source has better signal to noise ratio in intensity
        than Other.

        Parameters
        ----------
        other : Source
            The other Source object to be compared with.

        Returns
        -------
        bool:
            True if this source has better signal to noise ratio in intensity
            than Other. False otherwise.

        """

        if self.I.snr >= other.I.snr:
            return True
        else:
            return False

    def has_better_significance(self,other):
        """
        Checks if this source has better signification in polarization
        than Other.

        Parameters
        ----------
        other : Source
            The other Source object to be compared with.

        Returns
        -------
        bool:
            True if this source has better signification in polarization
            than Other. False otherwise.

        """
        if self.P.significance >= other.P.significance:
            return True
        else:
            return False

# %% -- quality checks

    @property
    def flag_photometry(self,tol=0.1):
        """
        Defines a quality flag depending on the agreement between
        the matched filter and Gaussian fitting photometric estimators
        for total intensity I.

        Parameters
        ----------
        tol : float, optional
            The tolerance for the discrepancy. The default is 0.1.

        Returns
        -------
        bool:
            True if the matched filter and the Gaussian fitting photometric
            estimators disagree by a relative factor greater than tol.
            False otherwise

        """

        rel = np.abs((self.I.value-self.Ifit.value)/self.I.value)
        if rel > tol:
            return True
        else:
            return False


    @property
    def flag_extension(self,tol=0.1):
        """
        Defines a quality flag depending on the agreement between
        the pre-defined beam FWHM and the Gaussian fitted FWHM.

        Parameters
        ----------
        tol : float, optional
            The tolerance for the discrepancy. The default is 0.1.

        Returns
        -------
        bool:
            True if both FWHMs disagree by more than tol (in relative)
            terms. False otherwise.

        """

        rel = np.abs((self.fwhm-self.estimated_fwhm)/self.fwhm).si.value
        if rel > tol:
            return True
        else:
            return False

# %% -- plotting

    def draw(self,lsize=None,tofile=None):

        """
        Plots the I,Q,U patches from which the Source has been
        extracted.

        Parameters
        ----------
        lsize : int or None
            If None, the method plots the entire patch. If it is an integer,
            then the method plots a (lsize,lsize) poststamp centered
            at the coordinate of the Source.

        tofile: string or None
            If not note, writes the plot to the given file name.
        """

        plt.figure(figsize=(16,12))

        if lsize is not None:

            self.diccio['Patch I'].stamp_central_region(lsize).draw(pos=221)
            plt.title('I [{0}]'.format(self.unit))
            plt.subplot(222)
            self.diccio['Patch Q'].stamp_central_region(lsize).draw(pos=222)
            plt.title('Q [{0}]'.format(self.unit))
            plt.subplot(223)
            self.diccio['Patch U'].stamp_central_region(lsize).draw(pos=223)
            plt.title('U [{0}]'.format(self.unit))

        else:

            self.diccio['Patch I'].draw(pos=221)
            plt.title('I [{0}]'.format(self.unit))
            plt.subplot(222)
            self.diccio['Patch Q'].draw(pos=222)
            plt.title('Q [{0}]'.format(self.unit))
            plt.subplot(223)
            self.diccio['Patch U'].draw(pos=223)
            plt.title('U [{0}]'.format(self.unit))

        if tofile is not None:
            plt.savefig(tofile)

    def mfdraw(self,lsize=None,tofile=None):
        """
        Plots the I,Q,U,P matched filtered patches from which the Source
        has been extracted.

        Parameters
        ----------
        lsize : int or None
            If None, the method plots the entire patch. If it is an integer,
            then the method plots a (lsize,lsize) poststamp centered
            at the coordinate of the Source.

        tofile: string or None
            If not note, writes the plot to the given file name.
        """

        plt.figure(figsize=(16,12))

        if lsize is not None:

            plt.subplot(221)
            self.diccio['Patch MF I'].stamp_central_region(lsize).draw(pos=221)
            plt.title('MF I [{0}]'.format(self.unit))
            plt.subplot(222)
            self.diccio['Patch MF Q'].stamp_central_region(lsize).draw(pos=222)
            plt.title('MF Q [{0}]'.format(self.unit))
            plt.subplot(223)
            self.diccio['Patch MF U'].stamp_central_region(lsize).draw(pos=223)
            plt.title('MF U [{0}]'.format(self.unit))
            plt.subplot(224)
            self.diccio['Patch MF P'].stamp_central_region(lsize).draw(pos=224)
            plt.title('MF P [{0}]'.format(self.unit))

        else:

            plt.subplot(221)
            self.diccio['Patch MF I'].draw(pos=221)
            plt.title('MF I [{0}]'.format(self.unit))
            plt.subplot(222)
            self.diccio['Patch MF Q'].draw(pos=222)
            plt.title('MF Q [{0}]'.format(self.unit))
            plt.subplot(223)
            self.diccio['Patch MF U'].draw(pos=223)
            plt.title('MF U [{0}]'.format(self.unit))
            plt.subplot(224)
            self.diccio['Patch MF P'].draw(pos=224)
            plt.title('MF P [{0}]'.format(self.unit))


        if tofile is not None:
            plt.savefig(tofile)


# %% -- photometry

    @property
    def I(self):
        """
        Returns the estimated intensity of the
        source, using the matched filter estimation on the I map.
        """
        return Photometry(self.diccio['I'],
                          self.diccio['I err'],
                          0,
                          self.copy())

    @property
    def Q(self):
        """
        Returns the estimated Q Stokes parameter of the
        source, using the matched filter estimation on the Q map.
        """
        return Photometry(self.diccio['Q'],
                          self.diccio['Q err'],
                          0,
                          self.copy())

    @property
    def U(self):
        """
        Returns the estimated U Stokes parameter of the
        source, using the matched filter estimation on the U map.
        """
        return Photometry(self.diccio['U'],
                          self.diccio['U err'],
                          0,
                          self.copy())

    @property
    def P(self):
        """
        Returns the estimated polarization of the
        source, using the Filtered Fusion (FF) estimation method.
        """
        return Photometry(self.diccio['debiased P'],
                          self.diccio['P err'],
                          self.diccio['P significance level'],
                          self.copy())

    @property
    def polfrac(self):
        """
        Returns the estimated polarization fraction of the
        source, using the Filtered Fusion (FF) estimation method.
        """
        return Photometry(self.diccio['pol frac [%]'],
                          self.diccio['pol frac err [%]'],
                          self.diccio['P significance level'],
                          self.copy())

    @property
    def angle(self):
        """
        Returns the estimated polarization angle of the
        source, using the matched filter estimation of Q and U.
        """
        return Photometry(self.diccio['pol angle'],
                          self.diccio['pol angle err'],
                          0,
                          self.copy())


    @property
    def Ifit(self):
        """
        Returns the intensity estimated by the Gaussian fitting
        of the I map.
        """
        return Photometry(self.diccio['Gaussian fit I'].amplitude,
                          self.diccio['Gaussian fit I'].residual.std(),
                          0,
                          self.copy())

    @property
    def Qfit(self):
        """
        Returns the Q Stokes parameter estimated by the Gaussian fitting
        of the Q map.
        """
        return Photometry(self.diccio['Gaussian fit Q'].amplitude,
                          self.diccio['Gaussian fit Q'].residual.std(),
                          0,
                          self.copy())

    @property
    def Ufit(self):
        """
        Returns the U Stokes parameter estimated by the Gaussian fitting
        of the U map.
        """
        return Photometry(self.diccio['Gaussian fit U'].amplitude,
                          self.diccio['Gaussian fit U'].residual.std(),
                          0,
                          self.copy())

    @property
    def Pfit(self):
        """
        Returns the polarization estimated by the Gaussian fitting
        of the Q and U maps.
        """
        return Photometry(self.diccio['Gaussian fit P'].amplitude,
                          self.diccio['Gaussian fit P'].residual.std(),
                          0,
                          self.copy())

    @property
    def polfrac_fit(self):
        """
        Returns the polarization fraction estimated by the Gaussian fitting
        of the Q and U maps.
        """
        return Photometry(100*self.Pfit.value/self.Ifit.value,
                          100*polfrac_error(self.Ifit.value,
                                            self.Pfit.value,
                                            self.Ifit.error,
                                            self.Pfit.error),
                          self.diccio['P significance level'],
                          self.copy())

    @property
    def angle_fit(self):
        """
        Returns the polarization angle estimated by the Gaussian fitting
        of the Q and U maps.
        """

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
        """
        Returns a Source object from the sky_map, around a given sky coordinate

        Parameters
        ----------
        sky_map : Fitsmap object (see the Fitsmap class in fits_maps.py)
            The input LiteBIRD image, as a Fitsmap object.

        coordinate : astropy.SkyCoord
            The coordinate of the source.

        Returns
        -------
        A Source object.

        """

        d = get_IQUP(sky_map,coordinate)
        d['Parent map FWHM'] = sky_map.fwhm[0].to(u.arcmin)
        d['Creation time']   = datetime.datetime.now()
        return Source(d)

    @classmethod
    def from_object_name(self,sky_map,name):
        """
        Returns a Source object from the sky_map, around a source with
        a known name.

        Parameters
        ----------
        sky_map : Fitsmap object (see the Fitsmap class in fits_maps.py)
            The input LiteBIRD image, as a Fitsmap object.

        name : string
            The name of the source.

        Returns
        -------
        A Source object.

        """

        coordinate = SkyCoord.from_name(name)
        return self.from_coordinate(sky_map,coordinate)

    @classmethod
    def from_tgz(self,fname):
        """
        Reads a Source object from a .tgz file.

        Parameters
        ----------
        fname : string
            The name of the input file.

        Returns
        -------
        A Source object.

        """

        tar = tarfile.open(fname, 'r')
        tar.extractall()
        tar.close()

        infile = open('Tempdir/dict.p','rb')
        new_dict = pickle.load(infile)
        infile.close()

        for images in ['Patch I',
                       'Patch Q',
                       'Patch U',
                       'Patch MF I',
                       'Patch MF Q',
                       'Patch MF U',
                       'Patch MF P']:
            n = images.replace(' ','_')
            n = n+'.fits'
            p = Imagen.from_file('Tempdir/'+n)
            new_dict[images] = p

        shutil.rmtree('Tempdir')

        return Source(new_dict)

    def write_tgz(self,fname):
        """
        Writes the Source object to a .tgz file.

        Parameters
        ----------
        fname : string
            The name of the output file.

        Returns
        -------
        None.

        """

        def make_tarfile(output_filename, source_dir):
            with tarfile.open(output_filename, "w:gz") as tar:
                tar.add(source_dir, arcname=os.path.basename(source_dir))

        os.mkdir('Tempdir')
        d    = self.diccio.copy()
        newd = {}
        keys = list(d.keys())
        for k in keys:
            if 'Patch' not in k:
                newd[k] = d[k]
        newd['Patch pixel size'] = d['Patch pixel size']
        pickle.dump(newd,open('Tempdir/dict.p','wb'))

        for images in ['Patch I',
                       'Patch Q',
                       'Patch U',
                       'Patch MF I',
                       'Patch MF Q',
                       'Patch MF U',
                       'Patch MF P']:
            p = d[images]
            n = images.replace(' ','_')
            n = n+'.fits'
            p.write('Tempdir/'+n)

        make_tarfile(fname,'Tempdir')
        shutil.rmtree('Tempdir')

    def info(self,include_coords=True,ID=None):
        """
        Returns a simplified dictionary with the essential information of
        the Source.

        Parameters
        ----------
        include_coords : bool, optional
            Whether to include coordinates in the output.
            The default is True.

        ID : int or string, optional
            If not None, adds an identificator to the output dictionary.

        Returns
        -------
        A dictionary containing the essential parameters of the Source.

        """

        odic = {}

        if ID is not None:
            odic['ID'] = ID

        if include_coords:
            odic['RA [deg]']   = self.coord.icrs.ra.deg
            odic['DEC [deg]']  = self.coord.icrs.dec.deg
            odic['GLON [deg]'] = self.coord.galactic.l.deg
            odic['GLAT [deg]'] = self.coord.galactic.b.deg

        # intensity

        odic['I [{0}]'.format(self.unit.to_string())]     = self.I.value
        odic['I err [{0}]'.format(self.unit.to_string())] = self.I.error
        odic['I [Jy]']                                    = self.I.Jy.value
        odic['I err [Jy]']                                = self.I.Jy.error
        odic['I SNR']                                     = self.I.snr

        # polarization

        odic['P [{0}]'.format(self.unit.to_string())]     = self.P.value
        odic['P err [{0}]'.format(self.unit.to_string())] = self.P.error
        odic['P [Jy]']                                    = self.P.Jy.value
        odic['P err [Jy]']                                = self.P.Jy.error
        odic['Angle [deg]']                               = self.angle.value.value
        odic['Angle err [deg]']                           = self.angle.error.value
        odic['P significance']                            = self.P.significance
        odic['Polarization fraction [%]']                 = self.polfrac.value
        odic['Polarization fraction error [%]']           = self.polfrac.error

        # flags

        odic['Extended flag']                             = self.flag_extension
        odic['Photometry flag']                           = self.flag_photometry

        return odic




