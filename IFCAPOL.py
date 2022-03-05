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

# %%  SKY PATCHING --------------------------------

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


# %%  MATCHED FILTER IN I,Q,U OVER A COORDINATE --------------------------------

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
        patches_dict['MF {0}'.format(stokes)] = fp

    return patches_dict


# %%  PATCH OPERATIONS --------------------------------

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

def peak_info(patch,fwhm,
              keyname       = 'I',
              take_positive = False,
              x             = None,
              y             = None,
              clip          = None):

#      Returns a dictionary containing the intensity, rms and position of a
#  local extremum near the center of a patch. If the input keywords X and Y
#  are non None, then the ruotine returns the values for that pixel instead.
#  If take_positive is set to True, then the routine returns either the peak
#  or valley values, depending on which one has the larger absolute value.

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

def peak_fit(patch,
             fwhm = None,
             x    = None,
             y    = None):

#    Fits a 2D Gaussian plus a planar baseline to the central region of a patch.
# If the FWHM or either one of X,Y keywords are not NONE, then the fit is constrained
# to that width and/or position

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


# %%  I,Q,U ON A COORDINATE --------------------------------

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



def get_IQUP(mapas,coord,ihorn,ifreq,
            smoothed_maps = False,
            ideal_beam    = use_ideal_beams,
            pre_projected = False,
            pre_filtered  = False,
            toplot        = False,
            return_abbrv  = False,
            QU_mode       = 'intensity'):

#       If QU_MODE == 'intensity', then Q and U are measured exactly at the
#   position where the maximum of the intensity map has been measured. If not,
#   they are measured on the local maxima (or minima) of the filtered patch

    if pre_filtered:

        patches = {'MF I':Imagen.from_file(patch_name(coord,ihorn,ifreq,0,filtered=True)),
                   'MF Q':Imagen.from_file(patch_name(coord,ihorn,ifreq,1,filtered=True)),
                   'MF U':Imagen.from_file(patch_name(coord,ihorn,ifreq,2,filtered=True)),
                   'FWHM':inst.get_info(ihorn,ifreq,smoothed_maps=smoothed_maps)['Effective map FWHM']}

    else:

        patches = filter_coordinate(mapas,coord,ihorn,ifreq,
                      input_from_file = pre_projected,
                      ideal_beam      = ideal_beam,
                      save_filtered   = True,
                      smoothed_maps   = smoothed_maps,
                      toplot          = False)

    fwhm                     = patches['FWHM'][0]
    out_dict                 = {'coord':coord.galactic}
    out_dict['IHORN']        = ihorn
    out_dict['IFREQ']        = ifreq
    out_dict['Smoothed']     = smoothed_maps
    out_dict['Ideal beams']  = ideal_beam
    out_dict['QU mode']      = QU_mode
    out_dict['Iterative MF'] = iterative_MF

    dictI     = peak_info(patches['MF I'],fwhm,keyname = 'I',take_positive=True)

    if QU_mode == 'intensity':
        dictQ = peak_info(patches['MF Q'],fwhm,keyname = 'Q',x=dictI['I X'],y=dictI['I Y'])
        dictU = peak_info(patches['MF U'],fwhm,keyname = 'U',x=dictI['I X'],y=dictI['I Y'])
    else:
        dictQ = peak_info(patches['MF Q'],fwhm,keyname = 'Q')
        dictU = peak_info(patches['MF U'],fwhm,keyname = 'U')

    out_dict.update(dictI)
    out_dict.update(dictQ)
    out_dict.update(dictU)

    psize    = mapas['HORN 1']['FREQ 1'][0].pixel_size

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

    out_dict['Patch I']    = patches['I']
    out_dict['Patch Q']    = patches['Q']
    out_dict['Patch U']    = patches['U']
    out_dict['Patch MF I'] = patches['MF I']
    out_dict['Patch MF Q'] = patches['MF Q']
    out_dict['Patch MF U'] = patches['MF U']

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
    out_dict['Gaussian fit P significance level'] = 1-np.count_nonzero(fitP.residual>=fitP.amplitude)/fitP.residual.size

    out_dict.update(inst.get_info(ihorn,ifreq,smoothed_maps=smoothed_maps))

    out_dict['UNIT'] = mapas['HORN {0}'.format(ihorn+1)]['FREQ {0}'.format(ifreq+1)].header['TUNIT1']

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

# %%  DETECTORSOURCE CLASS  --------------------------------

class DetectorSource:

    def __init__(self,diccio):
        self.diccio = diccio

    def copy(self):
        x = self.diccio.copy()
        return DetectorSource(x)

# %% -- properties

    @property
    def coord(self):
        return self.diccio['coord']

    @property
    def fwhm(self):
        return self.diccio['Effective map FWHM'].to(u.deg)

    @property
    def area(self):
        return self.diccio['Effective beam area'].to(u.sr)

    @property
    def nu(self):
        return self.diccio['Freq'].to(u.GHz)

    @property
    def int_GHz(self):
        return self.diccio['GHz']

    @property
    def ifreq(self):
        return self.diccio['IFREQ']

    @property
    def ihorn(self):
        return self.diccio['IHORN']

    @property
    def channel(self):
        return inst.ihorn_ifreq_to_detector_number(self.ihorn,self.ifreq)

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
        plt.title('I [mK]')
        plt.subplot(222)
        self.diccio['Patch Q'].draw()
        plt.title('Q [mK]')
        plt.subplot(223)
        self.diccio['Patch U'].draw()
        plt.title('U [mK]')
        if tofile is not None:
            plt.savefig(tofile)

    def mfdraw(self,tofile=None):

        plt.figure(figsize=(16,12))
        plt.subplot(221)
        self.diccio['Patch MF I'].draw()
        plt.title('MF I [mK]')
        plt.subplot(222)
        self.diccio['Patch MF Q'].draw()
        plt.title('MF Q [mK]')
        plt.subplot(223)
        self.diccio['Patch MF U'].draw()
        plt.title('MF U [mK]')
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
    def from_coordinate(self,coordinate,ihorn,ifreq,mapas=None,smoothed_maps=False):
        if mapas is not None:
            d = get_IQUP(mapas,coordinate,
                         ihorn,
                         ifreq,
                         smoothed_maps = smoothed_maps)
        else:
            maps = survey.get_current_maps(smoothed_maps=smoothed_maps)
            d    = get_IQUP(maps,coordinate,
                            ihorn,
                            ifreq,
                            smoothed_maps = smoothed_maps)
        return DetectorSource(d)

    @classmethod
    def from_name(self,name,ihorn,ifreq,mapas=None,smoothed_maps=False):
        coordinate = SkyCoord.from_name(name)
        return self.from_coordinate(coordinate,ihorn,ifreq,mapas=mapas,smoothed_maps=smoothed_maps)

    @property
    def suffix(self):
        return ' CHAN_{0} ({1} GHz)'.format(self.channel,self.int_GHz)

    @property
    def output(self):
        dic = {'RA':self.coord.icrs.ra.deg*u.deg,
               'DEC':self.coord.icrs.dec.deg*u.deg,
               'GLON':self.coord.galactic.l.deg*u.deg,
               'GLAT':self.coord.galactic.b.deg*u.deg,
               'I'+self.suffix:self.I.Jy.value*u.Jy,
               'I err'+self.suffix:self.I.Jy.error*u.Jy,
               'I SNR'+self.suffix:self.I.snr,
               'Q'+self.suffix:self.Q.Jy.value*u.Jy,
               'Q err'+self.suffix:self.Q.Jy.error*u.Jy,
               'U'+self.suffix:self.U.Jy.value*u.Jy,
               'U err'+self.suffix:self.U.Jy.error*u.Jy,
               'P'+self.suffix:self.P.Jy.value*u.Jy,
               'P err'+self.suffix:self.P.Jy.error*u.Jy,
               'P significance'+self.suffix:self.P.significance,
               'pol angle'+self.suffix:self.angle.value,
               'pol angle err'+self.suffix:self.angle.error,
               'pol fraction'+self.suffix:self.polfrac.value,
               'pol fraction err'+self.suffix:self.polfrac.error}
        return QTable([dic])

    def save(self,fname):
        t = self.output
        t.write(fname,overwrite=True)


    def pkl_name(self,ID):

        fname = id_chan_name(ID,self.channel)

        return fname


    def full_save(self,ID=None):

        if ID is None:
            file_in = self.pkl_name(coord2healpix(survey.nside,self.coord))
        else:
            file_in = self.pkl_name(ID)

        d = self.diccio.copy()
        with open(file_in, "wb") as f:
            pickle.dump(len(d), f)
            for value in d:
                pickle.dump(value,f)
            for value in d:
                pickle.dump(d[value],f)

    @classmethod
    def from_pickle(self,fname):
        d = {}
        k = []
        with open(fname, "rb") as f:
            n = pickle.load(f)
            for i in range(n):
                k.append(pickle.load(f))
            for i in range(n):
                d[k[i]] = pickle.load(f)
        return DetectorSource(d)



# %%  SOURCE CLASS  --------------------------------

class Source:

    def __init__(self,detector_list):
        self.detector_list = detector_list

    def copy(self):
        x = self.detector_list.copy()
        return Source(x)

# %% -- individual sources

    def from_chan_name(self,name):
        found = False
        for x in self.detector_list:
            if x.channel == name:
                y     = x.copy()
                found = True
        if found == False:
            print(' --- Warning: channel name not found in Source')
            y = None
        return y

    def from_index(self,index):
        return self.detector_list[index]

# %% -- properties

    @property
    def coord(self):
        return self.detector_list[0].coord

    @property
    def channel_list(self):
        return [x.channel for x in self.detector_list]

# %% -- input

    @classmethod
    def from_coordinate(self,coordinate,maps=None,smoothed=False):

        list_channels = []

        if maps is None:
            mapas = survey.get_current_maps(smoothed_maps=smoothed)
        else:
            mapas = maps.copy()

        for ihorn in range(1,4):
            for ifreq in range(2):
                list_channels.append(DetectorSource.from_coordinate(coordinate,
                                                                    ihorn,
                                                                    ifreq,
                                                                    mapas=mapas,
                                                                    smoothed_maps=smoothed))

        return Source(list_channels)

    @classmethod
    def from_name(self,name,maps=None,smoothed=False):
        coordinate = SkyCoord.from_name(name)
        return Source.from_coordinate(coordinate,maps=maps,smoothed=smoothed)

# %% -- photometry

    def channels(self,freq):
        return [x for x in self.channel_list if self.from_chan_name(x).int_GHz==freq]

    def best_I_chan(self,freq):
        chans = self.channels(freq)
        if len(chans) == 1:
            chname = chans[0]
        else:
            if self.from_chan_name(chans[0]).has_better_SNR(self.from_chan_name(chans[1])):
                chname = chans[0]
            else:
                chname = chans[1]
        return chname

    def best_I(self,freq):
        return self.from_chan_name(self.best_I_chan(freq)).I

    def best_Q(self,freq):
        return self.from_chan_name(self.best_I_chan(freq)).Q

    def best_U(self,freq):
        return self.from_chan_name(self.best_I_chan(freq)).U

    def best_P(self,freq):
        return self.from_chan_name(self.best_I_chan(freq)).P

    def best_angle(self,freq):
        return self.from_chan_name(self.best_I_chan(freq)).angle

    def best_polfrac(self,freq):
        return self.from_chan_name(self.best_I_chan(freq)).polfrac

    @property
    def best_freqs(self):
        return np.array([self.from_chan_name(self.best_I_chan(freq)).nu.value for freq in [11,13,17,19]])

    @property
    def Jyphot_I(self):
        f = [11,13,17,19]
        p =  {'channel'  :[self.best_I_chan(nu) for nu in f],
              'photo'    :[self.best_I(nu).Jy.value for nu in f],
              'photo err':[self.best_I(nu).Jy.error for nu in f],
              'sp index' : self.spectral_index_I}
        return p

    @property
    def Jyphot_P(self):
        f = [11,13,17,19]
        p =  {'channel'  :[self.best_I_chan(nu) for nu in f],
              'photo'    :[self.best_P(nu).Jy.value for nu in f],
              'photo err':[self.best_P(nu).Jy.error for nu in f],
              'sp index' : self.spectral_index_P}
        return p

# %% -- spectral index

    @property
    def fit_spectral_index_I(self):
        x      = self.best_freqs
        y      = np.array([self.best_I(nu).Jy.value for nu in [11,13,17,19]])
        sy     = np.array([self.best_I(nu).Jy.error for nu in [11,13,17,19]])
        w      = 1/sy
        m_init = PowerLaw1D(amplitude=y.mean(),x_0=15,alpha=0)
        m_init.x_0.fixed = True
        fit_pl = fitting.LevMarLSQFitter()
        f      = fit_pl(m_init,x,y,weights=w)
        return f,fit_pl

    @property
    def fit_cc_spectral_index_I(self):
        x      = self.best_freqs
        d      = self.colour_correct_I
        y      = d['photo']
        sy     = d['photo err']
        w      = 1/sy
        m_init = PowerLaw1D(amplitude=y.mean(),x_0=15,alpha=0)
        m_init.x_0.fixed = True
        fit_pl = fitting.LevMarLSQFitter()
        f      = fit_pl(m_init,x,y,weights=w)
        return f,fit_pl

    def spI(self,cc=False):

        if cc:
            f,g = self.fit_cc_spectral_index_I
        else:
            f,g = self.fit_spectral_index_I
        try:
            c   = g.fit_info['param_cov']
            return f.alpha.value,np.sqrt(c[1,1])
        except TypeError:
            return f.alpha.value,np.abs(self.spectral_index_I)

    @property
    def spectral_index_I(self):
        f,g = self.fit_spectral_index_I
        return f.alpha.value

    @property
    def spectral_index_I_err(self):
        f,g = self.fit_spectral_index_I
        try:
            c   = g.fit_info['param_cov']
            return np.sqrt(c[1,1])
        except TypeError:
            return np.abs(self.spectral_index_I)

    @property
    def spectral_cc_index_I(self):
        f,g = self.fit_cc_spectral_index_I
        return f.alpha.value

    @property
    def spectral_cc_index_I_err(self):
        f,g = self.fit_cc_spectral_index_I
        try:
            c   = g.fit_info['param_cov']
            return np.sqrt(c[1,1])
        except TypeError:
            return np.abs(self.spectral_index_I)


    def predict_I_at_freq_Jy(self,freq,m=1000,cc=False):

        if cc:
            f,g = self.fit_cc_spectral_index_I
        else:
            f,g  = self.fit_spectral_index_I

        mean = np.array([f.amplitude.value,f.alpha.value])
        cov  = g.fit_info['param_cov']
        x    = freq.to(u.GHz).value
        p    = np.random.multivariate_normal(mean,cov,m)

        if np.isscalar(x):
            n = 1
        else:
            n = x.size

        y = np.zeros((n,m))

        for i in range(m):
            model  = PowerLaw1D(amplitude=p[i,0],x_0=15,alpha=p[i,1])
            y[:,i] = model(x)

        s     = y.std(axis=1)
        model = PowerLaw1D(amplitude=mean[0],x_0=15,alpha=mean[1])
        y0    = model(x)

        if np.isscalar(x):
            return y0,s[0]
        else:
            return y0,s


    def plot_spectral_shades_I(self,x_vec,n=100,m=1000,alpha=0.25,cc=False):

        x     = np.linspace(x_vec.min(),x_vec.max(),n)*u.GHz

        y0,s  = self.predict_I_at_freq_Jy(x,m=m,cc=cc)
        plt.fill_between(x,y0+s,y0-s,alpha=alpha)

    @property
    def fit_spectral_index_P(self):
        x      = self.best_freqs
        y      = np.array([self.best_P(nu).Jy.value for nu in [11,13,17,19]])
        sy     = np.array([self.best_P(nu).Jy.error for nu in [11,13,17,19]])
        w      = 1/sy
        m_init = PowerLaw1D(amplitude=y.mean(),x_0=15,alpha=0)
        m_init.x_0.fixed = True
        fit_pl = fitting.LevMarLSQFitter()
        f      = fit_pl(m_init,x,y,weights=w)
        return f,fit_pl

    @property
    def fit_cc_spectral_index_P(self):
        x      = self.best_freqs
        d      = self.colour_correct_P
        y      = d['photo']
        sy     = d['photo err']
        w      = 1/sy
        m_init = PowerLaw1D(amplitude=y.mean(),x_0=15,alpha=0)
        m_init.x_0.fixed = True
        fit_pl = fitting.LevMarLSQFitter()
        f      = fit_pl(m_init,x,y,weights=w)
        return f,fit_pl

    @property
    def spectral_index_P(self):
        f,g = self.fit_spectral_index_P
        return f.alpha.value

    @property
    def spectral_cc_index_P(self):
        f,g = self.fit_cc_spectral_index_P
        return f.alpha.value

    @property
    def spectral_index_P_err(self):
        f,g = self.fit_spectral_index_P
        try:
            c   = g.fit_info['param_cov']
            return np.sqrt(c[1,1])
        except TypeError:
            return np.abs(self.spectral_index_P)

    @property
    def spectral_cc_index_P_err(self):
        f,g = self.fit_cc_spectral_index_P
        try:
            c   = g.fit_info['param_cov']
            return np.sqrt(c[1,1])
        except TypeError:
            return np.abs(self.spectral_index_P)

    def spP(self,cc=False):

        if cc:
            f,g = self.fit_cc_spectral_index_P
        else:
            f,g = self.fit_spectral_index_P
        try:
            c   = g.fit_info['param_cov']
            return f.alpha.value,np.sqrt(c[1,1])
        except TypeError:
            return f.alpha.value,np.abs(self.spectral_index_P)

    def predict_P_at_freq_Jy(self,freq,m=1000,cc=False):

        if cc:
            f,g = self.fit_cc_spectral_index_P
        else:
            f,g  = self.fit_spectral_index_P

        mean = np.array([f.amplitude.value,f.alpha.value])
        cov  = g.fit_info['param_cov']
        x    = freq.to(u.GHz).value
        p    = np.random.multivariate_normal(mean,cov,m)

        if np.isscalar(x):
            n = 1
        else:
            n = x.size

        y = np.zeros((n,m))

        for i in range(m):
            model  = PowerLaw1D(amplitude=p[i,0],x_0=15,alpha=p[i,1])
            y[:,i] = model(x)

        s     = y.std(axis=1)
        model = PowerLaw1D(amplitude=mean[0],x_0=15,alpha=mean[1])
        y0    = model(x)

        if np.isscalar(x):
            return y0,s[0]
        else:
            return y0,s

    def plot_spectral_shades_P(self,x_vec,n=100,m=1000,alpha=0.25,cc=False):

        x     = np.linspace(x_vec.min(),x_vec.max(),n)*u.GHz

        y0,s  = self.predict_P_at_freq_Jy(x,m=m,cc=cc)

        plt.fill_between(x,y0+s,y0-s,alpha=alpha)

# %% -- colour corrections

    @property
    def colour_correct_I(self):

        initial_photometry = self.Jyphot_I
        dout              = colour_correction_source(initial_photometry,return_n=True)

        return dout

    @property
    def colour_correct_P(self):

        initial_photometry = self.Jyphot_P
        dout               = colour_correction_source(initial_photometry,return_n=True)

        return dout

# %% -- oputput

    def output(self,source_ID=None):

        dic = {'RA'  :self.coord.icrs.ra.deg*u.deg,
               'DEC' :self.coord.icrs.dec.deg*u.deg,
               'GLON':self.coord.galactic.l.deg*u.deg,
               'GLAT':self.coord.galactic.b.deg*u.deg}

        if source_ID is not None:
            dic['ID'] = source_ID

        for f in [11,13,17,19]:

            dic['Best chan ({0} GHz)'.format(f)]     = self.best_I_chan(f)

            dic['I ({0} GHz)'.format(f)]             = self.best_I(f).Jy.value*u.Jy
            dic['I err ({0} GHz)'.format(f)]         = self.best_I(f).Jy.error*u.Jy
            dic['I SNR ({0} GHz)'.format(f)]         = self.best_I(f).snr

            dic['Q ({0} GHz)'.format(f)]             = self.best_Q(f).Jy.value*u.Jy
            dic['Q err ({0} GHz)'.format(f)]         = self.best_Q(f).Jy.error*u.Jy

            dic['U ({0} GHz)'.format(f)]             = self.best_U(f).Jy.value*u.Jy
            dic['U err ({0} GHz)'.format(f)]         = self.best_U(f).Jy.error*u.Jy

            dic['P ({0} GHz)'.format(f)]             = self.best_P(f).Jy.value*u.Jy
            dic['P err ({0} GHz)'.format(f)]         = self.best_P(f).Jy.error*u.Jy
            dic['P signif ({0} GHz)'.format(f)]      = self.best_P(f).significance

            dic['Pol angle ({0} GHz)'.format(f)]     = self.best_angle(f).value
            dic['Pol angle err ({0} GHz)'.format(f)] = self.best_angle(f).error

            dic['Pol frac ({0} GHz)'.format(f)]      = self.best_polfrac(f).value
            dic['Pol frac err ({0} GHz)'.format(f)]  = self.best_polfrac(f).error

            dic['Spectral index I']                  = self.spectral_index_I
            dic['Spectral index err I']              = self.spectral_index_I_err

            dic['Spectral index P']                  = self.spectral_index_P
            dic['Spectral index err P']              = self.spectral_index_P_err

        return QTable([dic])

    def save(self,fname,source_ID=None):
        t = self.output(source_ID=source_ID)
        t.write(fname,overwrite=True)

    def full_save(self,ID=None):
        for d in self.detector_list:
            d.full_save(ID=ID)
        fname = self.detector_list[0].pkl_name(ID)
        dire  = fname.split('source')[0]
        fname = dire+'source_{0}.fits'.format(ID)
        self.save(fname,source_ID=ID)

    @classmethod
    def from_pickle(self,ID=None):

        l = []

        patchdir  = QUIJOTE_dir+'Source_Extraction/Results/Patches/{0}_{1}/'.format(inst.instrument_name,
                                                                                    inst.instrument_version)
        if not os.path.exists(patchdir):
            os.mkdir(patchdir)

        patchdir += '{0}/'.format(survey.map_version.upper())
        if not os.path.exists(patchdir):
            os.mkdir(patchdir)

        patchdir += 'Source_{0}/'.format(ID)
        if not os.path.exists(patchdir):
            os.mkdir(patchdir)

        for ihorn in range(1,4):
            for ifreq in range(2):
                fname = patchdir + 'source_{0}_{1}.pkl'.format(ID,inst.ihorn_ifreq_to_detector_number(ihorn,ifreq))
                l.append(DetectorSource.from_pickle(fname))

        return Source(l)


# %% -- plotting

    def plot_I(self,
               tofile          = None,
               source_ID       = None,
               newfig          = True,
               plot_powlaw     = False,
               color_corrected = False,
               overlay         = False):

        if newfig:
            plt.figure()

        t  = self.output(source_ID=source_ID)
        x  = np.array([11,13,17,19])
        y  = np.array([t['I ({0} GHz)'.format(n)][0].value for n in x])
        sy = np.array([t['I err ({0} GHz)'.format(n)][0].value for n in x])
        x  = self.best_freqs

        if color_corrected:
            d   = self.colour_correct_I

        if color_corrected:
            if overlay:
                plt.errorbar(x,y,yerr=sy,fmt='o',capsize=2,label='not colour corrected')
                plt.errorbar(x+0.1,d['photo'],yerr=d['photo err'],fmt='o',capsize=2,label='colour corrected')
            else:
                plt.errorbar(x,d['photo'],yerr=d['photo err'],fmt='o',capsize=2,label='colour corrected')
            plt.legend()
        else:
            plt.errorbar(x,y,yerr=sy,fmt='o',capsize=2,label='not colour corrected')

        plt.xlabel('Freq [GHz]')
        plt.ylabel('I [Jy]')
        if source_ID is not None:
            plt.title(source_ID)

        if plot_powlaw:

            if color_corrected:

                if overlay:

                    f,g  = self.fit_spectral_index_I
                    x    = np.linspace(x.min(),x.max(),1000)
                    y    = f(x)
                    plt.plot(x,y)
                    self.plot_spectral_shades_I(x)

                    f,g  = self.fit_cc_spectral_index_I
                    y    = f(x)
                    plt.plot(x,y)
                    self.plot_spectral_shades_I(x,cc=True)

                else:

                    f,g  = self.fit_cc_spectral_index_I
                    x    = np.linspace(x.min(),x.max(),1000)
                    y    = f(x)
                    plt.plot(x,y)
                    self.plot_spectral_shades_I(x,cc=True)

            else:

                f,g  = self.fit_spectral_index_I
                x    = np.linspace(x.min(),x.max(),1000)
                y    = f(x)
                plt.plot(x,y)
                self.plot_spectral_shades_I(x)

        if tofile is not None:
            plt.savefig(tofile)

    def plot_P(self,
               tofile          = None,
               source_ID       = None,
               newfig          = True,
               plot_powlaw     = False,
               color_corrected = False,
               overlay         = False):

        if newfig:
            plt.figure()

        t  = self.output(source_ID=source_ID)
        x  = np.array([11,13,17,19])
        y  = np.array([t['P ({0} GHz)'.format(n)][0].value for n in x])
        sy = np.array([t['P err ({0} GHz)'.format(n)][0].value for n in x])
        x  = self.best_freqs

        if color_corrected:
            d   = self.colour_correct_P

        if color_corrected:
            if overlay:
                plt.errorbar(x,y,yerr=sy,fmt='o',capsize=2,label='not colour corrected')
                plt.errorbar(x+0.1,d['photo'],yerr=d['photo err'],fmt='o',capsize=2,label='colour corrected')
            else:
                plt.errorbar(x,d['photo'],yerr=d['photo err'],fmt='o',capsize=2,label='colour corrected')
            plt.legend()
        else:
            plt.errorbar(x,y,yerr=sy,fmt='o',capsize=2,label='not colour corrected')

        plt.xlabel('Freq [GHz]')
        plt.ylabel('P [Jy]')
        if source_ID is not None:
            plt.title(source_ID)

        if plot_powlaw:

            if color_corrected:

                if overlay:

                    f,g  = self.fit_spectral_index_P
                    x    = np.linspace(x.min(),x.max(),1000)
                    y    = f(x)
                    plt.plot(x,y)
                    self.plot_spectral_shades_P(x)

                    f,g  = self.fit_cc_spectral_index_P
                    y    = f(x)
                    plt.plot(x,y)
                    self.plot_spectral_shades_P(x,cc=True)

                else:

                    f,g  = self.fit_cc_spectral_index_P
                    x    = np.linspace(x.min(),x.max(),1000)
                    y    = f(x)
                    plt.plot(x,y)
                    self.plot_spectral_shades_P(x,cc=True)

            else:

                f,g  = self.fit_spectral_index_P
                x    = np.linspace(x.min(),x.max(),1000)
                y    = f(x)
                plt.plot(x,y)
                self.plot_spectral_shades_P(x)

        if tofile is not None:
            plt.savefig(tofile)

    def plot_Polfrac(self,tofile=None,source_ID=None,newfig=True):

        if newfig:
            plt.figure()

        t  = self.output(source_ID=source_ID)
        x  = np.array([11,13,17,19])
        y  = np.array([t['Pol frac ({0} GHz)'.format(n)][0] for n in x])
        sy = np.array([t['Pol frac err ({0} GHz)'.format(n)][0] for n in x])
        x  = self.best_freqs

        plt.errorbar(x,y,yerr=sy,fmt='o',capsize=2)
        plt.xlabel('Freq [GHz]')
        plt.ylabel('Polarization fraction [%]')
        if source_ID is not None:
            plt.title(source_ID)

        if tofile is not None:
            plt.savefig(tofile)

    def plot_ang(self,tofile=None,source_ID=None,newfig=True):

        if newfig:
            plt.figure()

        t  = self.output(source_ID=source_ID)
        x  = np.array([11,13,17,19])
        y  = np.array([t['Pol angle ({0} GHz)'.format(n)][0].value for n in x])
        sy = np.array([t['Pol angle err ({0} GHz)'.format(n)][0].value for n in x])
        x  = self.best_freqs

        plt.errorbar(x,y,yerr=sy,fmt='o',capsize=2)
        plt.xlabel('Freq [GHz]')
        plt.ylabel('Polarization angle [deg]')
        if source_ID is not None:
            plt.title(source_ID)

        if tofile is not None:
            plt.savefig(tofile)