#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:03:16 2017

@author: herranz
"""

import os
import uuid
import numpy as np
from numpy import dtype
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord
import scipy.stats.mstats as mst
import time
from astropy.units import UnitConversionError
from astropy.visualization import hist
from astropy.units.quantity import Quantity
from astropy import modeling
from astropy.time import Time
from scipy.stats import normaltest,norm
import matplotlib.pyplot as plt
import matplotlib
import re
from sky_images import Imagen
from myutils import coord2vec,fwhm2sigma,sigma2fwhm
from unit_conversions import Kcmb,parse_unit,convert_factor
from mapview import skyview


# %% ------- Example files, used for testing: --------------------------------

testfile  = '/Users/herranz/Trabajo/Planck/Data/PLA/LFI_SkyMap_030-BPassCorrected_0256_R2.01_full.fits'
testfile2 = '/Users/herranz/Dropbox/test.fits'
testfile3 = '/Users/herranz/Dropbox/testmask.fits'
Planck_masks_file = '/Users/herranz/Trabajo/Planck/Non_Thermal_Catalogue/Data/Planck_masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits'

# %% ------- FITSMAP CLASS

class Fitsmap:

#    Class for Healpix maps and masks. Each FITMAPS instance
#     contains n-dimensional maps (DATA) with its corresponding
#     n-dimensional mask, plus a HEADER dictionary and an ordering
#     (RING or NEST) parameter.
#
#     The class is intended to facilitate Healpy programming,
#     in particular allowing for direct map arithmetic operations such
#     as  MAP3 = MAP1 * MAP2 taking into account masks
#     and masked map statistics
#
#     Important notice: the mask convention used in this class is
#     as follows:
#        mask == True   means the pixel has to be masked (equivalent to IDL mask = 0)
#        mask == False  means the pixel is not masked (equivalent to IDL mask=1)


    comment_count  = 0
    obs_frequency  = None
    fwhm_base      = None
    beam_area_base = None

    def __init__(self,data,mask,header,ordering):       # Instantiation
        self.data     = data
        self.mask     = mask
        self.header   = header
        self.ordering = ordering

    def __getitem__(self,sliced):                        # Slicing operation

        n = self.nmaps
        if n == 1:
            datos = self.data.copy()
            mask  = self.mask.copy()
        else:
            if isinstance(sliced,list) and len(sliced)==1:
                datos = self.data[sliced[0],:].copy()
                mask  = self.mask[sliced[0],:].copy()
            elif isinstance(sliced,str):
                elemento = None
                for nombre,valor in self.header.items():
                    if valor == sliced:
                        elemento = nombre
                if elemento is None:
                    print(' --- Warning: field not found in Fitsmap header')
                    datos = self.data.copy()
                    mask  = self.mask.copy()
                else:
                    islice = int(re.findall('\d+',elemento)[0])-1
                    datos  = self.data[islice,:].copy()
                    mask   = self.mask[islice,:].copy()
            else:
                datos = self.data[sliced,:].copy()
                mask  = self.mask[sliced,:].copy()

        ordering = self.check_order()
        header   = self.header.copy()

        if isinstance(sliced,list):
            nombre = [self.columns[slc] for slc in sliced]
            unidad = [self.units[slc] for slc in sliced]
        elif isinstance(sliced,str):
            nombre = self.columns[islice]
            unidad = self.units[islice]
        else:
            nombre = self.columns[sliced]
            unidad = self.units[sliced]

        for i in range(n):
            try:
                del header['TTYPE{0}'.format(i+1)]
            except KeyError:
                pass
            try:
                del header['TUNIT{0}'.format(i+1)]
            except KeyError:
                pass
            try:
                del header['TFORM{0}'.format(i+1)]
            except KeyError:
                pass

        if isinstance(nombre,list):
            for i in range(len(nombre)):
                header['TTYPE{0}'.format(i+1)] = nombre[i]
                header['TUNIT{0}'.format(i+1)] = unidad[i]
            header['TFIELDS'] = len(nombre)
        else:
            header['TTYPE1']  = nombre
            header['TUNIT1']  = unidad
            header['TFIELDS'] = 1

        outmap = Fitsmap(datos,mask,header,ordering)
        outmap.obs_frequency = self.obs_frequency
        if np.isscalar(self.fwhm):
            outmap.fwhm_base          = self.fwhm
            outmap.beam_area_base     = self.beam_area
        else:
            try:
                outmap.fwhm_base      = self.fwhm[sliced]
                outmap.beam_area_base = self.beam_area[sliced]
            except TypeError:
                outmap.fwhm_base      = self.fwhm
                outmap.beam_area_base = self.beam_area

        return outmap

# %% -------   ARITHMETIC AND LOGICAL OPERATORS ---------------------------

    def __neg__(self):                                  # MINUS operator
        if self.ismask:
            r = Fitsmap(np.invert(self.data.copy()),
                        np.invert(self.mask.copy()),
                        self.header.copy(),self.check_order())
        else:
            r = Fitsmap(-1*self.data.copy(),
                        self.mask.copy(),
                        self.header.copy(),self.check_order())
        return r

    def __add__(self,other):                           # SUM operator

        compatible,ctype = self.check_compatibility(other)
        header = self.header.copy()
        if compatible:
            if ctype == 'map':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                datos  = self.data.copy() + other.data.copy()
                mask   = self.mask.copy() + other.mask.copy()
            elif ctype == 'scalar':
                datos    = self.data.copy() + other
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == '1Darray':
                datos    = self.data.copy() + other[0]
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'vector':
                datos = self.data.copy()
                for j in range(self.nmaps):
                    datos[j,:] += other[j]
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'mask':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                datos = self.data.copy()
                mask  = self.mask.copy() + other.mask.copy()
            else:
                datos    = self.data.copy()
                mask     = self.mask.copy()
                ordering = self.check_order()
        else:
            datos    = self.data.copy()
            mask     = self.mask.copy()
            ordering = self.check_order()

        outmap = Fitsmap(datos,mask,header,ordering)
        outmap.obs_frequency  = self.obs_frequency
        outmap.fwhm_base      = self.fwhm_base
        outmap.beam_area_base = self.beam_area_base

        return outmap

    def __radd__(self, other):                         # reverse SUM
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self,other):                           # SUBTRACT operator
        compatible,ctype = self.check_compatibility(other)
        header = self.header.copy()
        if compatible:
            if ctype == 'map':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                datos  = self.data.copy() - other.data.copy()
                mask   = self.mask.copy() + other.mask.copy()
            elif ctype == 'scalar':
                datos    = self.data.copy() - other
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == '1Darray':
                datos    = self.data.copy() - other[0]
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'vector':
                datos = self.data.copy()
                for j in range(self.nmaps):
                    datos[j,:] -= other[j]
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'mask':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                datos = self.data.copy()
                mask  = self.mask.copy() + other.mask.copy()
            else:
                datos    = self.data.copy()
                mask     = self.mask.copy()
                ordering = self.check_order()
        else:
            datos    = self.data.copy()
            mask     = self.mask.copy()
            ordering = self.check_order()

        outmap = Fitsmap(datos,mask,header,ordering)
        outmap.obs_frequency  = self.obs_frequency
        outmap.fwhm_base      = self.fwhm_base
        outmap.beam_area_base = self.beam_area_base

        return outmap

    def __rsub__(self, other):                         # reverse SUBTRACT
        if other == 0:
            return self
        else:
            return self.__sub__(other)

    def __mul__(self,other):                           # MULTIPLY operator
        compatible,ctype = self.check_compatibility(other)
        header = self.header.copy()
        if compatible:
            if ctype == 'map':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                datos  = np.multiply(self.data.copy(),other.data.copy())
                mask   = self.mask.copy() * other.mask.copy()
            elif ctype == 'scalar':
                datos    = self.data.copy() * other
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == '1Darray':
                datos    = self.data.copy() * other
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'vector':
                datos = self.data.copy()
                for j in range(self.nmaps):
                    datos[j,:] = other[j]*datos[j,:]
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'mask':
                self.to_ring()
                other.to_ring()
                ordering = 'RING'
                mask  = self.mask.copy() * other.mask.copy()
                if self.ismask and not other.ismask:
                    datos = other.data.copy()
                elif self.ismask and other.ismask:
                    datos = mask
                else:
                    datos = self.data.copy()
            else:
                datos    = self.data.copy()
                mask     = self.mask.copy()
                ordering = self.check_order()
        else:
            datos    = self.data.copy()
            mask     = self.mask.copy()
            ordering = self.check_order()

        outmap = Fitsmap(datos,mask,header,ordering)
        outmap.obs_frequency  = self.obs_frequency
        outmap.fwhm_base      = self.fwhm_base
        outmap.beam_area_base = self.beam_area_base

        return outmap

    def __rmul__(self, other):                        # reverse MULTIPLY
        if other == 0:
            return self
        else:
            return self.__mul__(other)

    def __pow__(self,p):                              # POWER operator
        compatible,ctype = self.check_compatibility(p)
        header = self.header.copy()
        if compatible:
            if ctype == 'map':
                self.to_ring()
                p.to_ring()
                ordering = 'RING'
                datos  = np.power(self.data.copy(),p.data.copy())
                mask   = self.mask.copy() + p.mask.copy()
            elif ctype == 'scalar':
                datos    = np.power(self.data.copy(),p)
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == '1Darray':
                datos    = np.power(self.data.copy(),p[0])
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'vector':
                datos = self.data.copy()
                for j in range(self.nmaps):
                    datos[j,:] = np.power(datos[j,:],p[j])
                mask     = self.mask.copy()
                ordering = self.check_order()
            elif ctype == 'mask':
                self.to_ring()
                p.to_ring()
                ordering = 'RING'
                mask  = self.mask.copy() + p.mask.copy()
                if self.ismask:
                    datos = mask
                else:
                    datos = self.data.copy()
            else:
                datos    = self.data.copy()
                mask     = self.mask.copy()
                ordering = self.check_order()
        else:
            datos    = self.data.copy()
            mask     = self.mask.copy()
            ordering = self.check_order()

        outmap = Fitsmap(datos,mask,header,ordering)
        outmap.obs_frequency  = self.obs_frequency
        outmap.fwhm_base      = self.fwhm_base
        outmap.beam_area_base = self.beam_area_base

        return outmap

    def check_compatibility(self,other):
        """
        Checks if SELF is compatible with OTHER for arithmetic operations

        Parameters
        ----------
        other : TYPE
            A python object.

        Returns
        -------
        compatibility : bool
            Whether the two arguments are compatible or not.
        ctype : str
            A string describing the kind of compatible object.

        """

        compatibility = False
        ctype         = None

        comp_type = type(other) == type(self)
        if comp_type:
            if self.data.shape == other.data.shape:
                compatibility = True
                if self.ismask or other.ismask:
                    ctype = 'mask'
                else:
                    ctype = 'map'
            else:
                compatibility = False
                print(' --- Warning: fits data dimensions do not match')
        else:
            if (type(other)==int) or (type(other)==float) or (type(other)==complex):
                compatibility = True
                ctype = 'scalar'
            if np.size(other)==1 and ctype!='scalar':
                try:
                    dt = other.dtype
                    if dt == dtype('float64') or dt == dtype('int64') or dt == dtype('complex128'):
                        compatibility = True
                        ctype = '1Darray'
                except AttributeError:
                    pass
            if isinstance(other,np.ndarray) and np.size(other)>1 and np.size(other)==self.nmaps:
                compatibility = True
                ctype = 'vector'

        if not compatibility:
            print(' --- Warning: unable to find object compatibility')
            ctype = 'none'

        return compatibility,ctype


# %% -------   COPY METHOD -------------------------------------------------

    def check_order(self):
        """
        Returns the ordering scheme of the `Fitsmap`.

        Returns
        -------
        order : string
            "RING" or "NEST".

        """
        if self.ordering.upper() == 'RING':
            order = 'RING'
        else:
            order = 'NEST'
        return order

    def copy(self):
        """
        Creates a copy of the `Fitsmap` instance.

        Returns
        -------
        copia : `Fitsmap`
            Copy of the `Fitsmap`.

        """

        copia = Fitsmap(self.data.copy(),self.mask.copy(),
                        self.header.copy(),self.check_order())

        copia.comment_count  = self.comment_count
        copia.obs_frequency  = self.obs_frequency
        copia.fwhm_base      = self.fwhm
        copia.beam_area_base = self.beam_area

        return copia

# %% -------   MASKS -------------------------------------------------------

    def mask_value(self,value):
        """
        This method masks those pixels whose value is equal to a given
        input.

        Parameters
        ----------
        value : float
            Value to be masked.

        Returns
        -------
        None.

        """
        self.mask = self.data == value

    def masked_data(self,i=-1):
        """
        Returns the mask as a data array.

        Parameters
        ----------
        i : int, optional
            Dimension of the `Fitsmap,data` along which the mask is returned.
            The default is -1, which means all the dimensions
            are considered and the output will be stored in a list of
            dimension `Fitsmap.nmaps`.

        Returns
        -------
        v : array or list of array
            DESCRIPTION.

        """
        if (self.nmaps == 1) or (i<0):
            v = np.ma.array(self.data.copy(),mask=self.mask.copy())
        else:
            v = np.ma.array(self.data[i,:].copy(),mask=self.mask[i,:].copy())
        return v

    def add_mask(self,mask):
        """
        This method adds a mask map (in HEALPix format) to the `Fitsmap`. The
        sum follows the boolean logic rules.

        Parameters
        ----------
        mask : array
            Mask values, as HEALPix map.

        Returns
        -------
        None.

        """
        self.mask = self.mask + mask

    def mask_band(self,band_size):
        """
        Masks a band of a given size around the Equator.

        Parameters
        ----------
        band_size : float or `~astropy.units.quantity.Quantity`
            Half-size of the band. The band is assumed to be symmetrical
            around the equator (`band_size` counts in both directions).
            If a float is provided, the method assumes the band size is
            expressed in degrees. If not, the argument must be an
            angular `astropy.Quantity`.

        Returns
        -------
        None.

        """

        if isinstance(band_size,u.quantity.Quantity):
            t1 = band_size
        else: # assume the value is in degrees
            t1 = band_size*u.deg

        tuno = np.pi/2-t1.to(u.rad).value
        tdos = np.pi/2+t1.to(u.rad).value

        isnest = False
        if self.ordering.upper() == 'NEST':
            isnest = True

        ipix = hp.query_strip(self.nside,tuno,tdos,nest=isnest)
        n    = self.nmaps

        if n==1:
            self.mask[ipix] = True
            if self.ismask:
                self.data[ipix] = True
        else:
            for i in range(n):
                self.mask[i,ipix] = True
                if self.ismask:
                    self.data[i,ipix] = True

    def grow_mask_1pix(self):
        """
        This method grows the mask of the `Fitsmap` in one pixel.

        Returns
        -------
        None.

        """
        if self.nmaps == 1:
            mask   = self.mask.copy()
        else:
            mask   = self.mask[0,:].copy()
        if self.ordering == 'NEST':
            nested = True
        else:
            nested = False
        listpix = [i for i in range(self.npix) if not mask[i]]
        for pixel in listpix:
            ipix = hp.get_all_neighbours(self.nside,pixel,nest=nested)
            n    = np.count_nonzero(mask[ipix])
            if n>0:
                if self.nmaps == 1:
                    self.mask[pixel] = True
                else:
                    self.mask[:,pixel] = True

    def grow_mask_radius(self,radius):
        """
        Grows the mask by a certain radius.

        Parameters
        ----------
        radius : `~astropy.units.quantity.Quantity`
            Growth radius.

        Returns
        -------
        None.

        """
        if self.nmaps == 1:
            mask   = self.mask.copy()
        else:
            mask   = self.mask[0,:].copy()
        if self.ordering == 'NEST':
            nested = True
        else:
            nested = False
        listpix = [i for i in range(self.npix) if not mask[i]]
        for pixel in listpix:
            v    = hp.pix2vec(self.nside,pixel,nest=nested)
            ipix = hp.query_disc(self.nside,v,radius.to(u.rad).value,
                                 inclusive=True,
                                 nest=nested)
            n    = np.count_nonzero(mask[ipix])
            if n>0:
                if self.nmaps == 1:
                    self.mask[pixel] = True
                else:
                    self.mask[:,pixel] = True

    def to_mask(self):
        """
        Transfer the mask values to the data extension of the `Fitsmap`.

        Returns
        -------
        None.

        """
        self.data = self.mask.copy()

    @property
    def mask_area(self):
        """
        Returns the area of the sky that is covered by the mask.

        Returns
        -------
        `~astropy.units.quantity.Quantity`
            Area of the masked sky

        """
        return np.count_nonzero(self.mask)*self.pixel_area

    @property
    def unmasked_area(self):
        """
        Returns the area of the sky that is not covered by the mask.

        Returns
        -------
        `~astropy.units.quantity.Quantity`
            Area of the non-masked sky

        """
        return 4.0*np.pi*u.sr-self.mask_area

    @property
    def masked_fraction(self):
        """
        Returns the fraction of pixels that are masked.

        Returns
        -------
        float
            Fraction of masked pixels.

        """
        return np.count_nonzero(self.mask)/self.npix

    @property
    def unmasked_fraction(self):
        """
        Returns the fraction of pixels that are not masked.

        Returns
        -------
        float
            Fraction of non-masked pixels.

        """
        return 1.0-self.masked_fraction

# %% -------   SMOOTHING ---------------------------------------------------

    def smooth(self,fwhm):
        """
        Smooths the map with a Gaussian symmetric beam.

        Parameters
        ----------
        fwhm : `~astropy.units.quantity.Quantity`
            The Full Width at Half Maximum of the Gaussian beam.

        Returns
        -------
        mapa : `Fitsmap`
            Smoothed map.

        """

        mapa = self.copy()
        mapa.to_ring()
        if mapa.nmaps == 1:
            x = mapa.data
            x = hp.smoothing(x,fwhm=fwhm.to(u.rad).value,pol=False)
            mapa.data = x
            mapa.fwhm_base = np.sqrt((mapa.fwhm_base**2+fwhm**2).si)
        elif mapa.nmaps == 3:
            x = [mapa.data[0,:],mapa.data[1,:],mapa.data[2,:]]
            x = hp.smoothing(x,fwhm=fwhm.to(u.rad).value,pol=True)
            mapa.data = np.array(x)
            mapa.fwhm_base = np.sqrt((mapa.fwhm_base**2+fwhm**2).si)
        else:
            for i in range(mapa.nmaps):
                x = mapa.data[i,:]
                x = hp.smoothing(x,fwhm=fwhm.to(u.rad).value,pol=False)
                mapa.data[i,:] = x
                mapa[i,:].fwhm_base = np.sqrt((mapa[i,:].fwhm_base**2+fwhm**2).si)
        mapa.ordering = 'RING'
        return mapa

# %% -------   STATISTICS ---------------------------------------------------

    def ngood(self,i=-1):
        """
        Returns the number of "good" pixels of the data
        contained in the `Fitsmap`. Good pixels are those non marked
        as BADVAL in the HEALPix convention.

        Parameters
        ----------
        i : int, optional
            Dimension of the `Fitsmap,data` along which the normality test
            is performed. The default is -1, which means all the dimensions
            are considered and the output will be stored in a list of
            dimension `Fitsmap.nmaps`.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        r : int or array
            The number of good pixels.

        """

        if i>=0:
            a  = self.masked_data(i)
            x  = a.data
            m1 = np.invert(a.mask)
            m2 = hp.mask_good(x,self.badval)
            r  = np.count_nonzero(m1*m2)
        else:
            r = []
            for k in range(self.nmaps):
                a  = self.masked_data(k)
                x  = a.data
                m1 = np.invert(a.mask)
                m2 = hp.mask_good(x,self.badval)
                r.append(np.count_nonzero(m1*m2))
            r = np.array(r)
        return r

    def maxval(self,i=-1):
        """
        Returns the maximum value of the data
        contained in the `Fitsmap`.

        Parameters
        ----------
        i : int, optional
            Dimension of the `Fitsmap,data` along which the normality test
            is performed. The default is -1, which means all the dimensions
            are considered and the output will be stored in a list of
            dimension `Fitsmap.nmaps`.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        r : float or array
            The maximum value.

        """

        if i>=0:
            a = self.masked_data(i)
            r = a.max()
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(a.max())
            r = np.array(r)
        return r

    def minval(self,i=-1):
        """
        Returns the minimum value of the data
        contained in the `Fitsmap`.

        Parameters
        ----------
        i : int, optional
            Dimension of the `Fitsmap,data` along which the normality test
            is performed. The default is -1, which means all the dimensions
            are considered and the output will be stored in a list of
            dimension `Fitsmap.nmaps`.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        r : float or array
            The minimum value.

        """

        if i>=0:
            a = self.masked_data(i)
            r = a.min()
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(a.min())
            r = np.array(r)
        return r

    def mean(self,i=-1):
        """
        Returns the mean of the data
        contained in the `Fitsmap`.

        Parameters
        ----------
        i : int, optional
            Dimension of the `Fitsmap,data` along which the normality test
            is performed. The default is -1, which means all the dimensions
            are considered and the output will be stored in a list of
            dimension `Fitsmap.nmaps`.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        r : float or array
            The mean.

        """

        if i>=0:
            a = self.masked_data(i)
            r = a.mean()
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(a.mean())
            r = np.array(r)
        return r

    def std(self,i=-1):
        """
        Returns the standard deviation of the data
        contained in the `Fitsmap`.

        Parameters
        ----------
        i : int, optional
            Dimension of the `Fitsmap,data` along which the normality test
            is performed. The default is -1, which means all the dimensions
            are considered and the output will be stored in a list of
            dimension `Fitsmap.nmaps`.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        r : float or array
            The standard deviation.

        """

        if i>=0:
            a = self.masked_data(i)
            r = a.std()
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(a.std())
            r = np.array(r)
        return r

    def skew(self,i=-1,**kwargs):
        """
        Returns the skewness of the data
        contained in the `Fitsmap`.

        Parameters
        ----------
        i : int, optional
            Dimension of the `Fitsmap,data` along which the normality test
            is performed. The default is -1, which means all the dimensions
            are considered and the output will be stored in a list of
            dimension `Fitsmap.nmaps`.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        r : float or array
            The skewness.

        """

        if i>=0:
            a = self.masked_data(i)
            r = np.asscalar(mst.skew(a,**kwargs).data)
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(np.asscalar(mst.skew(a,**kwargs).data))
            r = np.array(r)
        return r

    def kurtosis(self,i=-1,**kwargs):
        """
        Returns the kurtosis of the data
        contained in the `Fitsmap`.

        Parameters
        ----------
        i : int, optional
            Dimension of the `Fitsmap,data` along which the normality test
            is performed. The default is -1, which means all the dimensions
            are considered and the output will be stored in a list of
            dimension `Fitsmap.nmaps`.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        r : float or array
            The kurtosis.

        """

        if i>=0:
            a = self.masked_data(i)
            r = mst.kurtosis(a,**kwargs)
        else:
            r = []
            for k in range(self.nmaps):
                a = self.masked_data(k)
                r.append(mst.kurtosis(a,**kwargs))
            r = np.array(r)
        return r

    def statistics(self,i=-1):
        """
        Returns a dictionary of statistical descriptors for the data
        contained in the `Fitsmap`.

        Parameters
        ----------
        i : int, optional
            Dimension of the `Fitsmap,data` along which the normality test
            is performed. The default is -1, which means all the dimensions
            are considered and the output will be stored in a list of
            dimension `Fitsmap.nmaps`.

        Returns
        -------
        dict
            A dictionary containing:
                * 'mean': mean of the data
                * 'std': standard deviation of the data
                * 'min': minimum value
                * 'max': maximum value
                * 'skew': skewness
                * 'kurt': kurtosis
                * 'ngood': number of good samples

        """
        return {'mean':self.mean(i=i),
                'std':self.std(i=i),
                'max':self.maxval(i=i),
                'min':self.minval(i=i),
                'skew':self.skew(i=i),
                'kurt':self.kurtosis(i=i),
                'ngood':self.ngood(i=i)}

    def test_normal(self,i=-1,toplot=False,tofile=None,threshold=1.e-3):
        """
        Normality test of the non-masked elements of the `Fitsmap`
        object. This method invokes the normaltest function from scipy [1]_.
        The function tests the null hypothesis that a sample comes
        from a normal distribution.  It is based on D'Agostino and
        Pearson's [2]_, [3]_ test that combines skew and kurtosis to
        produce an omnibus test of normality.

        Parameters
        ----------
        i : int, optional
            Dimension of the `Fitsmap,data` along which the normality test
            is performed. The default is -1, which means all the dimensions
            are considered and the output will be stored in a list of
            dimension `Fitsmap.nmaps`.
        toplot : bool, optional
            If *True*, a plot with the results is generated. The default is False.
        tofile : bool, optional
             Name of the file where the results will be stored. The default is None.
        threshold : float, optional
            Threshold for the p-value. If the p-value is lesser than the
            specified threshold, the test returns *False*
            (the distribution is not considered to be normal). The default is 1.e-3.

        Returns
        -------
        dict
            A dictionary containing:
                * 'p-value': A 2-sided chi squared probability for the hypothesis test.
                * 'is_normal': A boolean answer to the question about whether or not the distribution is normal.
                * 'statistics': ``s^2 + k^2``, where ``s`` is the z-score returned by `skewtest` and ``k`` is the z-score returned by `kurtosistest` (see the documentation for `scipy.stats.normaltest`



        References
        ----------
        .. [1] Virtanen, P. et al. (2020), "SciPy 1.0: Fundamental Algorithms
               for Scientific Computing in Python", Nature Methods, 17(3), 261-272

        .. [2] D'Agostino, R. B. (1971), "An omnibus test of normality for
               moderate and large sample size", Biometrika, 58, 341-348

        .. [3] D'Agostino, R. and Pearson, E. S. (1973), "Tests for departure from
               normality", Biometrika, 60, 613-622


        """


        if self.nmaps==1:
            x = self.data.copy()
            m = self.mask.copy()
            y = x[np.invert(m)]
            statistic,pvalue = normaltest(y,nan_policy='omit')
            if pvalue<threshold:
                isnormal = False
            else:
                isnormal = True
        elif isinstance(i,int):
            x = self.data[i,:].copy()
            m = self.mask[i,:].copy()
            y = x[np.invert(m)]
            statistic,pvalue = normaltest(y,nan_policy='omit')
            if pvalue<threshold:
                isnormal = False
            else:
                isnormal = True
        elif isinstance(i,np.ndarray):
            toplot = False
            tofile = None
            statistic = []
            pvalue    = []
            isnormal  = []
            for j in range(self.nmaps):
                x = self.data[j,:].copy()
                m = self.mask[j,:].copy()
                y = x[np.invert(m)]
                stat,p = normaltest(y,nan_policy='omit')
                statistic.append(stat)
                pvalue.append(p)
                if p<threshold:
                    isnormal.append(False)
                else:
                    isnormal.append(True)

        if tofile is not None:
            toplot = True

        if toplot:
            columns = self.columns
            plt.figure()
            h = hist(y,bins='knuth',
                     histtype='stepfilled',
                     alpha=0.2,
                     normed=True,
                     color='blue',
                     label=columns[i])
            lax = h[1]
            m   = y.mean()
            s   = y.std()
            lay = norm.pdf(lax,m,s)
            plt.plot(lax,lay,'r')
            plt.xlabel(columns[i])
            plt.ylabel('normalized histogram')
            if tofile is not None:
                plt.savefig(tofile)


        return {'statistics':statistic,'p-value':pvalue,'is_normal':isnormal}


# %% -------   PIXEL OPERATIONS AND COORDINATES ----------------------------

    def to_ring(self):
        """
        Converts the `Fitsmap`to the HEALPix ordering scheme RING.

        Returns
        -------
        None.

        """
        if self.ordering.upper() == 'NEST':
            self.ordering = 'RING'
            self.data     = hp.reorder(self.data,n2r=True)
            self.mask     = hp.reorder(self.mask,n2r=True)
            self.header['ORDERING'] = 'RING'

    def to_nest(self):
        """
        Converts the `Fitsmap`to the HEALPix ordering scheme NEST.

        Returns
        -------
        None.

        """
        if self.ordering.upper() == 'RING':
            self.ordering = 'NEST'
            self.data     = hp.reorder(self.data,r2n=True)
            self.mask     = hp.reorder(self.mask,r2n=True)
            self.header['ORDERING'] = 'NEST'

    def pixel_to_coordinates(self,i):
        """
        Returs the sky coordinate(s) corresponding to the HEALPix pixel
        index or indexes `i`. The index is referred
        to the ordering scheme of the `Fitsmap` object.

        Parameters
        ----------
        i : int
            The HEALPix index.

        Returns
        -------
        `~astropy.coordinates.SkyCoord`
            Sky coordinate or array of sky coordinates.

        """
        cframe = self.coordsys
        if cframe == 'G':
            framsys = 'galactic'
        if cframe == 'C':
            framsys = 'icrs'
        if cframe == 'E':
            framsys = 'geocentrictrueecliptic'
        if self.ordering.upper() == 'NEST':
            innest = True
        else:
            innest = False

        lon,lat = hp.pix2ang(self.nside,i,nest=innest,lonlat=True)

        return SkyCoord(lon*u.deg,lat*u.deg,frame=framsys)

    def coordinates_to_pixel(self,coord):
        """
        Returns the HEALPix pixel index(es) corresponding to a given
        sky coordinate or set of sky coordinates. The index is referred
        to the ordering scheme of the `Fitsmap` object.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            Coordinate(s) to be converted to pixel index.

        Returns
        -------
        pix : int
            Pixel index, or array of pixel indexes.

        """
        cframe = self.coordsys
        if cframe == 'C':
            lon = coord.icrs.ra.deg
            lat = coord.icrs.dec.deg
        elif cframe == 'E':
            lon = coord.geocentrictrueecliptic.lon.deg
            lat = coord.geocentrictrueecliptic.lat.deg
        else:
            lon = coord.galactic.l.deg
            lat = coord.galactic.b.deg
        nested = False
        if self.ordering.upper() == 'NEST':
            nested = True
        pix = hp.ang2pix(self.nside,lon,lat,nest=nested,lonlat=True)
        return pix

    def coordinates_to_vector(self,coord):
        """
        Returns the vector(s) (HEALPix convention) corresponding to a sky
        coordinate or set of sky coordinates.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            Coordinate(s) to be converted to vectors.

        Returns
        -------
        x, y, z floats, scalar or array-like
            Vectors (HEALPix convention)
            corresponding to the input coordinates.

        """
        if self.ordering.upper() == 'NEST':
            nested = True
        else:
            nested = False
        return hp.pix2vec(self.nside,self.coordinates_to_pixel(coord),
                          nest=nested)

    def ud_grade(self,nside_new):
        """
        Upgrade or degrade resolution of a `Fitsmap`

        Parameters
        ----------
        nside_new : int
            New (upgraded or degraded) nside parameter.

        Returns
        -------
        outmap : Fitsmap
            The upgraded or degraded `Fitsmap`.

        """

        nmap = hp.ud_grade(self.data.copy(),nside_out=nside_new,
                           order_in=self.check_order(),
                           order_out=self.check_order())
        nmask = hp.ud_grade(self.mask.copy(),nside_out=nside_new,
                            order_in=self.check_order(),
                            order_out=self.check_order())
        self.header['NSIDE'] = nside_new
        outmap = Fitsmap(nmap,nmask,self.header.copy(),self.check_order())
        outmap.obs_frequency  = self.obs_frequency
        outmap.fwhm_base      = self.fwhm_base
        outmap.beam_area_base = self.beam_area_base

        return outmap

    def disc_around_coordinates(self,coord,radius):
        """
        Returns pixels whose centers lie within the disk defined by a coordinate
        and a disk radius.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            Coordinate of the center of the circle.
        radius : `~astropy.units.quantity.Quantity`
            Radius of the circle.

        Returns
        -------
        int, array
            The pixels which lie within the given disk.

        """

        pix = self.coordinates_to_pixel(coord)
        nested = False
        if self.ordering.upper() == 'NEST':
            nested = True
        vec = hp.pix2vec(self.nside,pix,nest=nested)
        return hp.query_disc(self.nside,vec,radius.to(u.rad).value,
                             inclusive=True,nest=nested)

    def fraction_masked_disk(self,coord,radius):
        """
        Return the fraction of masked pixels inide a circle around a given
        coordinate.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            Coordinate of the center of the circle.
        radius : `~astropy.units.quantity.Quantity`
            Radius of the circle.

        Returns
        -------
        float
            Fraction (between 0 and 1) of pixels that are masked.

        """
        n_masked = 0
        n_total  = 0
        for i in range(self.nmaps):
            temp     = self[i]
            inx      = temp.disc_around_coordinates(coord,radius)
            n_masked += np.count_nonzero(temp.mask[inx])
            n_total  += temp.data[inx].size
        return float(n_masked)/float(n_total)

    @property
    def pixel_area(self):
        """
        Pixel area. This method invokes `healpy.pixelfunc.nside2pixarea`.

        Returns
        -------
        `~astropy.units.quantity.Quantity`
            The pixel area, in sr.

        """
        return hp.nside2pixarea(self.nside)*u.sr

    @property
    def pixel_window_function(self):
        """
        Return the pixel window function for the given nside. This method
        invokes `healpy.sphtfunc.pixwin` only for temperature. Polarization
        is not included.

        Returns
        -------
        array
            The temperature pixel window function.

        """
        return hp.pixwin(self.nside)

    def pixel_beam_function(self,theta):
        return hp.bl2beam(self.pixel_window_function,theta)

    @property
    def pixel_fwhm(self):
        """
        Returns the FWHM of the best Gaussian fit to the pixel window function.

        Returns
        -------
        `~astropy.units.quantity.Quantity`
            The pixel FWHM, in arcmin.

        """

        theta_max = 5*hp.nside2resol(self.nside)
        thetarr   = np.linspace(-theta_max,theta_max,1000)
        pixbeam   = self.pixel_beam_function(thetarr)
        fitter    = modeling.fitting.LevMarLSQFitter()
        model     = modeling.models.Gaussian1D(amplitude=pixbeam.max(),
                                               mean=0,
                                               stddev=hp.nside2resol(self.nside))
        fmodel    = fitter(model,thetarr,pixbeam)
        sigma     = fmodel.stddev.value * u.rad

        return sigma2fwhm*sigma.to(u.arcmin)

    @property
    def pixel_sigma(self):
        """
        Returns the width of the best Gaussian fit to the pixel window function.

        Returns
        -------
        `~astropy.units.quantity.Quantity`
            The pixel function width, in arcmin.

        """
        return fwhm2sigma*self.pixel_fwhm

    @property
    def pixel_size(self):
        """
        Alias for `Fitsmap.resolution`

        Returns
        -------
        float
            The pixel size, in arcmin.

        """
        return self.resolution

    @property
    def pixsize(self):
        """
        Alias for `Fitsmap.resolution`

        Returns
        -------
        float
            The pixel size, in arcmin.

        """
        return self.resolution



# %% -------   VISUALIZATION  -----------------------------------------------

    def moll(self,i=0,tofile=None,norm='hist',**kwargs):
        """
        Plot one of the data elements of the `Fitsmap` in Mollweide projection.

        Parameters
        ----------
        i : int, optional
            The index value of the data array to be plotted. The default is 0.
        tofile : string, optional
            File name where the plot will be written. If None, nothing is
            written to file. The default is None.
        norm : {‘hist’, ‘log’, None}, optional
            Color normalization, hist= histogram equalized color mapping,
            log= logarithmic color mapping. The default is 'hist'.
        **kwargs : TYPE
            See `healpy.visufunc.mollview` for additional kwargs.

        Returns
        -------
        None.

        """

        c = self.columns
        n = self.nmaps
        if n > 1:
            x = self.data[i,:].copy()
            m = self.mask[i,:].copy()
        else:
            x = self.data.copy()
            m = self.mask.copy()

        if np.count_nonzero(m)>0:
            x[m] = hp.UNSEEN

        hp.mollview(x,title=c[i]+' ['+self.units[i]+']',norm=norm,**kwargs)

        if tofile is not None:
            plt.savefig(tofile)

    def skyview(self,coord,i=0,tofile=None,title=None,zoom_size=4*u.deg):
        """
        Invokes mapview.skyview to visualize the sky map around a given
        coordinate in spherical view. The visualization is made in the equatorial
        coordinate system, using the input coordinate as center of the projection.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            Sky coordinate where the plot will be centered. It is also used
            as center of the zooming region.
        i : int, optional
            The `Fitsmap` data extension. The default is 0.
        tofile : srt or None, optional
            File where the plot is saved. The default is None.
        title : str or None, optional
            Figure title. The default is None.
        zoom_size : `~astropy.units.quantity.Quantity`
            The size of the zoom. The default is 4*u.deg.


        Returns
        -------
        None.

        """

        n = self.nmaps

        if n > 1:
            x = self.data[i,:].copy()
        else:
            x = self.data.copy()

        coordsys = self.header['COORDSYS']

        skyview(x,
                coord,
                coordsys  = coordsys,
                zoom_size = zoom_size,
                title     = title,
                tofile    = tofile)




# %% -------   PATCHING  ----------------------------------------------------

    def patch(self,coord,
              npix           = None,
              psi_deg        = 0.0,
              deltatheta_deg = 14.658,
              resampling     = 1,
              toplot         = False,
              tofile         = None):
        """
        Projects a flat patch around a given sky coordinate.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            Sky coordinate of the center of the flat patch.
        npix : int, optional
            Size (number of pixels per side) of the flat patch.
            If None, the size is automatically set to
            `Fitsmap.nside` / 4.  The default is None.
        psi_deg : float, optional
            Rotation angle of the patch, in degrees. See the `rot` argument
            of `healpy.visufunc.gnomview` for more information. The default is 0.0.
        deltatheta_deg : float, optional
            The angular size of the patch, in degrees. The default is 14.658.
        resampling : integer, optional
            Resampling factor to be applied. On output, each pixel is subdivided
            into (resampling x resampling) sub-pixels. The default is 1
            (no resampling).
        toplot : bool, optional
            If True, the patch is plotted as a figure in a new window. The
            default is False.
        tofile : str or None, optional
            If a string is provided, it is the name of a file to which the
            patch is written. The default is None.

        Returns
        -------
        imag : `sky_images.Imagen`
            A flat patch centered around the given coordinate.

        """

        matplotlib.use('Agg', force=True)

        if not toplot:
            plt.ioff()

        if self.ordering.upper() == 'NEST':
            innest = True
        else:
            innest = False

        if resampling == 1:
            mapa  = self.copy()
        else:
            nside = np.int(resampling)*(self.nside)
            mapa  = self.ud_grade(nside)

        pixel      = mapa.coordinates_to_pixel(coord)
        theta,phi  = hp.pix2ang(mapa.nside,pixel,nest=innest)
        theta      = np.rad2deg(theta)
        phi        = np.rad2deg(phi)

        if npix is None:
            xsize = mapa.nside//4
        else:
            xsize = npix*resampling

        if mapa.fraction_masked_disk(coord,deltatheta_deg*u.deg/np.sqrt(2))>=0.9:

            img  = Imagen(np.zeros((npix,npix)),
                          np.array([theta,phi]),
                          np.array([npix,npix]),
                          deltatheta_deg*u.deg/(npix))

        else:

#            psiz_deg = deltatheta_deg/(self.nside/4)
#            psiz_min = 60.0*psiz_deg
#            objeto   = hp.gnomview(self.data,rot=(phi,90.0-theta,psi_deg),
#                                   reso=psiz_min, xsize=self.nside//4,
#                                   flip="astro",coord=self.coordsys,
#                                   return_projected_map=True)

            psiz_deg = deltatheta_deg/(xsize)
            psiz_min = 60.0*psiz_deg
            objeto   = hp.gnomview(self.data,
#                                   no_plot=True,
                                   rot=(phi,90.0-theta,psi_deg),
                                   reso=psiz_min, xsize=xsize,
                                   flip="astro",coord=self.coordsys,
                                   return_projected_map=True)

            plt.close()
            datos    = objeto.data
            datos    = np.flipud(datos)
            s = datos.shape
            N1 = s[0]
            N2 = s[1]

            img  = Imagen(datos,
                          np.array([theta,phi]),
                          np.array([N1,N2]),
                          deltatheta_deg/N1*u.deg)

            if resampling > 1:

                img = img.downsample(factor=resampling,
                                     func=np.mean)

        if xsize//resampling < img.lsize:
            imag = img.stamp_central_region(xsize)
        else:
            imag = img

        if toplot:
            imag.draw(newfig=True)

        imag.image_header = None
        h = imag.header
        del(h)

        imag.image_header.update(self.header)

        try:
            imag.image_header.update({('FWHM',self.fwhm.to(u.arcmin).value,'[arcmin] beam FWHM')})
            imag.image_header.update({('BEAM',self.beam_area.to(u.sr).value,'[steradian] beam area')})
            imag.image_header.update({('FREQ',self.obs_frequency.to(u.GHz).value,'[GHz] observation frequency')})
            imag.image_header.update({('RA',coord.icrs.ra.deg,'[deg] RA coordinate of projection')})
            imag.image_header.update({('DEC',coord.icrs.dec.deg,'[deg] DEC coordinate of projection')})
            imag.image_header.update({('GLON',coord.galactic.l.deg,'[deg] l galactic coordinate of projection')})
            imag.image_header.update({('GLAT',coord.galactic.b.deg,'[deg] b galactic coordinate of projection')})
            imag.image_header.update({('DATE',Time.now().iso,'date of projection')})
        except AttributeError:
            pass

        if not toplot:
            plt.ion()

        return imag

# %% -------   BEAM OPERATIONS ----------------------------------------------

    @property
    def beam_bls(self):
        """
        Gaussian beam window function. Computes the spherical transform of an axisimmetric gaussian beam.
        For a sky of underlying power spectrum C(l) observed with beam of given FWHM,
        the measured power spectrum will be C(l)_meas = C(l) B(l)^2 where B(l) is given by
        gaussbeam(Fwhm,Lmax). The polarization beam is also provided
        assuming a perfectly co-polarized beam (e.g., Challinor et al 2000, astro-ph/0008228)

        Returns
        -------
        bls : beam window function, as a [lmax+1, 4] array:
            * Temperature beam
            * Grad/electric polarization beam
            * Curl/magnetic polarization beam
            * Temperature * grad beam

        """

        fwhm_rad = self.fwhm.to(u.rad)
        if self.nmaps == 1:
            fwhm_rad = fwhm_rad.value
        else:
            fwhm_rad = fwhm_rad[0].value
        lmax = self.nside*3-1

        bls = hp.gauss_beam(fwhm_rad,lmax,pol=True)

        return bls


    def spherical_beam(self,coordinate,bls,thetamax=10*u.deg,upscale_fact=4):
        """
        Returns a `Fitsmap` containing a sky image of a beam at a given
        position (coordinate).

        Parameters
        ----------
        coordinate : `~astropy.coordinates.SkyCoord`
            Sky coordinate where the beam function is centered.
        bls : array
            Window function b(l) of the beam.
        thetamax : `~astropy.units.quantity.Quantity`, optional
            Maximum angle for the calculation of the beam. This argument
            is used to save time and storage by cutting the calculations
            above a given angular scale. Outside this radius, the beam map
            takes the value 0. The default is 10*u.deg.
        upscale_fact: int
            Upscaling factor used to refine the beam. Beam is generated
            at resolution nside2 = self.nside * upscale_fact.
            The default is 4.

        Returns
        -------
        outmap : `Fitsmap`
            A `Fitsmap` object containing a sky image of a beam at a given
            position (coordinate).

        """

        if self.ordering.upper() == 'RING':
            nest = False
        else:
            nest = True

        v1       = coord2vec(coordinate)
        nside2   = upscale_fact*self.nside
        npix2    = hp.nside2npix(nside2)
        beam_map = np.zeros(npix2)

        listpix = hp.query_disc(nside2,v1,
                                thetamax.to(u.rad).value,
                                inclusive=True,
                                nest=nest)

        for pixel in listpix:
            v2 = hp.pix2vec(nside2,pixel,nest=nest)
            d  = hp.rotator.angdist(v1,v2)*u.rad
            d  = d.value
            beam_map[pixel] = hp.bl2beam(bls,d)

        beam_map = hp.ud_grade(beam_map,self.nside,order_in=self.ordering)
        beam_map = beam_map/beam_map.max()

        outmap = Fitsmap(beam_map,self.mask,self.header,self.ordering)
        outmap.obs_frequency  = self.obs_frequency
        outmap.fwhm_base      = self.fwhm_base
        outmap.beam_area_base = self.beam_area_base

        return outmap

    def flat_beam(self,coordinate,bls,npix=512,deltatheta_deg=14.658):
        """
        Returns a `sky_images.Imagen` flat sky patch containing an image
        of the beam at a given sky coordinate.

        Parameters
        ----------
        coordinate : `~astropy.coordinates.SkyCoord`
            Sky coordinate where the beam function is centered.
        bls : array
            Window function b(l) of the beam.
        npix : int, optional
            Size of the patch. The default is 512x512 pixels.
        deltatheta_deg : float, optional
            Angular size of the patch, in degrees. The default is 14.658.

        Returns
        -------
        p2 : `sky_images.Imagen`
            A flat sky patch containing an image
            of the beam at a given sky coordinate.

        """

        p0   = self.patch(coordinate,
                          npix*4,
                          deltatheta_deg=deltatheta_deg)

        dmap = np.zeros(p0.datos.shape)
        cent = [x//2 for x in p0.size]
        for i in range(npix*4):
            for j in range(i,npix*4):
                d         = p0.angular_distance(cent[0],cent[1],i,j)[0].to(u.rad).value
                dmap[i,j] = d
                dmap[j,i] = d
        p1   = Imagen(hp.bl2beam(bls,dmap),p0.centro,p0.size,p0.pixsize)
        p2   = p1.downsample(factor=4,func=np.mean)

        return p2

    def update_fwhm(self,fwhm):
        """
        Defines (or updates) the FWHM and beam area of the `Fitsmap` object.

        Parameters
        ----------
        fwhm : `~astropy.units.quantity.Quantity`
            Full Width Half at Maximum (FWHM).

        Returns
        -------
        None.

        """
        if self.nmaps == 1:
            self.fwhm_base      = fwhm
            s                   = fwhm2sigma*fwhm
            a                   = 2.0*np.pi*(s*s).si
            self.beam_area_base = a
        else:
            self.fwhm_base      = []
            self.beam_area_base = []
            n                   = self.nmaps
            for i in range(n):
                self.fwhm_base.append(fwhm)
                s = fwhm2sigma*fwhm
                a = 2.0*np.pi*(s*s).si
                self.beam_area_base.append(a)

    def update_beam_area(self,area):
        """
        Defines (or updates) the FWHM and beam area of the `Fitsmap` object.

        Parameters
        ----------
        area : `~astropy.units.quantity.Quantity`
            Beam area.

        Returns
        -------
        None.

        """
        a = area
        if self.nmaps == 1:
            self.beam_area_base = a
            self.fwhm_base      = sigma2fwhm*(np.sqrt(a/(2*np.pi))).si
        else:
            self.fwhm_base      = []
            self.beam_area_base = []
            n                   = self.nmaps
            for i in range(n):
                self.beam_area_base.append(a)
                fwhm = sigma2fwhm*(np.sqrt(a/(2*np.pi))).si
                self.fwhm_base.append(fwhm)

# %% -------   UNIT CONVERSIONS ---------------------------------------------


    def to_unit(self,final_unit,
                omega_B=None,
                freq=None):
        """
        Converts the data in the `Fitsmap` to the specified unit.

        Parameters
        ----------
        final_unit : `~astropy.units.quantity.Quantity`
            Unit to which the data is transformed.
        omega_B : angular area `~astropy.units.quantity.Quantity`, optional
            Beam area. The default is None.
        freq : frequency `~astropy.units.quantity.Quantity`, optional
            The frequency of observation. The default is None.

        Returns
        -------
        None.

        """

        input_units = self.physical_units
        m           = self.nmaps

        if freq is None:
            freq      = self.obs_frequency

        if omega_B is None:
            beam_area = self.beam_area
        else:
            if isinstance(omega_B,list) and len(omega_B)==m:
                beam_area = omega_B
            elif isinstance(omega_B,list) and len(omega_B)!=m:
                beam_area = [omega_B[0]]*m
                print(' --- Warning: beam areas did not have the same dimension as nmaps. Taking first only')
            elif isinstance(omega_B.value,np.ndarray) and omega_B.size==m:
                beam_area = [z for z in omega_B]
            elif isinstance(omega_B.value,np.ndarray) and omega_B.size!=m:
                beam_area = [omega_B[0]]*m
                print(' --- Warning: beam areas did not have the same dimension as nmaps. Taking first only')
            else:
                beam_area = [omega_B]*m

        for imap in range(m):
            fconv = convert_factor(input_units[imap],
                                   final_unit,
                                   nu=freq,
                                   beam_area=beam_area[imap])
            if m==1:
                self.data         = fconv*self.data
            else:
                self.data[imap,:] = fconv*self.data[imap,:]

            self.header['TUNIT{0}'.format(imap+1)] = final_unit.to_string()


    def to_Jy(self,barea=None,freq=None):
        """
        Applies the unit conversion to janskys. This method invokes `Fitsmap.to_unit`,
        setting *final_unit* to astropy.units.Jy

        Parameters
        ----------
        barea : angular area `~astropy.units.quantity.Quantity`, optional
            Beam area. The default is None.
        freq : frequency `~astropy.units.quantity.Quantity`, optional
            The frequency of observation. The default is None.

        Returns
        -------
        None.

        """
        self.to_unit(u.Jy,omega_B=barea,freq=freq)

    def to_K(self,barea=None,freq=None):             # thermodynamic conversion is assumed
        """
        Applies the unit conversion to Klvin (thermodynamic).
        This method invokes `Fitsmap.to_unit`,
        setting *final_unit* to astropy.units.K

        Parameters
        ----------
        barea : angular area `~astropy.units.quantity.Quantity`, optional
            Beam area. The default is None.
        freq : frequency `~astropy.units.quantity.Quantity`, optional
            The frequency of observation. The default is None.

        Returns
        -------
        None.

        """
        self.to_unit(Kcmb,omega_B=barea,freq=freq)


# %% -------   HEADER MANIPULATION ------------------------------------------

    def set_name(self,column,nombre):
        """
        Updates the TTYPE header information of a given extension (column)
        of the `Fitsmap`.

        Parameters
        ----------
        column : int
            The number of the extension (column). Be aware that the FITS
            standard for numbering extensions starts in 1, while numpy arrays
            start in 0. For example, in oder to set the TTYPE1, *column* must
            be set to 0.
        nombre : str
            The string to be stored in the TTYPE.

        Returns
        -------
        None.

        """
        self.header['TTYPE{0}'.format(column+1)] = nombre

    def set_unit(self,column,nombre):
        """
        Updates the TUNIT header information of a given extension (column)
        of the `Fitsmap`.

        Parameters
        ----------
        column : int
            The number of the extension (column). Be aware that the FITS
            standard for numbering extensions starts in 1, while numpy arrays
            start in 0. For example, in oder to set the TUNIT1, *column* must
            be set to 0.
        nombre : str
            The string to be stored in the TUNIT.

        Returns
        -------
        None.

        """
        self.header['TUNIT{0}'.format(column+1)] = nombre


    def set_units(self,unit_string):
        """
        Sets all the TUNITs in the `Fitsmap` header to a given unit.

        Parameters
        ----------
        unit_string : string
            The unit.

        Returns
        -------
        None.

        """
        for i in range(self.nmaps):
            self.header['TUNIT{0}'.format(i+1)] = unit_string

    def add_comment(self,comment):
        """
        Adds a comment to the `Fitsmap` header

        Parameters
        ----------
        comment : str
            A comment.

        Returns
        -------
        None.

        """
        self.comment_count += 1
        self.header['COMMENT{0}'.format(self.comment_count)] = comment

    def locate_type(self,tstring):
        """
        Searches the the TTYPEs in the `Fitsmap` header for a particular
        string.

        Parameters
        ----------
        tstring : str
            Search string.

        Returns
        -------
        list
            List of integers giving the index of each extension for which the
            TTYPE is equal to the search string.

        """
        return [i for i in range(self.nmaps) if (self.header['TTYPE{0}'.format(i+1)] == tstring)]

# %% -------   INPUT/OUTPUT -------------------------------------------------

    def write(self,filename,new_type=None,new_unit=None,additional_header=None):
        """
        Writes the `Fitsmap` to a .fits file.

        Parameters
        ----------
        filename : str
            Name of the file where the `Fitsmap` will be written.
        new_type : str, optional
            If not None, the value of this parameter sets the TTYPE keyword in the
            file header. The default is None.
        new_unit : str, optional
            If not None, the value of this parameter sets the TUNIT keyword in the
            file header. The default is None.
        additional_header : list of header entries, optional
            Additional values for the file header. The default is None.

        Returns
        -------
        None.

        """

        if new_type is not None:
            self.set_name(0,new_type)
        if new_unit is not None:
            self.set_unit(0,new_unit)

        columns  = self.columns
        if columns == ['']:
            columns =[]
            for i in range(self.nmaps):
                columns.append('col{0}'.format(i+1))
        units    = self.units
        coordsys = self.coordsys
        hextr    = []
        hextr.append(('COMMENT{0}'.format(self.comment_count),'--- Created by D. Herranz ----'))
        if self.ordering.upper() == 'NEST':
            nestb = True
        else:
            nestb = False

        try:
            pconv = self.header['POLCCONV']
            hextr.append(('POLCCONV',pconv))
        except KeyError:
            hextr.append(('POLCCONV','COSMO'))

        try:
            pconv = self.header['BAD_DATA']
            hextr.append(('BAD_DATA',pconv))
        except KeyError:
            hextr.append(('BAD_DATA',self.badval))

        hextr.append(('CREATION',time.asctime()))

        if additional_header is not None:
            hextr += additional_header

        if self.ismask:
            hp.write_map(filename,
                         self.data.astype('float'),
                         nest = nestb,
                         coord = coordsys,
                         extra_header = hextr,
                         dtype = float,
                         overwrite = True)
        else:
            hp.write_map(filename,
                         np.ma.array(data=self.data,
                                     mask=self.mask,
                                     fill_value=self.badval),
                         nest = nestb,
                         coord = coordsys,
                         column_names = columns,
                         column_units = units,
                         extra_header = hextr,
                         dtype = float,
                         overwrite = True)

    @classmethod
    def empty(self,n,verbose=True):
        """
        Generates an empty instance of `Fitsmap` with a given **nside**
        resolution parameter.

        Parameters
        ----------
        n : int
            The HEALPix **nside** paraneter.
        verbose : bool, optional
            If True, the method writes some information on screen.
            n particular, this method writes and posteriorly deletes a temporary
            FITS file. The verbose option returns the name of this temporary file.
            The default is True.

        Returns
        -------
        vacio : `Fitsmap`
            An empty `Fitsmap` instance.

        """

        x = np.zeros(hp.nside2npix(n))
        t = str(uuid.uuid4())
        f = os.getenv('HOME')+'/Temp/'+t+'.fits'
        if verbose:
            print('Writing temporary file ',f)
        hp.write_map(f,x,overwrite=True,dtype=float)
        vacio = read_healpix_map(f,verbose=False)
        if verbose:
            print('Deleting temporary file',f)
        os.remove(f)
        return vacio

    @classmethod
    def from_file(self,fname,**kwargs):
        """
        Reads a `Fitsmap` instance from a .fits file
        in HEALPix format.

        Parameters
        ----------
        fname : string
            The name of the file from which the `Fitsmap`
            is to be read.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        maps : `Fitsmap`
            A `Fitsmap` object.

        """
        maps = read_healpix_map(fname,**kwargs)
        return maps

# %% -------   OBJECT PROPERTIES ------------------------------------------

    @property
    def print_info(self):
        """
        Prints some basic info about the object on screen.

        Returns
        -------
        None.

        """
        print(' ')
        print(' Number of maps  = {0}'.format(self.nmaps))
        print(' Nside           = {0}'.format(self.nside))
        print(' FWHM (arcmin)   = {0}'.format(self.fwhm.to(u.arcmin).value))
        print(' Frequency (GHz) = {0}'.format(self.obs_frequency.to(u.GHz).value))
        print(' Units           = {0}'.format(self.units))
        print(' ')

    @property
    def fwhm(self):
        """
        Returns the Full Width at Half Maximum (FWHM) of the maps.

        Returns
        -------
        output :  list of `~astropy.units.quantity.Quantity`
            The FWHM associated to each map.

        """
        if self.fwhm_base is None:
            if self.beam_area_base is None:
                output = None
            else:
                try:
                    nf = len(self.beam_area_base)
                except TypeError:
                    nf = 1
                if nf>1:
                    output = [sigma2fwhm*(np.sqrt(b/(2*np.pi))).si for b in self.beam_area_base]
                else:
                    output = self.nmaps*[sigma2fwhm*(np.sqrt(self.beam_area_base/(2*np.pi))).si]
                self.fwhm_base = output
        else:
            if self.nmaps > 1:
                try:
                    nf = len(self.fwhm_base)
                except TypeError:
                    nf = 1
                if nf == 1:
                    output = [self.fwhm_base]*self.nmaps
                elif nf != self.nmaps:
                    output = [self.fwhm_base[0]]*self.nmaps
                else:
                    output = self.fwhm_base
                output = np.array([g.si.value for g in output])*output[0].si.unit
            else:
                output = self.fwhm_base

            self.fwhm_base = output

        output = output

        return output

    @property
    def beam_area(self):
        """
        Returns, and sets up if not presente, the values of the **beam_area_base**
        attribute.

        Returns
        -------
        a : list of `~astropy.units.quantity.Quantity`
            The beam area associated to each map.

        """
        if self.beam_area_base is None:
            if self.fwhm is None:
                a = self.pixel_area
                print(' Map beam area not found. Using pixel area instead ')
            else:
                f = self.fwhm
                s = fwhm2sigma*f
                a = 2.0*np.pi*(s*s).si
        else:
            a = self.beam_area_base

        try:
            if a.size != self.nmaps:
                a = self.nmaps*[a]
        except AttributeError:
            pass

        self.beam_area_base = a
        return a

    @property
    def T(self):
        """
        Returns the first map (assumed to be T), if it exists

        Returns
        -------
        array like
            The first data layer (map) in the `Fitsmap`.

        """
        if self.data.size == self.npix:
            return self.data
        else:
            return self.data[0,:]

    @property
    def Q(self):
        """
        Returns the second map (assumed to be Q), if it exists

        Returns
        -------
        array like
            The second data layer (map) in the `Fitsmap`.

        """
        if self.data.size == self.npix:
            return np.array([])
        else:
            return self.data[1,:]

    @property
    def U(self):
        """
        Returns the third map (assumed to be U), if it exists

        Returns
        -------
        array like
            The third data layer (map) in the `Fitsmap`.

        """

        if self.data.size == self.npix:
            return np.array([])
        else:
            return self.data[2,:]

    @property
    def badval(self):
        """
        Returns the value of the pixels that are considered *bad pixels*. The
        methood reads this value from the `Fitsmap` header. If the header
        does not contain the **BAD_DATA** keyword, then this methods returns
        the default healpy **UNSEEN** value.

        Returns
        -------
        b : float
            Bad value numerical value.

        """
        try:
            b = self.header['BAD_DATA']
        except KeyError:
            self.header['BAD_DATA'] = hp.UNSEEN
            b = hp.UNSEEN
        return b

    @property
    def nmaps(self):
        """
        Returns the number of maps contained in the `Fitsmap` object.

        Returns
        -------
        n : int
            The number of maps.

        """
        if self.data.size == self.npix:
            n = 1
        else:
            n = self.data.shape[0]
        return n

    @property
    def nside(self):
        """
        Returns the HEALPix **nside** parameter for the maps contained in the
        `Fitsmap`.

        Returns
        -------
        int
            The HEALPix **nside** parameter for the maps contained in the
            `Fitsmap`.

        """
        return hp.get_nside(self.data)

    @property
    def npix(self):
        """
        Returns the HEALPix **npixel** parameter for the maps contained in the
        `Fitsmap`.

        Returns
        -------
        int
            The HEALPix **npixel** parameter for the maps contained in the
            `Fitsmap`.

        """

        return hp.pixelfunc.nside2npix(self.nside)

    @property
    def resolution(self):
        """
        Returns the approximate pixel size, in arcmin, by invoking the
        healpy `nside2resol` method.

        Returns
        -------
        float
            The pixel size, in arcmin.

        """

        return (hp.nside2resol(self.nside,arcmin=True))*u.arcmin

    @property
    def coordsys(self):
        """
        Returns the coordinate system (by default, Galactic) of the maps. This
        is read from the **COORDSYS** keyword in the header, if possible. If
        not, Galactic coordinates are assumed.

        Returns
        -------
        csys : strign
            'G' for Galactic coordinates, 'C' for equatorial.

        """

        try:
            csys = self.header['COORDSYS']
            if csys.upper() == 'EQUATORIAL':
                csys = 'C'
            else:
                csys = csys[0].upper()
        except KeyError:
            csys = 'G'
            self.header['COORDSYS'] = 'GALACTIC'
        return csys

    @property
    def columns(self):
        """
        Returns, as a list of strings, the names of the maps (if they exist).

        Returns
        -------
        cnames : list of strings
            The names of the maps contained in the `Fitsmap` object.

        """

        n      = self.nmaps
        cnames = []
        for i in range(n):
            st = 'TTYPE{0}'.format(i+1)
            try:
                s = self.header[st]
            except KeyError:
                s = ''
                self.header[st] = ''
            cnames.append(s)
        return cnames

    @property
    def set_beam_areas(self):
        """
        Ensures that the **beam_area_base** attribute of the `Fitsmap` is initialized.

        Returns
        -------
        None.

        """
        sigmas = fwhm2sigma*self.fwhm
        self.beam_area_base = 2.0*np.pi*(sigmas**2).si


    @property
    def units(self):
        """
        Returns, as a list of strings, the units of the maps, as read from the
        `Fitsmap` header. If the header does not contain the appropriate
        TUNIT keywords, the method returns an empty list.

        Returns
        -------
        cnames : list of strings
            The units of the maps.

        """

        n      = self.nmaps
        cnames = []
        for i in range(n):
            st = 'TUNIT{0}'.format(i+1)
            try:
                s = self.header[st]
            except KeyError:
                s = ''
                self.header[st] = ''
            cnames.append(s)
        return cnames

    @property
    def physical_units(self):
        """
        This method tries to automatically determine which are the `astropy.unit`
        of the `Fitsmap`. If this method is unable to parse the unit, it returns an
        empty list instead.

        Returns
        -------
        unit_phys : list of astropy.units
            A list containing an `astropy.unit` per each one of the maps.

        """


        unit_string = self.units
        if isinstance(unit_string,str):
            unit_phys = [parse_unit(unit_string)]
        else:
            unit_phys = [parse_unit(x) for x in unit_string]

        return unit_phys

    @property
    def ismask(self):
        """
        Checks whether the data contained in the `Fitsmap` object is a
        mask (boolean) or not.

        Returns
        -------
        bool
            True if the `Fitsmap` object contains a sky mask.

        """
        return self.data.dtype == np.dtype('bool')


"""
===============================================================================

"""

def read_healpix_map(fname,maskmap=False,freq=None,fwhm=None,maskval=None,verbose=True):

    if maskmap:
        maps,h = hp.read_map(fname,field=None,h=True,dtype=None)
        maps   = np.invert(maps.astype(np.bool))        # set mask = True where the pixel value is zero, False where the pixel value is one
    else:
        maps,h = hp.read_map(fname,field=None,h=True,dtype=None)

    ordering = 'RING'
    header   = dict(h)
    header['ORDERING'] = 'RING'

    if maskmap:
        mask      = maps
    elif maskval is not None:
        mask = maps==maskval
    else:
        try:
            bdata = header['BAD_DATA']
            mask  = maps==bdata
        except KeyError:
            mask  = np.zeros(maps.shape,dtype='bool')

    mapa = Fitsmap(maps,mask,header,ordering)

    mapa.obs_frequency = freq

    if fwhm is not None:
        mapa.update_fwhm(fwhm)

    return mapa


def get_pixel_fwhm(nside):

    theta_max = 5*hp.nside2resol(nside)
    thetarr   = np.linspace(-theta_max,theta_max,1000)
    windowf   = hp.pixwin(nside)
    pixbeam   = hp.bl2beam(windowf,thetarr)
    fitter    = modeling.fitting.LevMarLSQFitter()
    model     = modeling.models.Gaussian1D(amplitude=pixbeam.max(),
                                           mean=0,
                                           stddev=hp.nside2resol(nside))
    fmodel    = fitter(model,thetarr,pixbeam)
    sigma     = fmodel.stddev.value * u.rad

    return sigma2fwhm*sigma