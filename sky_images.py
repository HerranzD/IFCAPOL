#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:43:56 2018

@author: herranz
"""

import numpy as np
import matplotlib.pyplot as plt
import matched_filter as mf
import healpy as hp
import astropy.units as u
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.nddata import block_reduce,block_replicate
from astropy.nddata import Cutout2D
from astropy.table  import Table
from astropy.time   import Time
from scipy.stats import describe
from image_utils import ring_min,ring_max,ring_mean,ring_std,ring_sum,ring_count,ring_median
from image_utils import min_in_circle,max_in_circle,sum_in_circle,count_in_circle,median_in_circle
from gauss_window import makeGaussian
from myutils import coord2vec,img_shapefit,sigma2fwhm,fwhm2sigma
from gauss2dfit import fit_single_peak
from scipy.ndimage import gaussian_filter,convolve


class Imagen:

    """
    Imagen
    ======

    A class used to represent a sky image.


    Attributes
    ----------
    datos : numpy.ndarray
        A numpy ndarray in two dimensions, containing the data (pixels) of the
        image.
    centro : tuple or np.array
        The colatitude and longitude (theta,phi) coordinates, in degrees, of
        the center of the image, using the Planck healpix convenion.
    size : np.array
        The size in pixels of the image (xsize,ysize).
    pixsize : astropy.units.quantity.Quantity
        The angular size of each pixel of the image.
    image_header : dictionary
        The header of the image, as read from a fits file or generated
        inside the class, cointainin astrometric and bookkeeping information.
    image_coordsys : str
        The coordinate system of the image. The default is 'galactic'

    Methods
    -------
    copy()
        Returns a copy instance of an `Imagen` object.
    tipo()
        Returns the string 'sky image' as the data type of this class.

    std()
        Returns the standard deviation of the data in the datos attribute.
    minmax()
        Return the minimum and the maximum values of the image.
    stats_in_rings(inner,outer,clip)
        Calculates statistics in a ring of pixels around the center of the
        image, possibly applying a sigma clipping recipe.
    stats()
        Compute several descriptive statistics of the image data.
    pixsize_deg
        The pixel angular size, in degrees.
    lsize
        The number of pixels along the first dimension of the image.
    write(filename)
        Writes the `Imagen` to a file in a format readable by the `Imagen` class.

    """

    image_header   = None
    image_coordsys = 'galactic'

    def __init__(self,datos,centro,size,pixsize):
        self.datos   = datos
        self.centro  = centro     # np.array([Theta, Phi]) in the Planck
                                  #     coordinate convenion.
        self.size    = size       # in pixels
        self.pixsize = pixsize    # with astropy.units

    def __add__(self,other):
        if isinstance(other, (int, float, complex)):
            datos = self.datos+other
        else:
            datos = self.datos+other.datos
        result                = Imagen(datos,self.centro,self.size,self.pixsize)
        result.image_header   = self.image_header.copy()
        result.image_coordsys = self.image_coordsys
        return result

    def __radd__(self, other):                         # reverse SUM
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self,other):
        if isinstance(other, (int, float, complex)):
            datos = self.datos-other
        else:
            datos = self.datos-other.datos
        result                = Imagen(datos,self.centro,self.size,self.pixsize)
        result.image_header   = self.image_header.copy()
        result.image_coordsys = self.image_coordsys
        return result

    def __rsub__(self,other):                         # reverse SUM
        if other == 0:
            return self
        else:
            return self.__sub__(other)

    def __mul__(self,other):
        if isinstance(other, (int, float, complex)):
            datos = self.datos*other
        else:
            datos = self.datos*other.datos
        result                = Imagen(datos,self.centro,self.size,self.pixsize)
        result.image_header   = self.image_header.copy()
        result.image_coordsys = self.image_coordsys
        return result

    def __rmul__(self, other):                         # reverse SUM
        if other == 0:
            return self
        else:
            return self.__mul__(other)

    def __pow__(self,other):
        datos                 = self.datos**other
        result                = Imagen(datos,self.centro,self.size,self.pixsize)
        result.image_header   = self.image_header.copy()
        result.image_coordsys = self.image_coordsys
        return result

    def copy(self):
        x = Imagen(self.datos,self.centro,self.size,self.pixsize)
        x.image_header   = self.image_header.copy()
        x.image_coordsys = self.image_coordsys
        return x

    def tipo(self):
        return 'sky image'

# %% ---- BASIC DESCRIPTION AND STATISTICS -------------

    def std(self):
        """
        Returns the standard deviation of the data stored in the
        datos attribute.

        Returns
        -------
        float
            Standard deviation, calculated using the numpy std function.

        """
        return self.datos.std()

    def mean(self):
        """
        Returns the mean value of the data stored in the
        datos attribute.

        Returns
        -------
        float
            Mean, calculated using the numpy mean function.

        """
        return self.datos.mean()

    def minmax(self):
        """
        Returns the minimum and maximum values of the image.

        Returns
        -------
        float,float
            Minumum, maximum of the values stored in the
            datos attribute.

        """
        return self.datos.min(),self.datos.max()

    def stats_in_rings(self,inner,outer,clip=None):
        """
        Returns statistics calculated in a ring around
        the geometric center of the image.

        Parameters
        ----------
        inner : astropy.units.quantity.Quantity
            Inner radius of the ring. It has physical angular units
            (e.g. 10 arcmin).
        outer : astropy.units.quantity.Quantity
            Outer radius of the ring. It has physical angular units
            (e.g. 20 arcmin).
        clip : float or None, optional
            If not None, the statistics are computed using a sigma clipping
            rule. The default is None.

        Returns
        -------
        dict
            A dictionary containing: minimum and maximum values inside the
            inner radius, minimum and maximum value within the ring, mean,
            standard deviation, median, sum and count (number of elements)
            within the ring.

        """
        rmin = (inner/self.pixsize).si.value
        rmax = (outer/self.pixsize).si.value
        img  = self.datos
        if rmin > 0.0:
            s = ring_sum(img,rmin,rmax)
            n = ring_count(img,rmin,rmax)
            m = ring_median(img,rmin,rmax)
        else:
            s = sum_in_circle(img,rmax)
            n = count_in_circle(img,rmax)
            m = median_in_circle(img,rmax)

        return {'min_inside':min_in_circle(img,rmin),
                'max_inside':max_in_circle(img,rmin),
                'min_ring':ring_min(img,rmin,rmax,clip=clip),
                'max_ring':ring_max(img,rmin,rmax,clip=clip),
                'mean':ring_mean(img,rmin,rmax,clip=clip),
                'std':ring_std(img,rmin,rmax,clip=clip),
                'sum':s,
                'count':n,
                'median':m}

    @property
    def stats(self):
        """
        Invokes the scipy.stats.describe method on the data stored in datos.

        Returns
        -------
        dictionary
            A scipy.stats.describe instance.

        """
        return describe(self.datos,axis=None)

    @property
    def pixsize_deg(self):
        """
        Gives the pixel angular size in degrees.

        Returns
        -------
        float
            The pixel angular size, in degrees.

        """
        return self.pixsize.to(u.deg).value

    @property
    def lsize(self):
        """
        The first dimension of the shape of the datos array.

        Returns
        -------
        int
            The first dimension of the shape of the datos array.

        """
        return self.datos.shape[0]

# %% ---- INPUT/OUTPUT --------------------------------

    def write(self,fname):
        """
        Writes the `Imagen` to a file readable by the `Imagen` class. Preferred
        format is FITS, but this method supports also ascii files.

        Parameters
        ----------
        fname : str
            File name for the output.

        Returns
        -------
        None.

        """
        exts_fits  = ['fits','fits.gz']
        exts_ascii = ['txt','dat']
        exts_img   = ['eps','jpeg','jpg','pdf','pgf','png',
                      'ps','raw','rgba','svg','svgz','tif','tiff']
        extension = fname.split('.')[-1].lower()
        if extension in exts_fits:

            hdu = fits.PrimaryHDU(np.fliplr(np.flipud(self.datos)))
            hdu.header.update(self.image_header)
            hdu.writeto(fname,overwrite=True)

        elif extension in exts_ascii:

            n = self.lsize
            with open(fname, 'w') as f:
                f.write('# GLON: {0}\n'.format(self.center_coordinate.galactic.l.deg))
                f.write('# GLAT: {0}\n'.format(self.center_coordinate.galactic.b.deg))
                f.write('# PIXSIZE: {0}\n'.format(self.pixsize_deg))
                for i in range(n):
                    for j in range(n):
                        f.write('{0}  {1}  {2}\n'.format(i+1,j+1,self.datos[i,j]))

        elif extension in exts_img:

            fig = plt.figure()
            self.draw(newfig=False)
            fig.savefig(fname)
            plt.close()

        else:

            print(' --- Unknown file type')

    @classmethod
    def from_ascii_file(self,fname):
        """
        Reads a `Imagen` object from an ASCII file with header. 
        The ASCII file must be structured as a header containing first the 
        lsize, glon, glat, pixel size keywords, after which the image data
        is written as a three column (X,Y,Z) list separated by blank spaces.

        Parameters
        ----------
        fname : string
            The name of the input file.

        Returns
        -------
        `Imagen`
            A sky image, stored as a `Imagen` object.

        """
        with open(fname) as f:
            lines = f.readlines()
        lsize  = int(np.sqrt(len(lines)-3))
        datos  = np.zeros((lsize,lsize))
        glon   = float((lines[0].split('GLON: ')[-1]).split('\n')[0])
        glat   = float((lines[1].split('GLAT: ')[-1]).split('\n')[0])
        psiz   = float((lines[2].split('PIXSIZE: ')[-1]).split('\n')[0])
        centro = np.array([90.0-glat,glon])
        size   = (lsize,lsize)
        pixs   = psiz*u.deg
        for k in range(3,len(lines)):
            l = lines[k].split(' ')
            i = int(l[0])-1
            j = int(l[2])-1
            z = float((l[4]).split('\n')[0])
            datos[i,j] = z
        return Imagen(datos,centro,size,pixs)

    @classmethod
    def from_ascii_3col_file(self,fname,
                             theta_deg,phi_deg,
                             pixsize_arcmin):
        """
        Reads a `Imagen` object from an ASCII file without header. 
        The ASCII file must be structured as a three column (X,Y,Z) 
        list separated by blank spaces.

        Parameters
        ----------
        fname : str
            Input file name.
        theta_deg : float
            Colatitude (Galactic) of the centre of the image, in degrees.
        phi_deg : float
            Galactic longitude of the centre of the image, in degrees.
        pixsize_arcmin : float
            Pixel size in arcmin.

        Returns
        -------
        `Imagen`
            A sky image, stored as a `Imagen` object.

        """

        tlines = Table.read(fname,format='ascii.no_header')
        lsize  = int(np.sqrt(len(tlines)))
        datos  = np.zeros((lsize,lsize))
        centro = np.array([theta_deg,phi_deg])
        size   = (lsize,lsize)
        pixs   = pixsize_arcmin*u.arcmin

        datos[tlines['col1']-1,tlines['col2']-1] = tlines['col3']

        return Imagen(datos,centro,size,pixs)

    @classmethod
    def from_fits_file(self,fname):
        """
        Reads a `Imagen` object from an astronomical FITS file. 
        This method invokes the lower-level from_hdu method.

        Parameters
        ----------
        fname : str
            Input file name.

        Returns
        -------
        `Imagen`
            A sky image, stored as a `Imagen` object.

        """
        hdul = fits.open(fname)
        imag = self.from_hdu(hdul[0])
        hdul.close()

        i,j  = imag.size[0]/2,imag.size[1]/2
        cc   = imag.pixel_coordinate(i,j)
        imag.centro = np.array([90.0-cc.galactic.b.deg,
                                cc.galactic.l.deg])

        return imag

    @classmethod
    def from_hdu(self,hdul):
        """
        Reads a `Imagen` object from an astronomical HDU list
        using the astropy.io.fits modules.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            An astropy HDUlist with some header plus a data extension.

        Returns
        -------
        `Imagen`
            A sky image, stored as a `Imagen` object.

        """
        h    = hdul.header
        w    = wcs.WCS(h)
        co   = w.pixel_to_world(h['NAXIS1']/2,h['NAXIS2']/2)

        try:
            pixsize = u.Quantity(np.abs(h['CDELT1']),unit=h['CUNIT1'])
        except KeyError:
            try:
                pixsize = u.Quantity(np.abs(h['CDELT1']),unit=u.deg)
            except KeyError:
                pixsize = np.abs(w.pixel_scale_matrix).mean()*u.deg

        centro  = np.array([90.0-co.galactic.b.deg,co.galactic.l.deg])
        datos   = np.fliplr(np.flipud(hdul.data))
        size    = hdul.data.shape

        self.image_header = h
        if (('RA' in h['CTYPE1'].upper()) or ('RA' in h['CTYPE2'].upper())):
            self.image_coordsys = 'icrs'
        else:
            self.image_coordsys = 'galactic'

        return Imagen(datos,centro,size,pixsize)

    @classmethod
    def from_file(self,fname):
        """
        A wrapper of the from_ascii_file, from_ascii_3col_file and from_fits_file
        class methods, this class method reads a `Imagen` instance from a file.

        Parameters
        ----------
        fname : string
            Input file name. It must have a extension in {'fits','txt','dat'}

        Returns
        -------
        newimg : `Imagen`
            A sky image, stored as a `Imagen` object.

        """
        exts_fits  = ['fits']
        exts_ascii = ['txt','dat']
        extension = fname.split('.')[-1].lower()
        if extension in exts_fits:
            newimg = self.from_fits_file(fname)
        elif extension in exts_ascii:
            newimg = self.from_ascii_file(fname)
        else:
            print(' --- Unknown file type')
            newimg = []
        return newimg

    @classmethod
    def empty(self,npix,pixsize):
        """
        Creates an empty `Imagen` instance. The `Imagen` is centered at the mock
        colatitude theta=phi=45 degree coordinates.

        Parameters
        ----------
        npix : integer
            The number of pixels per side of the image.
        pixsize : astropy.units.quantity.Quantity
            Angular size of each pixel (assumed to be square).

        Returns
        -------
        `Imagen`
            An Image object with size and pixel size specified by the 
            **npix** and **pixsize** arguments, and with **datos** = 0.0

        """
        d = np.zeros((npix,npix))
        return Imagen(d,(45.0,45.0),(npix,npix),pixsize)


# %% ---- PLOTTING ----------------------

    def plot(self):
        """
        Basic plotting of the data in Imagen.datos

        Returns
        -------
        None.

        """
        wcs = self.wcs
        fig = plt.figure()
        fig.add_subplot(111, projection=wcs)
        plt.imshow(np.flipud(np.fliplr(self.datos)),  cmap=plt.cm.viridis)
        plt.xlabel('RA')
        plt.ylabel('Dec')


    def draw(self,
             pos        = 111,
             newfig     = False,
             animated   = False,
             coord_grid = False,
             colorbar   = True,
             tofile     = None):
        """
        Advanced plotting of the data in *Imagen.datos* .
        The plot uses World Coordinate System and can be placed inside a 
        given subplot, have or not and overlaid coordinate grid and sent
        to a file, on demand.

        Parameters
        ----------
        pos : int, optional
            The identifier of a subplot where the plot is sent. 
            The default is 111.
        newfig : bool, optional
            Whether to open a new figure or not. The default is False.
        animated : bool, optional
            Whether the figure will be part of an animation or not. 
            The default is False.
        coord_grid : bool, optional
            Whether to overlay a coordinate grid or not. 
            The default is False.
        colorbar : bool, optional
            Whether to add a color bar or not. The default is True.
        tofile : str or None, optional
            If not None, the method saves the figure to a file specified by
            this parameter. The default is None.

        Returns
        -------
        None.

        """

        wcs = self.wcs
        if newfig:
            plt.figure()
        plt.subplot(pos,projection=wcs)
        plt.imshow(np.flipud(np.fliplr(self.datos)),origin='lower')
        if coord_grid:
            plt.grid(color='white', ls='dotted')
        if self.image_coordsys == 'galactic':
            plt.ylabel('GLAT [deg]')
            plt.xlabel('GLON [deg]')
        else:
            plt.ylabel('RA [deg]')
            plt.xlabel('DEC [deg]')
        if colorbar:
            plt.colorbar()
        if tofile is not None:
            plt.savefig(tofile)

# %% ---- HEADER --------------------

    @property
    def header(self):
        """
        Gets the header from the `Imagen`.

        Returns
        -------
        dictionary
            The header of the image as a dictionary.

        """
        if self.image_header is None:
            w      = self.wcs
            hdu    = fits.PrimaryHDU(self.datos)
            hdu.header.update(w.to_header())
            self.image_header = hdu.header
        return self.image_header


# %% ---- COORDINATES  -------------

    @property
    def wcs(self):
        """
        Returns the World Coordinate System information corresponding to the
        `Imagen` (see https://docs.astropy.org/en/stable/wcs/index.html)

        Returns
        -------
        w : `astropy.wcs`  
            The World Coordinate System (WCS) associated to the `Imagen`.

        """

        if self.image_header is not None:

            w = wcs.WCS(self.image_header)

        else:

            n = self.size[0]//2
            c = self.center_coordinate
            w = wcs.WCS(naxis=2)
            w.wcs.crpix = [n,n]
            w.wcs.cdelt = np.array([self.pixsize_deg,self.pixsize_deg])
            if self.image_coordsys == 'galactic':
                w.wcs.crval = [c.galactic.l.deg, c.galactic.b.deg]
                w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
            else:
                w.wcs.crval = [c.icrs.ra.deg, c.icrs.dec.deg]
                w.wcs.ctype = ["RA-TAN", "DEC-TAN"]
            w.wcs.cunit = ['deg','deg']

        return w

    @property
    def center_coordinate(self):
        """
        Returns the `SkyCoord` of the center of the `Imagen`.

        Returns
        -------
        `~astropy.coordinates.SkyCoord`
            The sky coordinate of the center of the `Imagen`.

        """
        return SkyCoord(frame='galactic',
                        b=(90.0-self.centro[0])*u.deg,
                        l=self.centro[1]*u.deg)

    def pixel_coordinate(self,i,j):
        """
        Returns the sky coordinate of a given pixel of the `Imagen`.

        Parameters
        ----------
        i : int or float
            The pixel index along the x-axis.
        j : int or float
            The pixel index along the y-axis.

        Returns
        -------
        `~astropy.coordinates.SkyCoord`
            The sky coordinate of the i,j pixel.

        """
        s = self.size
        p = np.array(self.wcs.wcs_pix2world(s[1]-j,s[0]-i,1))
        return SkyCoord(p[0],p[1],unit='deg',frame=self.image_coordsys)

    def coordinate_pixel(self,coord):
        """
        Return the pixel index of the pixel nearest to a given coordinate.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            A coordinate in the sky.

        Returns
        -------
        int
            The pixel index along the x-axis.
        int
            The pixel index along the y-axis.

        """
        if self.image_coordsys == 'galactic':
            l = coord.galactic.l.deg
            b = coord.galactic.b.deg
        else:
            l = coord.icrs.ra.deg
            b = coord.icrs.dec.deg
        x = self.wcs.wcs_world2pix(l,b,1)
        return x[1],x[0]

    def angular_distance(self,i1,j1,i2,j2):
        """
        Angular distance between pixel coordinates (I1,J1) and (I2,J2)

        Parameters
        ----------
        i1 : int
            The first pixel index along the x-axis.
        j1 : int
            The first pixel index along the y-axis.
        i2 : int
            The second pixel index along the x-axis.
        j2 : TYPE
            The second pixel index along the x-axis.

        Returns
        -------
        d : `~astropy.units.quantity.Quantity` 
            Angular separation (on the sphere) between the first and the 
            second pixel.

        """
        c1 = self.pixel_coordinate(i1,j1)
        c2 = self.pixel_coordinate(i2,j2)
        d  = hp.rotator.angdist(coord2vec(c1),coord2vec(c2))*u.rad
        return d


# %% ---- POSTSTAMPS -------------

    def stamp_central_region(self,lado):
        """
        Creates a poststamp of the sky `Imagen` around its center.

        Parameters
        ----------
        lado : int
            The size, in number of pixels per side, of the poststamp.

        Returns
        -------
        r : `Imagen` 
            A new `Imagen` (the poststamp) with size (*lado,lado*) centered
            at the same postion of the parent `Imagen`.

        """
        c      = self.size[0]//2
        l      = lado//2
        d      = self.datos
        subcut = d[c-l:c+l,c-l:c+l]
        r      = Imagen(subcut,self.centro,np.array(subcut.shape),self.pixsize)
        r.image_coordsys = self.image_coordsys
        r.image_header = None
        return r

    def stamp_coord(self,coord,lado):
        """
        Creates a poststamp of the sky `Imagen` around a given coordinate.

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            The coordinate around which the poststamp is generated.
        lado : int
            The size, in number of pixels per side, of the poststamp.

        Returns
        -------
        output_img : `Imagen` 
            A new `Imagen` (the poststamp) with size (*lado,lado*) centered
            at the coordinate *coord*.

        """

        imagen      = fits.PrimaryHDU(self.datos)
        wcs0        = wcs.WCS(self.header)
        wcs1        = wcs0.copy()
        cutout      = Cutout2D(imagen.data,
                               position=coord,
                               size=lado,
                               wcs=wcs1,
                               mode='partial')
        imagen.data = cutout.data
        imagen.header.update(cutout.wcs.to_header())
        output_img  = Imagen.from_hdu(imagen)

        return output_img


# %% ---- FILTERING -------------------------

    def matched(self,fwhm=1.0*u.deg,toplot=False):
        """
        Runs a matched filter.

        Parameters
        ----------
        fwhm : `~astropy.units.quantity.Quantity` or ndarray, optional
            The profile assumed for the matched filter. 
                - If **fwhm** is an astropy Quantity (for example fwhm=1*u.arcmin) the profile is assumed to be a Gaussian with the given full width half maximum.
                - If **fwhm** is a ndarray, it is interpreted as a image representation of the profile (in real space). The ndarray should have the same size as the `Imagen.datos` .
            The default is 1.0*u.deg.
        toplot : bool, optional
            If True, the matched filtered image is plotted in a new figure. 
            The default is False.

        Returns
        -------
        fmapa : `Imagen` 
            An `Imagen` with the same parameters as the parent `Imagen` 
            whose *datos* attribute contains the matched filtered version 
            of the parent `Imagen`.

        """
        if np.size(fwhm) == 1:
            s = (fwhm/self.pixsize).si.value
            fdatos = mf.matched_filter(self.datos,lafwhm=s,
                                       nbins=self.lsize//4,
                                       aplot=False)
        else:
            fdatos = mf.matched_filter(self.datos,
                                       gprof=False,
                                       lafwhm=1.0,
                                       tprof0=img_shapefit(fwhm,self.datos),
                                       nbins=self.lsize//4,
                                       aplot=False)
        fmapa  = Imagen(fdatos,self.centro,self.size,self.pixsize)

        fmapa.image_header = self.image_header.copy()
        fmapa.image_header.update({('COMMENT','Matched filtered image')})
        fmapa.image_header.update({('FDATE',Time.now().iso,'date of filtering')})

        if toplot:
            fmapa.draw(newfig=True)

        return fmapa

    def iter_matched(self,fwhm=1.0*u.deg,toplot=False):
        """
        Run an interative matched filter.

        Parameters
        ----------
        fwhm : `~astropy.units.quantity.Quantity` or ndarray, optional
            The profile assumed for the matched filter. 
                - If **fwhm** is an astropy Quantity (for example fwhm=1*u.arcmin) the profile is assumed to be a Gaussian with the given full width half maximum.
                - If **fwhm** is a ndarray, it is interpreted as a image representation of the profile (in real space). The ndarray should have the same size as the `Imagen.datos` .
            The default is 1.0*u.deg.
        toplot : bool, optional
            If True, the matched filtered image is plotted in a new figure. 
            The default is False.

        Returns
        -------
        fmapa : `Imagen` 
            An `Imagen` with the same parameters as the parent `Imagen` 
            whose *datos* attribute contains the non-interative matched filtered 
            version of the parent `Imagen`.
        fmapai : `Imagen` 
            An `Imagen` with the same parameters as the parent `Imagen` 
            whose *datos* attribute contains the iterative matched filtered version 
            of the parent `Imagen`.

        """
        s = (fwhm/self.pixsize).si.value
        fdat1,fdatos = mf.iterative_matched_filter(self.datos,
                                                   lafwhm=s,
                                                   nbins=self.lsize//4,
                                                   snrcut=5.0,
                                                   aplot=False,
                                                   topad=True,
                                                   kind='linear')
        fmapa  = Imagen(fdat1,self.centro,self.size,self.pixsize)
        fmapai = Imagen(fdatos,self.centro,self.size,self.pixsize)
        if toplot:
            fmapa.draw(newfig=True)
        return fmapa,fmapai

    def smooth(self,fwhm=1.0*u.deg,toplot=False):
        """
        Smooths the image with a Gaussian kernel.

        Parameters
        ----------
        fwhm : `~astropy.units.quantity.Quantity`, optional
            The FWHM of the Gaussian smoothing kernel. The default is 1.0*u.deg.
        toplot : bool, optional
            If True, the smoothed image is plotted in a new figure. 
            The default is False.

        Returns
        -------
        fmapa : `Imagen` 
            An `Imagen` with the same parameters as the parent `Imagen` 
            whose *datos* attribute contains the smoothed version 
            of the parent `Imagen`.
        """

        sigma  = (fwhm2sigma*fwhm/self.pixsize).si.value
        print(' --- Smoothing image with a Gaussian kernel of sigma = {0} pixels'.format(sigma))
        fdatos = gaussian_filter(self.datos,sigma=sigma)
        fmapa  = Imagen(fdatos,self.centro,self.size,self.pixsize)
        if toplot:
            fmapa.draw(newfig=True)
        return fmapa
    
    def convolve(self,kernel,toplot=False):
        """
        Convolves the image with a given kernel

        Parameters
        ----------
        kernel : ndarray
            Array of convolution weights, same number of dimensions as 
            input *datos*.
        toplot : bool, optional
            If True, the convolved image is plotted in a new figure. 
            The default is False.

        Returns
        -------
        cmap : `Imagen` 
            An `Imagen` with the same parameters as the parent `Imagen` 
            whose *datos* attribute contains the convolved version 
            of the parent `Imagen`.

        """
        
        input_array  = self.datos.copy()
        output_array = convolve(input_array,kernel)
        cmap         = self.copy()
        cmap.datos   = output_array 
        if toplot:
            cmap.draw(newfig=True)
            
        return cmap

# %% ---- FITTING -----------------------

    def central_gaussfit(self,return_output=False,verbose=True):
        """
        Performs a Least Squares fitting of the `Imagen.datos` to
        a Gaussian profile plus a linear baseline.

        Parameters
        ----------
        return_output : bool, optional
            If True, returns a `Ajuste` fit object. The default is False.
        verbose : bool, optional
            If True, the result of the fitting is printed on screen. 
            The default is True.

        Returns
        -------
        cfit : `Ajuste`
            The output Gaussian fitting parameters, models and images.

        """
        patch = self.datos.copy()
        cfit  = fit_single_peak(patch)
        sigma = self.pixsize * cfit.sigma
        area  = (2*np.pi*sigma*sigma).si
        fwhm  = sigma2fwhm*sigma
        if verbose:
            print(' --- Fitted beam area = {0}'.format(area))
            print(' --- Fitted beam fwhm = {0}'.format(fwhm))
            print(' --- Fitted amplitude = {0}'.format(cfit.amplitude))
            print(' --- Fitted centre    = ({0},{1})'.format(cfit.x,cfit.y))
        if return_output:
            return cfit


# %% ---- MASKING -----------------------

    def mask_value(self,value):
        """
        Masks pixels equal to a given value.

        Parameters
        ----------
        value : float
            Value to be masked.

        Returns
        -------
        None.

        """
        d = self.datos
        self.datos = np.ma.masked_array(data=d,mask=d==value)

    def mask_brighter(self,value):
        """
        Masks pixels brighter than a given value.

        Parameters
        ----------
        value : float
            Threshold for masking.

        Returns
        -------
        None.

        """
        d = self.datos
        self.datos = np.ma.masked_array(data=d,mask=d>value)

    def mask_fainter(self,value):
        """
        Masks pixels fainter than a given value.

        Parameters
        ----------
        value : float
            Threshold below which the data is masked.

        Returns
        -------
        None.

        """
        d = self.datos
        self.datos = np.ma.masked_array(data=d,mask=d<value)

    def mask_border(self,nbpix):
        """
        Masks the border of the `Imageen` 

        Parameters
        ----------
        nbpix : int
            The number of pixels of the border.

        Returns
        -------
        None.

        """
        d = self.datos
        z = np.zeros(d.shape,dtype=bool)
        m = d.shape[0]
        z[0:nbpix,:]   = True
        z[m-nbpix:m,:] = True
        m = d.shape[1]
        z[:,0:nbpix]   = True
        z[:,m-nbpix:m] = True
        self.datos = np.ma.masked_array(data=d,mask=z)

    def mask_brightest_fraction(self,fraction):
        """
        Masks a fraction of the brightest pixels of the `Imagen`.

        Parameters
        ----------
        fraction : float
            The fraction of pixels to be masked. For example, if fraction=0.1,
            the 10% brightest pixels will be masked.

        Returns
        -------
        None.

        """
        d = self.datos
        if np.ma.is_masked(d):
            x = d.flatten()
            y = x[x.mask==False]
            z = np.sort(y)
            s = round(fraction*z.size)
            v = z[-s]
        else:
            x = np.sort(d.flatten())
            s = round(fraction*x.size)
            v = x[-s]
        self.datos = np.ma.masked_array(data=d,mask=d>=v)

    def fraction_masked(self):
        """
        Returns the fraction of pixels that are masked

        Returns
        -------
        float
            The fraction (between 0 and 1) of pixels that are masked.

        """
        return np.count_nonzero(self.datos.mask)/self.datos.size

# %% ---- PSF ----------------------------

    def psfmap(self,fwhm):
        """
        Returns a synthetic Gaussian `Imagen` with a given FWHM.

        Parameters
        ----------
        fwhm : `~astropy.units.quantity.Quantity`
            The Gaussian FWHM.

        Returns
        -------
        `Imagen` 
            A synthetic Gaussian `Imagen` with a given FWHM.

        """
        fwhm_pix = (fwhm/self.pixsize).si.value
        g        = makeGaussian(self.size[0],fwhm=fwhm_pix)
        return Imagen(g,self.centro,self.size,self.pixsize)


# %% ---- PROJECTIONS -------------------

    @property
    def gnomic_projector(self):
        """
        Returns a `healpy` Gnomonic projector object around the
        coordinate of the centre of the `Imagen`.

        Returns
        -------
        p : `~healpy.projector.GnomonicProj` 
            Gnomonic projector around the centre of the `Imagen`.

        """
        c = self.center_coordinate
        b = c.b.deg
        l = c.l.deg
        p = hp.projector.GnomonicProj(rot=[b,90.0-l],
                                      coord='G',xsize=self.size[0],
                                      ysize=self.size[1],
                                      reso=60*self.pixsize_deg)
        return p

# %% ---- RESAMPLING -------------------

    def downsample(self,factor=2,func=np.sum):
        """
        Returns a lower resolution (downsampled) version of the `Image`.

        Parameters
        ----------
        factor : int, optional
            The downsampling factor. The default is 2.
        func : function, optional
            The funcion used to downsample. The default is np.sum. Other 
            possibilities include the median, maximum, minimum...

        Returns
        -------
        `Imagen` 
            A downsampled version of the `Image` with the same centre but
            size (*size//factor,size//factor*).

        """
        data0 = self.datos
        data1 = block_reduce(data0,block_size=factor,func=func)
        return Imagen(data1,self.centro,
                      tuple(ti//factor for ti in self.size),
                      self.pixsize*factor)

    def upsample(self,factor=2,conserve_sum=True):
        """
        Returns a higher resolution version of the `Image`

        Parameters
        ----------
        factor : int, optional
            The upsampling factor. The default is 2.
        conserve_sum : bool, optional
            If true, the upsampling preserves the area integral of the pixels. 
            The default is True.

        Returns
        -------
        `Imagen` 
            A upsampled version of the `Image` with the same centre but
            size (*size x factor,size x factor*).
        """
        data0 = self.datos
        data1 = block_replicate(data0,block_size=factor,conserve_sum=conserve_sum)
        return Imagen(data1,self.centro,
                      tuple(ti*factor for ti in self.size),
                      self.pixsize/factor)


# %% ---- EXAMPLE FILES -------------

example1 = '/Users/herranz/Trabajo/Test_Data/f001a066.fits'
