#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:08:07 2017

@author: herranz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq,curve_fit
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import erf
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.units import UnitConversionError
import astropy.io.fits as fits
import healpy as hp
from pynverse import inversefunc
import os
import pandas as pd
from astropy.table import QTable
from astropy import wcs
import time
import pickle
import fnmatch

# %%   BASIC CONSTANTS AND DEFINITIONS

fwhm2sigma   = 1.0/(2.0*np.sqrt(2.0*np.log(2.0)))
sigma2fwhm   = 1.0/fwhm2sigma

# %%    VARIABLE CHECKING

def isscalar(x):

    """
    Checks if a variable is a scalar, that is, whether it has dimensions or
    not.

    Parameters
    ----------
    x : any
        Variable.

    Returns
    -------
    bool
        True if the variable is a scalar.

    """

    if np.isscalar(x):
        return True
    else:
        if isinstance(x,np.ndarray):
            if x.size == 1:
                return True
            else:
                return False


# %%    LIST OPERATIONS

def find_element_index(item,lista):

    """
    Finds where in a list is a given element.

    Parameters
    ----------
    item : any type
        The element to be searched.
    lista : list
        A list where the elemment is going to be seached for.

    Returns
    -------
    list
        A list cointaining the indexes of elements of lista that are
        equal to item.

    """

    return [i for i, elem in enumerate(lista) if item == elem]

def any_element_in(list1,list2):

    """
    Checks if any of the elements of a list is in another list.

    Parameters
    ----------
    list1 : list
        The list whose elements we want to check.
    list2 : list
        The list against which we are doing the checking.

    Returns
    -------
    r : bool
        True if any of the elements of list1 is in list2.

    """

    r = False
    for x in list1:
        if x in list2:
            r = True
    return r


# %%    SORTING AND SELECTING PERCENTILES AND QUANTILES

def sort_dictionary_by_keyname(dicc):

    """
    Sorts a dictionary by keyname. It returns an alphabetically sorted copy
    of the input dictionary.

    Parameters
    ----------
    dicc : dictionary
        Input dictionary.

    Returns
    -------
    f : dictionary
        A copy of the input dictionary, with keys arranged in alphabetical
        order.

    """
    d = dicc.copy()
    l = [k for k in d.keys()]
    l.sort()
    f = {}
    for k in l:
        f[k] = d[k]
    return f

def get_indexes_between_percentiles(x,lowp=0,upp=100):

    """
    Given an array, it returns the location of the elements whose values
    are within the lowp and upp percentile values.

    Parameters
    ----------
    x : numpy array
        Input array.
    lowp : float, optional
        Lower percentile, between 0 and 100. The default is 0.
    upp : float, optional
        Upper percentile, between 0 and 100. The default is 100.

    Returns
    -------
    list
        Indexes of the elements whose values are between the lowp and upp
        percentiles.

    """
    n1 = np.percentile(x,lowp,interpolation='nearest')
    n2 = np.percentile(x,upp,interpolation='nearest')
    if lowp==0:
        I  = np.nonzero((x>=n1)&(x<=n2))
    else:
        I  = np.nonzero((x>n1)&(x<=n2))
    return [y for y in I[0]]

def binned_array_indexes(x,nbins):

    """
    Given an array, it returns a list of lists of indexes. The Nth element
    of the list gives the indexes of the array whose values are in the Nth
    bin of possible values. For example, if nbins=10 the first element would
    give the indexes of elements between percentile 0 and 0.1, the second
    those elements with values between percentiles 0.1 and 0.2, and so on.

    Parameters
    ----------
    x : numpy array
        Input array.
    nbins : int
        Number of equispaced bins into which the array will be divided.

    Returns
    -------
    l : list of lists
        Indexes belonging to each bin.

    """

    b = np.linspace(0,100,nbins+1)
    l = []
    for i in range(nbins):
        l.append(get_indexes_between_percentiles(x,lowp=b[i],upp=b[i+1]))
    return l

# %%    BASIC STATISTICS

def normal_confidence_interval(nsigmas):

    """
    Returns the confidence interval corresponding to a given sigma value
    in a Gaussian distribution.

    Parameters
    ----------
    nsigmas : float
        The sigma level.

    Returns
    -------
    float
        The corresponding confidence interval. It is a value between 0 and 1.

    """
    return erf(nsigmas/np.sqrt(2.0))

def confidence_limits(array,cl):

    """
      For a given array, gets the lower and upper values corresponding to
      a confidence limit cl. This confidence limit takes value 0.0<=cl<=1.0
    """

    x = array.copy()
    x.sort()
    p = (1.0-cl)/2
    lower = int(p*x.size)
    upper = x.size-lower
    return np.array([x[lower],x[upper]])

# %%    LEAST SQUARES FITTING

def lsq_fit(func,x,y,sigma=None,p0=None):
    """
    Least squares fit of a function to data. It uses the scipy.optimize
    curve_fit function.

    Parameters
    ----------
    func : function
        The function to be fitted.
    x : numpy array
        The x values of the data.
    y : numpy array
        The y values of the data.
    sigma : numpy array, optional
        The uncertainties of the data. The default is None.
    p0 : list, optional
        Initial guess for the parameters of the function. The default is None.

    Returns
    -------
    popt : list
        The best fit parameters.
    perr : list
        The uncertainties of the best fit parameters.
    pcov : numpy array
        The covariance matrix of the best fit parameters.

    """

    if sigma is not None:
        popt, pcov = curve_fit(func,x,y,sigma=sigma,maxfev=20000)
        if p0 is not None:
            popt, pcov = curve_fit(func,x,y,sigma=sigma,maxfev=20000,p0=p0)
    else:
        popt, pcov = curve_fit(func,x,y,maxfev=20000)
        if p0 is not None:
            popt, pcov = curve_fit(func,x,y,maxfev=20000,p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return popt,perr,pcov

def asymm_chi2(pars,func,x,y,l,u):
    """
    Chi2 function for asymmetric errors.

    Parameters
    ----------
    pars : list
        Parameters of the function.
    func : function
        The function to be fitted.
    x : numpy array
        The x values of the data.
    y : numpy array
        The y values of the data.
    l : numpy array
        The lower uncertainties of the data.
    u : numpy array
        The upper uncertainties of the data.

    Returns
    -------
    float
        The chi2 value.
    """

    z  = func(x,*pars)
    sl = y-l
    su = u-y
    d  = np.zeros(z.size)
    d[y>=z] = ((y[y>=z]-z[y>=z])/sl[y>=z])**2
    d[y<z]  = ((y[y<z]-z[y<z])/su[y<z])**2

    return d.sum()

def asymm_lsq_fit(func,x,y,l,u,p0=None):
    """
    Least squares fit of a function to data with asymmetric errors. It uses
    the scipy.optimize minimize function.

    Parameters
    ----------
    func : function
        The function to be fitted.
    x : numpy array
        The x values of the data.
    y : numpy array
        The y values of the data.
    l : numpy array
        The lower uncertainties of the data.
    u : numpy array
        The upper uncertainties of the data.
    p0 : list, optional
        Initial guess for the parameters of the function. The default is None.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
        The result of the minimization.

    """

    res = minimize(asymm_chi2,p0,args=(func,x,y,l,u))

    return res

def plot_confidence_band(func,x,popt,pcov,nstd=1,alpha=0.15,noplot=False):
    
    """
        Plots the confidence band of a fit. It uses the scipy.optimize
        curve_fit function.

        Parameters
        ----------
        func : function
            The function to be fitted.
        x : numpy array
            The x values of the data.
        popt : list
            The best fit parameters.
        pcov : numpy array
            The covariance matrix of the best fit parameters.
        nstd : float, optional
            The number of sigmas to be plotted. The default is 1.
        alpha : float, optional
            The transparency of the band. The default is 0.15.
        noplot : bool, optional
            If True, it does not plot anything. The default is False.

        Returns
        -------
        fit_dw : numpy array
            The lower confidence band.
        fit : numpy array
            The best fit.
        fit_up : numpy array
            The upper confidence band.

     """

    if noplot:
        x = np.asarray([x])

    cl      = 1.0-2*norm.sf(nstd)
    nrandom = 10000
    random_parameters = np.random.multivariate_normal(popt,pcov,nrandom)

    lines   = np.zeros((nrandom,x.size))
    for i in range(nrandom):
        pars = random_parameters[i,:]
        pars = np.array([y for y in pars])
        lines[i,:] = func(x,pars)
    shp     = lines.shape

    fit     = func(x, popt)
    fit_up  = np.zeros(fit.shape)
    fit_dw  = np.zeros(fit.shape)

    for i in range(x.size):
        if shp[0] == x.size:
            yarr = lines[i,:]
        else:
            yarr = lines[:,i]
        cli  = confidence_limits(yarr,cl)
        fit_dw[i] = cli[0]
        fit_up[i] = cli[1]

    if noplot:
        return fit_dw,fit,fit_up
    else:
        plt.plot(x,fit,'r',lw=1,label='best fit curve')
        plt.fill_between(x,fit_up,fit_dw,
                         alpha=alpha,
                         color='b') #,
#                         label='{0} sigma interval'.format(nstd))


# %%     RANDOM NUMBER GENERATORS

def random_samples_rejection(fdata,nsamples):
    """
    Generates NSAMPLES random samples drawn, using the  rejection method, 
    from the (differential) pdf stored in the histogram FDATA

    Parameters
    ----------
    fdata : numpy array
        The histogram data.
    nsamples : int
        The number of samples to be generated.

    Returns
    -------
    r : numpy array
        The random samples.

    """  

    nmax = np.int64(100*nsamples)
    x    = fdata[0]
    y    = fdata[1]
    pdf  = interp1d(x,y,bounds_error=False,fill_value=(0,0))
    xmin = x.min()
    xmax = x.max()
    ymax = y.max()
    sigue = True
    while sigue:
        xr   = (xmax-xmin)*np.random.rand(nmax)+xmin
        yr   = ymax*np.random.rand(nmax)
        m    = yr<=pdf(xr)
        if np.count_nonzero(m)>=nsamples:
            x = xr[m]
            r = x[0:np.int64(nsamples)]
            sigue = False
        else:
            nmax = 10*nmax
    return r


def random_from_cdf(cdf,nsamples):
    """
    Generates NSAMPLES random samples drawn, using the inverse transform
    sampling method, from a cumulative distribution function CDF.

    Parameters
    ----------
    cdf : function
        The cumulative distribution function.
    nsamples : int
        The number of samples to be generated.

    Returns
    -------
    y : numpy array
        The random samples.

    """
    x = np.random.rand(nsamples)
    y = inversefunc(cdf,x)
    return y


def random_from_discrete_distribution(histdata,nsamples,
                                      logbase=None,
                                      toplot=False):

    """
      Generate NSAMPLES random samples drawn, using the
      inverse transform sampling method, from the
      (differential) discrete pdf stored in the
      histogram HISTDATA.

      HISTDATA[0] ---> x axis
      HISTDATA[1] ---> histogram in x

      If LOGBASE is different from None, it is assumed that
      the histogram is in logarithmic scale, with base
      LOGBASE (e.g. e for natural logarithm, 10 for decimal...)
    """

    if np.ndim(nsamples) == 0:
        n = nsamples
    else:
        n = nsamples.size

    x = np.array(histdata[0])
    y = np.array(histdata[1])
    d = x.size-y.size
    if d>0:
        x = x[1:]
    m = y>0

    if logbase is None:
        x = x[m]
        y = y[m]
    else:
        x = x[m]
        y = y[m]
        x = logbase**(x)
        y = logbase**(y)

    c = np.cumsum(y)
    c = c/c.max()

    r,m = np.unique(c,return_index=True)

    x = x[m]
    c = c[m]

    cdf  = interp1d(x,c,bounds_error=False,fill_value=(0,1))

    if toplot:
        plt.figure()
        plt.plot(x,cdf(x))
        plt.xlabel('x')
        plt.ylabel('cfd(x)')

    def cdf_root(a,b):
        t = cdf(a)-b
        if (a<=x.min())&(t>0):
            t = -t
        return t

    U    = np.random.rand(n)

    rvar = np.array([brentq(cdf_root,x.min(),x.max(),args=t) for t in U])

    if np.ndim(nsamples)>0:
        rvar = np.reshape(rvar,nsamples)

    return rvar
                    
def random_sky_coord(cshape=None):
    """
    Generates a random sky coordinate.      

    Parameters
    ----------
    cshape : tuple, optional
        The shape of the output array. The default is None.

    Returns
    -------
    SkyCoord
        The random sky coordinate.

    """

    if cshape is not None:
        phi   = 2*np.pi*np.random.rand(cshape)
        theta = np.arccos(2*np.random.rand(cshape)-1.0)
    else:
        phi   = 2*np.pi*np.random.rand()
        theta = np.arccos(2*np.random.rand()-1.0)
    ra    = phi
    dec   = np.pi/2-theta
    return SkyCoord(ra,dec,frame='icrs',unit=u.rad)

def color_noise_powlaw(size,rms,index):
    """
    Generates a random noise field with a power law power spectrum.

    Parameters
    ----------
    size : int
        The size of the output array.
    rms : float
        The rms of the noise.
    index : float
        The power law index.

    Returns
    -------
    noise : numpy array
        The random noise field.

    """
    
    x     = np.arange(0, size, 1, float)
    y     = x[:,np.newaxis]
    x0    = y0 = size//2
    r     = np.sqrt((x-x0)**2+(y-y0)**2)
    r[x0,y0] = 1.0
    PSmap = np.sqrt(np.power(r,-index))
    PSmap[x0,y0] = PSmap[x0,y0+1]

    noise = np.random.randn(size,size)
    fnois = np.fft.fftshift(np.fft.fft2(noise))
    fnois = np.multiply(fnois,PSmap)
    noise = np.fft.ifft2(np.fft.ifftshift(fnois)).real

    noise = noise-noise.mean()
    noise = noise/noise.std()
    noise *= rms

    return noise


# %%     CHI2 STATISTICS:

def vector_chi2(vect,covmat):
    """
    Calculates the chi2 value of a vector with respect to a covariance matrix.

    Parameters
    ----------
    vect : numpy array
        The vector.
    covmat : numpy array
        The covariance matrix.

    Returns
    -------
    float
        The chi2 value.

    """

    cinv = np.matrix(np.linalg.inv(covmat))
    x    = np.matrix(vect)
    return x*cinv*x.T


# %%     DISTANCE BETWEEN DISTRIBUTIONS:

def normal_Mahalanobis(mu1,cov1,mu2,cov2):
    """
    Calculates the Mahalanobis distance between two multi-variate
    normal distributions with mean vectors MU1, MU2 and covariance
    matrices COV1 and COV2

    Parameters
    ----------
    mu1 : numpy array
        The mean vector of the first distribution.
    cov1 : numpy array
        The covariance matrix of the first distribution.
    mu2 : numpy array
        The mean vector of the second distribution.
    cov2 : numpy array
        The covariance matrix of the second distribution.

    Returns
    -------
    float
        The Mahalanobis distance.

    """ 

    x      = np.matrix(mu1-mu2)
    Sigma  = np.matrix(cov1+cov2)
    iSigma = np.linalg.inv(Sigma)

    d      = x*(iSigma*x.T)

    return np.sqrt(d[0,0])


# %%     ARITHMETICS

def truncate_number(x,ndec):
    """
    Truncates a number to a given number of decimal places.

    Parameters
    ----------
    x : float
        The number to be truncated.
    ndec : int
        The number of decimal places.

    Returns
    -------
    float
        The truncated number.

    """

    from math import trunc
    f = 10**ndec
    return trunc(f*x)/f

def quantity_append(qt1,qt2):
    """
    Appends two astropy quantities.
    
    Parameters
    ----------
    qt1 : astropy quantity
        The first quantity.
    qt2 : astropy quantity  
        The second quantity.

    Returns
    -------
    astropy quantity
        The appended quantity.

    """
    return u.Quantity([x for x in qt1]+[y for y in qt2])

def msum(iterable):
    """
    Full precision summation using multiple floats for intermediate values.
    Rounded x+y stored in hi with the round-off stored in lo.  Together
    hi+lo are exactly equal to x+y.  The inner loop applies hi/lo summation
    to each partial so that the list of partial sums remains exact.
    Depends on IEEE-754 arithmetic guarantees.  See proof of correctness at:
    www-2.cs.cmu.edu/afs/cs/project/quake/public/papers/robust-arithmetic.ps
    
    Parameters
    ----------
    iterable : list
        The list of numbers to be summed.

    Returns
    -------
    float
        The sum of the numbers.

    """

    partials = []               # sorted, non-overlapping partial sums
    for x in iterable:
        i = 0
        for y in partials:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                partials[i] = lo
                i += 1
            x = hi
        partials[i:] = [x]
    return sum(partials, 0.0)


from math import frexp

def lsum(iterable):
    """
    Full precision summation using long integers for intermediate values.
    Transforms each float into a (mantissa, exponent) pair, then adds
    pairs with common exponents.  Because long integers have unlimited
    precision, this routine does not suffer from round-off error.
    
    Parameters
    ----------
    iterable : list
        The list of numbers to be summed.
    
    Returns
    -------
    float
        The sum of the numbers.

    """

    tmant, texp = np.long(0), 0
    for x in iterable:
        mant, exp = frexp(x)
        mant, exp = np.long(mant * 2.0 ** 53), exp-53
        if texp > exp:
            tmant <<= texp - exp
            texp = exp
        else:
            mant <<= exp - texp
        tmant += mant
    return float(str(tmant)) * 2.0 ** texp

from decimal import getcontext, Decimal, Inexact
getcontext().traps[Inexact] = True

def dsum(iterable):

    """
    Full precision summation using Decimal objects for intermediate values.
    Transforms each float x into a Decimal with value x, then adds the
    decimals together.
    Transform (exactly) a float to m * 2 ** e where m and e are integers.
    Convert (mant, exp) to a Decimal and add to the cumulative sum.
    If the precision is too small for exact conversion and addition,
    then retry with a larger precision.  
    
    Parameters
    ----------
    iterable : list
        The list of numbers to be summed.

    Returns
    -------
    float
        The sum of the numbers.

    """

    total = Decimal(0)
    for x in iterable:
        mant, exp = frexp(x)
        mant, exp = int(mant * 2.0 ** 53), exp-53
        while True:
            try:
                total += mant * Decimal(2) ** exp
                break
            except Inexact:
                getcontext().prec += 1
    return float(total)

from fractions import Fraction

def frsum(iterable):
    """
    Full precision summation using fractions for intermediate values.

    Parameters
    ----------
    iterable : list
        The list of numbers to be summed.

    Returns
    -------
    float
        The sum of the numbers.

    """

    return float(sum(map(Fraction.from_float, iterable)))

# %%    SKY COORDINATES

def coords_append(coord1,coord2):
    """
    Appends two lists of astropy sky coordinates.

    Parameters
    ----------
    coord1 : list of astropy SkyCoord
        The first list of coordinates.
    coord2 : list of astropy SkyCoord
        The second list of coordinates.

    Returns
    -------
    list of astropy SkyCoord
        The appended coordinates.

    """
    r = [v for v in coord1.icrs.ra]+[v for v in coord2.icrs.ra]
    d = [v for v in coord1.icrs.dec]+[v for v in coord2.icrs.dec]
    return SkyCoord(r,d,frame='icrs').flatten()

def coord2glonglat(c):
    """
    Converts a sky coordinate to a pair of Galactic longitude and latitude values in degrees.

    Parameters
    ----------
    c : astropy SkyCoord
        The sky coordinate.

    Returns
    -------
    lon : float
        The Galactic longitude in degrees.
    lat : float
        The Galactic latitude in degrees.

    """ 
    lat = c.galactic.b.deg
    lon = c.galactic.l.deg
    return lon,lat

def coord2thetaphi(c):
    """
    Converts a sky coordinate to a pair of colatitude and Galactic longitude values in degrees.

    Parameters
    ----------
    c : astropy SkyCoord
        The sky coordinate.

    Returns
    -------
    theta : float
        The colatitude in degrees.
    phi : float
        The Galactic longitude in degrees.

    """
    lon,lat = coord2glonglat(c)
    theta   = 90.0-lat
    phi     = lon
    return theta,phi

def table2skycoord(tabla):
    """
    Converts an astropy Table that contains, among other things, a couple of columns with
    sky coordinates to a list of astropy sky coordinates.  This routine tries to parse the
    names of the columns to find the sky coordinates among a list of possible names. 

    Parameters
    ----------
    tabla : astropy Table
        The table.

    Returns
    -------
    coords : list of astropy SkyCoord
        The list of sky coordinates.

    """

    ranames = ['RA','RAJ2000','RA_deg']
    denames = ['DEC','DEJ2000','DECJ2000','DEC_deg']
    lnames  = ['GLON']
    bnames  = ['GLAT']
    cols    = tabla.colnames

    isicrs  = False
    isgal   = False

    islon   = False
    islat   = False

    for c in cols:

        if not islon:
            if c in ranames:
                try:
                    unidad = tabla[c].unit.to_string()
                except AttributeError:
                    unidad = 'degrees'
                if unidad == 'degrees':
                    ra = tabla[c].data
                else:
                    try:
                        ra = tabla[c].to(u.deg).value
                    except TypeError:
                        ra = tabla[c].data
                    except UnitConversionError:
                        ra = tabla[c].data
                islon = True

        if not islat:
            if c in denames:
                try:
                    unidad = tabla[c].unit.to_string()
                except AttributeError:
                    unidad = 'degrees'
                if unidad == 'degrees':
                    dec = tabla[c].data
                else:
                    try:
                        dec = tabla[c].to(u.deg).value
                    except TypeError:
                        dec = tabla[c].data
                    except UnitConversionError:
                        dec = tabla[c].data
                islat  = True

    if islon or islat:
        isicrs = True

    if not isicrs:

        for c in cols:

            if not islon:
                if c in bnames:
                    try:
                        unidad = tabla[c].unit.to_string()
                    except AttributeError:
                        unidad = 'degrees'
                    if unidad == 'degrees':
                        b = tabla[c].data
                    else:
                        try:
                            b = tabla[c].to(u.deg).value
                        except TypeError:
                            b = tabla[c].data
                        except UnitConversionError:
                            b = tabla[c].data
                    islon = True

            if not islat:
                if c in lnames:
                    try:
                        unidad = tabla[c].unit.to_string()
                    except AttributeError:
                        unidad = 'degrees'
                    if unidad == 'degrees':
                        l = tabla[c].data
                    else:
                        try:
                            l = tabla[c].to(u.deg).value
                        except TypeError:
                            l = tabla[c].data
                        except UnitConversionError:
                            l = tabla[c].data
                    islat = True

        if islon or islat:
            isgal = True

    if isicrs:
        coords = SkyCoord(ra,dec,unit=u.deg)
    elif isgal:
        coords = SkyCoord(l=l,b=b,unit=u.deg,frame='galactic')
    else:
        try:
            coords = SkyCoord.guess_from_table(tabla)
        except ValueError:
            print(' Error: coordinate system not recognized')
            coords = np.array([])

    return coords

def read_image_coordinate(fname,timer=False):
    """
    Reads the coordinates of the center of a FITS image. It uses the WCS information
    stored in the FITS header.

    Parameters
    ----------
    fname : str
        The name of the FITS file.
    timer : bool, optional
        If True, it prints the time it takes to read the coordinates. The default is False.

    Returns
    -------
    c : astropy SkyCoord
        The sky coordinate of the center of the image.

    """

    t1        = time.time()

    hdulist   = fits.open(fname)
    header    = hdulist[0].header
    w         = wcs.WCS(header)

    hdulist.close()

    axis_type = w.axis_type_names

    try:
        unidad = header['CUNIT1']
    except KeyError:
        unidad = u.deg

    if axis_type[0] == 'RA':
        c = SkyCoord(w.wcs.crval[0],w.wcs.crval[1],unit=unidad,frame='icrs')
    else:
        c = SkyCoord(w.wcs.crval[0],w.wcs.crval[1],unit=unidad,frame='galactic')

    t2        = time.time()

    if timer:
        print(' --- Patch coordinates read in {0} seconds'.format(t2-t1))

    return c


def skycoord2file(coords,fname):
    """
    Writes a list of sky coordinates to a FITS file, using an astropoy QTable.  
    It writes the coordinates in both ICRS and Galactic frames.

    Parameters
    ----------
    coords : list of astropy SkyCoord
        The list of sky coordinates.
    fname : str
        The name of the FITS file.

    """
    t = QTable()
    t['RA']   = coords.icrs.ra
    t['DEC']  = coords.icrs.dec
    t['GLON'] = coords.galactic.l
    t['GLAT'] = coords.galactic.b
    t.write(fname,overwrite=True)

def coord2name(coords,frame=None,truncate=False):
    """
    Converts a list of sky coordinates to a list of names.  The names are
    the Galactic or equatorial coordinates in degrees, with the format
    XXX.XX+YY.YY or XXX.XX-YY.YY, where XXX.XX is the Galactic longitude
    or the right ascension and YY.YY is the Galactic latitude or the
    declination.  If TRUNCATE is True, the coordinates are truncated to
    two decimal places.

    Parameters
    ----------
    coords : list of astropy SkyCoord
        The list of sky coordinates.
    frame : str, optional
        The coordinate system. It can be 'G' for Galactic or 'E' for equatorial.
        The default is None.
    truncate : bool, optional
        If True, the coordinates are truncated to two decimal places. The default is False.

    Returns
    -------
    names : numpy array
        The list of names.

    """
    names = []
    if frame.upper()[0] == 'G':
        coords = coords.galactic

    for i in range(len(coords)):
        c   = coords[i]
        if frame.upper()[0] == 'G':
            glon = c.l.deg
            glat = c.b.deg
            if truncate:
                glon = truncate_number(glon,2)
                glat = truncate_number(glat,2)
            n   = '{:06.2f}'.format(glon)
            if glat>=0.0:
                m = '{:+06.2f}'.format(glat)
            else:
                m = '{:-06.2f}'.format(glat)
            names.append((n+m).replace(' ',''))
        else:
            ra  = c.ra.deg
            dec = c.dec.deg
            if truncate:
                ra  = truncate_number(ra,2)
                dec = truncate_number(dec,2)
            n   = '{:06.2f}'.format(ra)
            if dec>=0.0:
                m = '{:+06.2f}'.format(dec)
            else:
                m = '{:-06.2f}'.format(dec)
            names.append((n+m).replace(' ',''))
    return np.array(names)

def healpix2coord(nside,ipix,coordsys='G',nested=False):
    """
    Converts a HEALPix pixel to a sky coordinate.
    
    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    ipix : int
        The HEALPix pixel number.
    coordsys : str, optional
        The coordinate system. It can be 'G' for Galactic or 'E' for equatorial.
        The default is 'G'.
    nested : bool, optional
        If True, the HEALPix pixelization is nested. The default is False.

    Returns
    -------
    c : astropy SkyCoord
        The sky coordinate.

    """

    lon,lat = hp.pix2ang(nside,ipix,nest=nested,lonlat=True)
    if coordsys.upper()[0] == 'G':
        c = SkyCoord(lon,lat,unit=u.deg,frame='galactic')
    else:
        c = SkyCoord(lon,lat,unit=u.deg,frame='icrs')
    return c

def coord2healpix(nside,coord,coordsys='G',nested=False):
    """
    Converts a sky coordinate (or a list of sky coordinates) to 
    a HEALPix pixel (or a list of them).

    Parameters
    ----------
    nside : int
        The HEALPix nside parameter.
    coord : astropy SkyCoord or list of astropy SkyCoord
        The sky coordinate.
    coordsys : str, optional    
        The coordinate system. It can be 'G' for Galactic or 'E' for equatorial.
        The default is 'G'.
    nested : bool, optional
        If True, the HEALPix pixelization is nested. The default is False.

    Returns
    -------
    ipix : int or numpy array
        The HEALPix pixel number(s).

    """
    if coordsys.upper()[0] == 'G':
        lon = coord.galactic.l.deg
        lat = coord.galactic.b.deg
    else:
        lon = coord.icrs.ra.deg
        lat = coord.icrs.dec.deg
    ipix = hp.ang2pix(nside,lon,lat,nest=nested,lonlat=True)
    return ipix

def coord2vec(coord,coordsys='G'):
    """
    Converts a sky coordinate to a HEALPix vector.

    Parameters
    ----------
    coord : astropy SkyCoord
        The sky coordinate.
    coordsys : str, optional
        The coordinate system. It can be 'G' for Galactic or 'E' for equatorial.
        The default is 'G'.

    Returns
    -------
    vec : numpy array
        The HEALPix vector.

    """
    if coordsys.upper()[0] == 'G':
        lon = coord.galactic.l.deg
        lat = coord.galactic.b.deg
    else:
        lon = coord.icrs.ra.deg
        lat = coord.icrs.dec.deg
    vec = hp.ang2vec(lon,lat,lonlat=True)
    return vec


# %%   INTERPOLATION METHODS

def interpolate_between_arrays(z0,z,x,y):

    """
    Interpolate between two arrays of points Z1 and Z2, with
          Z1:  {x1[0],x1[1],...,x1[n]}, {y1[0],y1[1],...,y1[n]}
          Z2:  {x2[0],x2[1],...,x2[m]}, {y2[0],y2[1],...,y2[m]}
    for an intermediate point Z0.   The interpolation is done
    linearly in the y direction, and then in the x direction.
    The output is the interpolated value of y at Z0.
    This routine uses interp1d from scipy.interpolate.

    Parameters
    ----------
    z0 : float
        The intermediate point.
    z : numpy array
        The array of points.
    x : numpy array
        The x values.
    y : numpy array
        The y values.

    Returns
    -------
    y0 : float
        The interpolated value of y at Z0.

    """

    xmin = np.max((x[0].min(),x[1].min()))
    xmax = np.min((x[0].max(),x[1].max()))
    nx1  = np.count_nonzero((x[0]>=xmin)&(x[0]<=xmax))
    nx2  = np.count_nonzero((x[1]>=xmin)&(x[1]<=xmax))
    nx   = np.max((nx1,nx2))

    xout  = np.linspace(xmin,xmax,nx)

    f1   = interp1d(x[0],y[0])
    f2   = interp1d(x[1],y[1])

    y1   = f1(xout)
    y2   = f2(xout)

    dz   = z[1]-z[0]
    dy   = y2-y1

    yout = y1+(z0-z[0])*dy/dz

    return xout,yout

def positions_around(x,array):
    """
    Returns the positions of the elements around value X in the array ARRAY.  
    The array must be sorted.   

    Parameters
    ----------
    x : float
        The value.
    array : numpy array
        The array.

    Returns
    -------
    i1 : int
        The position of the element in the array that is smaller than X.
    i2 : int
        The position of the element in the array that is larger than X.

    """

    snu = array-x
    m1  = snu<0
    m2  = snu>0
    i1  = int(np.where(snu==snu[m1].max())[0])
    i2  = int(np.where(snu==snu[m2].min())[0])
    return i1,i2

# %%    ADVANCED FILE INPUT/OUTPUT

def save_object(obj,fname):
    """
    Saves an object to a file using pickle.

    Parameters
    ----------
    obj : object
        The object to be saved.
    fname : str
        The name of the file.

    """

    with open(fname+'.pkl','wb') as f:
        pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)

def load_object(fname):
    """
    Loads an object from a file using pickle.

    Parameters
    ----------
    fname : str
        The name of the file.

    Returns
    -------
    object
        The loaded object.

    """
    with open(fname+'.pkl','rb') as f:
        return pickle.load(f)

def write_blank(x):
    """
    Writes a blank line to the screen if the boolean X is True.

    Parameters
    ----------
    x : bool
        If True, the blank line is written.

    """
    if x:
        print(' ')

def save_ascii_list(lista,fname):
    """
    Saves a list to an ascii file.

    Parameters
    ----------
    lista : list
        The list to be saved.
    fname : str
        The name of the file.

    """

    with open(fname, 'w') as f:
        for item in lista:
            f.write('{0}\n'.format(item))

def append_ascii_list(lista,fname):
    """
    Appends a list to an ascii file.

    Parameters
    ----------
    lista : list
        The list to be appended.
    fname : str
        The name of the file.

    """
    with open(fname, 'a+') as f:
        for item in lista:
            f.write('{0}\n'.format(item))

def read_ascii_list(fname):
    """
    Reads an ascii file and returns a list.

    Parameters
    ----------
    fname : str
        The name of the file.

    Returns
    -------
    list
        The list.

    """
    with open(fname,'r',errors='replace') as f:
        lines = f.readlines()
    return [x.strip() for x in lines]

def read_ascii_column(filename,col_number,
                      delimiter = ' ',
                      header = None,
                      astype = float):
    """
    Reads a column from an ascii file.

    Parameters
    ----------
    filename : str
        The name of the file.
    col_number : int
        The column number.
    delimiter : str, optional
        The delimiter. The default is ' '.
    header : int, Sequence of int, ‘infer’ or None, default ‘infer’
        Row number(s) containing column labels and marking the start of the data (zero-indexed). 
        Default behavior is to infer the column names: if no names are passed the behavior is 
        identical to header=0 and column names are inferred from the first line of the file, 
        if column names are passed explicitly to names then the behavior is identical to
        header=None. Explicitly pass header=0 to be able to replace existing names. 
        The header can be a list of integers that specify row locations for a MultiIndex 
        on the columns e.g. [0, 1, 3]. Intervening rows that are not specified will be 
        skipped (e.g. 2 in this example is skipped). Note that this parameter ignores 
        commented lines and empty lines, so header=0 denotes the first line of data rather 
        than the first line of the file.
    astype : type, optional
        The data type. The default is float.

    Returns
    -------
    x : numpy array
        The column.

    """

    if header is not None:
        if header == 0:
            nheader = header
        else:
            nheader = header-1
    else:
        nheader = header


    if delimiter == ' ':
        t = pd.read_table(filename,usecols=[col_number],
                          delim_whitespace=True,
                          header=nheader)
    else:
        t = pd.read_table(filename,usecols=[col_number],
                          delimiter=delimiter,
                          header=nheader)

    x = np.array([t.values[i][0] for i in range(len(t))],dtype=astype)
    if (astype == str) or (astype == 'str'):
        x = np.array([np.unicode_(s.strip()) for s in x])

    return x

def add_array_line_csv(x,fname,header=None):
    """
    Adds a line to a csv file. The line is a numpy array.       

    Parameters
    ----------
    x : numpy array
        The array.
    fname : str
        The name of the file.
    header : str, optional
        The header. The default is None.

    """

    y = ''
    for numero in x:
        y += str(numero)+' , '
    y = y[:-3]
    y += '\n'

    if os.path.isfile(fname):
        with open(fname,'a') as f:
            f.write(y)
    else:
        with open(fname,'w') as f:
            if header is not None:
                f.write(header)
            f.write(y)

def search_in_file(fname,string):
    """
    Searches for a string in a file.

    Parameters
    ----------
    fname : str
        The name of the file.
    string : str
        The string.

    Returns
    -------
    sresults : list
        The list of lines in the file that contain the string.

    """

    with open(fname) as f:
        datafile = f.readlines()
    sresults = []
    for line in datafile:
        if string in line:
            sresults.append(line)
    return sresults

def clean_pdfs_in_LaTeX_dir(fdir,filename):

    fname = fdir+filename
    l = list_dir(fdir)
    l = [x for x in l if x.endswith('.pdf')]
    L = search_in_file(fname,'.pdf')
    L = [x.split('.pdf')[0].split('{')[-1]+'.pdf' for x in L]
    to_remove = []
    for x in l:
        sobra = True
        for y in L:
            if y in x:
                sobra = False
        if sobra:
            to_remove.append(x)
    for x in to_remove:
        os.remove(x)


# %%    FILE MODIFICATIONS

def change_dir_capitalization(directory):
    """
    Changes the capitalization of the directories in a given directory. It changes
    the names of the directories so that the first letter is capitalized and the
    rest are lower case. It also changes the names of the files inside the directories
    to match the new names of the directories.  It is useful when the directories
    are named using all capital letters.  It is recursive. It does not change the
    capitalization of the directories in the current directory.  

    Parameters
    ----------
    directory : str
        The name of the directory.

    """


    def reformat_dirname(dirname):
        nombre  = dirname.split('/')[-1]
        resto   = dirname.split('/')[:-1]
        path    = ''
        for r in resto:
            path += r+'/'
        nombre  = nombre.replace('___',' ')
        nombre  = nombre.replace('__',' ')
        nombre  = nombre.replace('_',' ')
        ln      = nombre.split(' ')
        nln     = [s.lower().capitalize() for s in ln]
        newname = ''
        for r in nln:
            newname += r+' '
        newname = newname[:-1]
        newdir  = path+newname
        newdir  = newdir.replace(' Y ',' y ')
        newdir  = newdir.replace(' De ',' de ')
        newdir  = newdir.replace(' En ',' en ')
        newdir  = newdir.replace(' El ',' el ')
        return newdir

    lista = sorted([x[0] for x in os.walk(directory)],key=len)
    for l in lista:
        os.rename(l,reformat_dirname(l))

def find_and_replace_text(directory,old_string,new_string,filePattern):
    """
    Finds and replaces a string in all the files in a given directory. It is recursive.

    Parameters
    ----------
    directory : str
        The name of the directory.
    old_string : str
        The string to be replaced.
    new_string : str
        The new string.
    filePattern : str
        The file pattern. It can be '*.txt' or '*.pdf', for example. 

    Example
    -------
    find_and_replace_text('/Users/herranz/Desktop/','antes','despues','*.txt')

    """

    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(old_string,new_string)
            with open(filepath,'w') as f:
                f.write(s)

def recursive_file_delete(parent_dir,fname):
    """
    Deletes recursively all the files with a given name in a given directory.

    Parameters
    ----------
    parent_dir : str
        The name of the directory.
    fname : str
        The name of the file.

    """
    for path, dirs, files in os.walk(os.path.abspath(parent_dir)):
        for filename in fnmatch.filter(files,fname):
            filepath = os.path.join(path,filename)
            os.remove(filepath)

# %%    FILE QUERIES

def list_dir(dir_name,ext=None):
    """
    Returns a list of files in a given directory. It is recursive. If EXT is not None,
    it returns only the files with that extension. 

    Parameters
    ----------
    dir_name : str
        The name of the directory.
    ext : str, optional
        The extension. The default is None.

    Returns
    -------
    files : list
        The list of files.

    """
    files = []
    for r,d,f in os.walk(dir_name):
        for file in f:
            if ext is None:
                files.append(os.path.join(r,file))
            else:
                if ext in file:
                    files.append(os.path.join(r,file))
    return files


def file_exists(fname):
    """
    Checks if a file exists.

    Parameters
    ----------
    fname : str
        The name of the file.

    Returns
    -------
    bool
        True if the file exists, False otherwise.

    """
    return os.path.isfile(fname)


# %%    ADVANCED ERROR BAR PLOTTING


def asymmetric_errorbar(x,y,xsamples=None,ysamples=None,nsigmas=1.0,
                        add_ymedian=False,
                        fmt='',median_fmt='o',
                        median_size=7,**kwargs):
    """
    Plots asymmetric error bars. The error bars are computed from the samples
    of the x and y values, and the confidence intervals are computed using
    confidence_limits for percentiles given by normal_confidence_interval.  The 
    median of the y samples can also be plotted. The error bars are plotted
    using the matplotlib errorbar function. 

    Parameters
    ----------
    x : float
        The x value.
    y : float
        The y value.
    xsamples : list of numpy arrays, optional
        The list of x samples. The default is None. 
    ysamples : list of numpy arrays, optional
        The list of y samples. The default is None.
    nsigmas : float, optional
        The number of sigmas for the confidence intervals. The default is 1.0.
    add_ymedian : bool, optional
        If True, the median of the y samples is plotted. The default is False.
    fmt : str, optional
        The format of the error bars. The default is ''.
    median_fmt : str, optional
        The format of the median. The default is 'o'.
    median_size : int, optional
        The size of the median filter. The default is 7.
    **kwargs : dict
        Additional arguments for the matplotlib errorbar function.

    """

    if xsamples is not None:
        l  = len(xsamples)
        xs = []
        for i in range(l):

            z = np.asarray(xsamples[i],dtype=np.float64)
            if z.size == 1:
                xs.append(float(z))
            else:
                lims    = confidence_limits(z,normal_confidence_interval(nsigmas))
                lims[0] = x[i]-lims[0]
                lims[1] = lims[1]-x[i]
                xs.append(lims)
        xs = np.asarray(xs,dtype=np.float64)
        xs = xs.transpose()
    else:
        xs = None

    if ysamples is not None:
        l    = len(ysamples)
        ys   = []
        ymed = []
        for i in range(l):
            z = np.asarray(ysamples[i],dtype=np.float64)
            if z.size == 1:
                ys.append(float(z))
            else:
                lims    = confidence_limits(z,normal_confidence_interval(nsigmas))
                lims[0] = y[i]-lims[0]
                lims[1] = lims[1]-y[i]
                ys.append(lims)
                ymed.append(np.median(z))
        ys = np.asarray(ys,dtype=np.float64)
        ys = ys.transpose()
    else:
        ys = None

    p   = plt.errorbar(x,y,yerr=ys,xerr=xs,fmt=fmt,**kwargs)
    col = p.get_children()[0].get_color()
    if add_ymedian:
        if ysamples is not None:
            plt.plot(x,ymed,median_fmt,color=col,
                     markersize=median_size,
                     markeredgewidth=0.5)

from matplotlib.patches      import Ellipse
import matplotlib.transforms as     transforms

def plot_confidence_ellipse(x0, y0,cov,ax,n_std=3.0,facecolor='none',**kwargs):

    """
    Create a plot of the covariance confidence ellipse for a data set
    centered at (x0,y0) with covariance matrix cov.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = x0

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = y0

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# %%  ADVANCED HISTOGRAM PLOTTING

def gaussian_density_plot(d,d_err,xlimits,
                          return_statistics=False,
                          return_density=False,**kwargs):
    """
    Plots the density of a set of points using a Gaussian kernel density estimator.
    It can also return the mean and median of the distribution.

    Parameters
    ----------
    d : numpy array
        The array of points.
    d_err : numpy array
        The array of errors.
    xlimits : tuple
        The x limits.   
    return_statistics : bool, optional
        If True, it returns the mean and median of the distribution. The default is False.
    return_density : bool, optional
        If True, it returns the x and y arrays of the density. The default is False.
    **kwargs : dict
        Additional arguments for the matplotlib fill_between function.

    Returns
    -------
    mean : float
        The mean of the distribution.
    median : float
        The median of the distribution.
    xarr : numpy array
        The x array of the density.
    yarr : numpy array
        The y array of the density.

    """

    n = d.size

    def gpdf(x,i):
        mu    = d[i]
        sigma = d_err[i]
        norm  = 1.0/(sigma*np.sqrt(2*np.pi))
        norm  = norm/n
        return norm*np.exp(-(x-mu)**2 / (2*sigma**2))

    def density(x):
        return np.array([gpdf(x,i) for i in range(n)]).sum()

    xarr    = np.linspace(xlimits[0],xlimits[1],10000)
    yarr    = np.array([density(xarr[i]) for i in range(xarr.size)])

    plt.fill_between(xarr,yarr,**kwargs)

    if return_statistics:
        intervl = xarr[1]-xarr[0]
        cdf     = np.cumsum(yarr)*intervl
        median  = xarr[np.where(cdf>0.5)[0][0]]
        mdf     = intervl*xarr*yarr
        mean    = mdf.sum()
        return mean,median

    if return_density:
        return xarr,yarr

# %%  DISTANCE TO ELEMENTS IN A MATRIX

def distance_matrix(i,j,shape):
    """
    Computes the distance of each element in a matrix to a given point.

    Parameters
    ----------
    i : int
        The row of the point.
    j : int
        The column of the point.
    shape : tuple
        The shape of the matrix.

    Returns
    -------
    d : numpy array
        The distance matrix. The shape is the same as the input shape. 

    """
    n = shape[0]
    m = shape[1]
    x = np.arange(0,m,1,float)
    y = np.arange(0,n,1,float)[:,np.newaxis]
    d = np.sqrt((x-j)**2+(y-i)**2)
    return d

# %%  MATRIX CROPPING / PADDING

def img_shapefit(img1,img2):

    """
    Forces image1 to have the same shape as image2. If image1 was larger than
    image2, then it is cropped in its central part. If image1 was smaller that
    image2, then it is padded with zeros. Dimensions must have even size
    """

    if img1.ndim == 2:
        (n1,n2) = img1.shape
    else:
        n1 = np.sqrt(img1.size)
        n2 = np.sqrt(img1.size)
    (m1,m2) = img2.shape
    (c1,c2) = (n1//2,n2//2)
    (z1,z2) = (m1//2,m2//2)

    img3 = np.zeros((m1,m2),dtype=img1.dtype)

    if n1<=m1:
        if n2<=m2:
            img3[z1-c1:z1+c1,z2-c2:z2+c2] = img1        # Standard padding
        else:
            img3[z1-c1:z1+c1,:] = img1[:,c2-z2:c2+z2]
    else:
        if n2<=m2:
            img3[:,z2-c2:c2+z2] = img1[c1-z1:c1+z1,:]
        else:
            img3 = img1[c1-z1:c1+z1,c2-z2:c2+z2]

    return img3

# %%  NESTED DICTIONARIES

from collections import defaultdict

def nested_dict(n):
    """
    Creates a nested dictionary with N levels.

    Parameters
    ----------
    n : int
        The number of levels.

    Returns
    -------
    defaultdict
        The nested dictionary.

    """
    if n == 1:
        return defaultdict()
    else:
        return defaultdict(lambda: nested_dict(n-1))


# %%  ARRAY MANIPULATION

def as_array(x):
    """
    Converts a number to a numpy array.

    Parameters
    ----------
    x : float
        The number.

    Returns
    -------
    numpy array
        The array.

    """
    return np.asarray(x).reshape(1, -1)[0,:]

# %%  BEAM CONVERSION UTILITIES

def fwhm_to_area(fwhm):
    """
    Converts a FWHM to an area.

    Parameters
    ----------
    fwhm : float
        The FWHM.

    Returns
    -------
    area : float
        The area.

    """
    return 2*np.pi*(fwhm*fwhm2sigma)**2

def area_to_fwhm(area):
    """
    Converts an area to a FWHM.

    Parameters
    ----------
    area : float
        The area.

    Returns
    -------
    fwhm : float
        The FWHM.

    """
    sigma = np.sqrt(area/(2*np.pi))
    return sigma*sigma2fwhm

# %%  FAST COMPUTATIONS USING NUMBA

import numba

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    """
    Computes the squared modulus of a complex number.

    Parameters
    ----------
    x : complex
        The complex number.

    Returns
    -------
    float
        The squared modulus.

    """
    return x.real**2 + x.imag**2
