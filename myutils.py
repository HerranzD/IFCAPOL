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

    z  = func(x,*pars)
    sl = y-l
    su = u-y
    d  = np.zeros(z.size)
    d[y>=z] = ((y[y>=z]-z[y>=z])/sl[y>=z])**2
    d[y<z]  = ((y[y<z]-z[y<z])/su[y<z])**2

    return d.sum()

def asymm_lsq_fit(func,x,y,l,u,p0=None):

    res = minimize(asymm_chi2,p0,args=(func,x,y,l,u))

    return res

def plot_confidence_band(func,x,popt,pcov,nstd=1,alpha=0.15,noplot=False):

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
    cinv = np.matrix(np.linalg.inv(covmat))
    x    = np.matrix(vect)
    return x*cinv*x.T


# %%     DISTANCE BETWEEN DISTRIBUTIONS:

def normal_Mahalanobis(mu1,cov1,mu2,cov2):

    # Calculates the Mahalanobis distance between two multi-variate
    #   normal distributions with mean vectors MU1, MU2 and covariance
    #   matrices COV1 and COV2

    x      = np.matrix(mu1-mu2)
    Sigma  = np.matrix(cov1+cov2)
    iSigma = np.linalg.inv(Sigma)

    d      = x*(iSigma*x.T)

    return np.sqrt(d[0,0])


# %%     ARITHMETICS

def truncate_number(x,ndec):
    from math import trunc
    f = 10**ndec
    return trunc(f*x)/f

def quantity_append(qt1,qt2):
    return u.Quantity([x for x in qt1]+[y for y in qt2])

def msum(iterable):
    "Full precision summation using multiple floats for intermediate values"
    # Rounded x+y stored in hi with the round-off stored in lo.  Together
    # hi+lo are exactly equal to x+y.  The inner loop applies hi/lo summation
    # to each partial so that the list of partial sums remains exact.
    # Depends on IEEE-754 arithmetic guarantees.  See proof of correctness at:
    # www-2.cs.cmu.edu/afs/cs/project/quake/public/papers/robust-arithmetic.ps

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
    "Full precision summation using long integers for intermediate values"
    # Transform (exactly) a float to m * 2 ** e where m and e are integers.
    # Adjust (tmant,texp) and (mant,exp) to make texp the common exponent.
    # Given a common exponent, the mantissas can be summed directly.

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
    "Full precision summation using Decimal objects for intermediate values"
    # Transform (exactly) a float to m * 2 ** e where m and e are integers.
    # Convert (mant, exp) to a Decimal and add to the cumulative sum.
    # If the precision is too small for exact conversion and addition,
    # then retry with a larger precision.

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
    "Full precision summation using fractions for itermediate values"
    return float(sum(map(Fraction.from_float, iterable)))

# %%    SKY COORDINATES

def coords_append(coord1,coord2):
    r = [v for v in coord1.icrs.ra]+[v for v in coord2.icrs.ra]
    d = [v for v in coord1.icrs.dec]+[v for v in coord2.icrs.dec]
    return SkyCoord(r,d,frame='icrs').flatten()

def coord2glonglat(c):
    lat = c.galactic.b.deg
    lon = c.galactic.l.deg
    return lon,lat

def coord2thetaphi(c):
    lon,lat = coord2glonglat(c)
    theta   = 90.0-lat
    phi     = lon
    return theta,phi

def table2skycoord(tabla):

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
    t = QTable()
    t['RA']   = coords.icrs.ra
    t['DEC']  = coords.icrs.dec
    t['GLON'] = coords.galactic.l
    t['GLAT'] = coords.galactic.b
    t.write(fname,overwrite=True)

def coord2name(coords,frame=None,truncate=False):

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
    lon,lat = hp.pix2ang(nside,ipix,nest=nested,lonlat=True)
    if coordsys.upper()[0] == 'G':
        c = SkyCoord(lon,lat,unit=u.deg,frame='galactic')
    else:
        c = SkyCoord(lon,lat,unit=u.deg,frame='icrs')
    return c

def coord2healpix(nside,coord,coordsys='G',nested=False):
    if coordsys.upper()[0] == 'G':
        lon = coord.galactic.l.deg
        lat = coord.galactic.b.deg
    else:
        lon = coord.icrs.ra.deg
        lat = coord.icrs.dec.deg
    ipix = hp.ang2pix(nside,lon,lat,nest=nested,lonlat=True)
    return ipix

def coord2vec(coord,coordsys='G'):
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
    Interpola entre dos pares de arrays
          Z1:  {x1[0],x1[1],...,x1[n]}, {y1[0],y1[1],...,y1[n]}
          Z2:  {x2[0],x2[1],...,x2[m]}, {y2[0],y2[1],...,y2[m]}
    para un punto intermedio Z0

a    """

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
    snu = array-x
    m1  = snu<0
    m2  = snu>0
    i1  = int(np.where(snu==snu[m1].max())[0])
    i2  = int(np.where(snu==snu[m2].min())[0])
    return i1,i2

# %%    ADVANCED FILE INPUT/OUTPUT

def save_object(obj,fname):
    with open(fname+'.pkl','wb') as f:
        pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)

def load_object(fname):
    with open(fname+'.pkl','rb') as f:
        return pickle.load(f)

def write_blank(x):
    if x:
        print(' ')

def save_ascii_list(lista,fname):
    with open(fname, 'w') as f:
        for item in lista:
            f.write('{0}\n'.format(item))

def append_ascii_list(lista,fname):
    with open(fname, 'a+') as f:
        for item in lista:
            f.write('{0}\n'.format(item))

def read_ascii_list(fname):
    with open(fname,'r',errors='replace') as f:
        lines = f.readlines()
    return [x.strip() for x in lines]

def read_ascii_column(filename,col_number,
                      delimiter = ' ',
                      header = None,
                      astype = float):

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

    # example: find_and_replace('/Users/herranz/Desktop/','antes','despues','*.txt')

    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(old_string,new_string)
            with open(filepath,'w') as f:
                f.write(s)

def recursive_file_delete(parent_dir,fname):
    for path, dirs, files in os.walk(os.path.abspath(parent_dir)):
        for filename in fnmatch.filter(files,fname):
            filepath = os.path.join(path,filename)
            os.remove(filepath)

# %%    FILE QUERIES

def list_dir(dir_name,ext=None):
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
    return os.path.isfile(fname)


# %%    ADVANCED ERROR BAR PLOTTING


def asymmetric_errorbar(x,y,xsamples=None,ysamples=None,nsigmas=1.0,
                        add_ymedian=False,
                        fmt='',median_fmt='o',
                        median_size=7,**kwargs):

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
    if n == 1:
        return defaultdict()
    else:
        return defaultdict(lambda: nested_dict(n-1))


# %%  ARRAY MANIPULATION

def as_array(x):
    return np.asarray(x).reshape(1, -1)[0,:]

# %%  BEAM CONVERSION UTILITIES

def fwhm_to_area(fwhm):
    return 2*np.pi*(fwhm*fwhm2sigma)**2

def area_to_fwhm(area):
    sigma = np.sqrt(area/(2*np.pi))
    return sigma*sigma2fwhm

# %%  FAST COMPUTATIONS USING NUMBA

import numba

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2
