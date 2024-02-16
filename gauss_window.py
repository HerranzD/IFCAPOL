"""
Created by D. Herranz, 2016




"""

import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.transform import downscale_local_mean


def makeGaussian(size,fwhm=3,
                 resample_factor=1,
                 center=None,
                 verbose=False,
                 toplot=False):

    """

    Makes a square image of a Gaussian kernel, returned as a numpy
    two-dimensional array. The Gaussian takes a maximum value = 1
    at the position defined in 'center'

    PARAMETERS:
        'size'    is the length of a side of the square
        'fwhm'    is full-width-half-maximum, in pixel units, of the
                     Gaussian kernel.
        'resample_factor' indicates how to increase (or not, if equal
                     to one) the image
        'center'  is a tuple,list or numpy array containing the (x,y)
                     coordinates (in pixel units) of the centre of the
                     Gaussian kernel. If certer=None, the Gaussian is
                     placed at the geometrical centre of the image
        'verbose' if true, the routine writes out the info about the
                     code
        'toplot'  if true, plots the output array

    """

    start_time = time.time()

    if center is None:
        x0 = y0 = resample_factor*size // 2
    else:
        x0 = resample_factor*center[0]
        y0 = resample_factor*center[1]

    if verbose:
        print(' ')
        print(' --- Generating a {0}x{0} image with a Gaussian kernel of FWHM = {1} pixels located at ({2},{3})'.format(size,fwhm,x0/resample_factor,y0/resample_factor))

    y = np.arange(0, resample_factor*size, 1, float)
    x = y[:,np.newaxis]                # This couple of lines generates a very efficient
                                       # structure of axes (horizontal and vertical)
                                       # that can be filled very quickly with a function
                                       # such as the Gaussian defined in the next line.
                                       # This is far much faster than the old-fashioned
                                       # nested FOR loops.

    u = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / (resample_factor*fwhm)**2)

    if resample_factor != 1:
        u = downscale_local_mean(u,(resample_factor,resample_factor))

    if toplot:
        plt.figure()
        plt.imshow(u)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')

    if verbose:
        print(' --- Gaussian template generated in {0} seconds'.format(time.time() - start_time))

    return u

def make_map_of_Gaussians(size,xyz,fwhm=3,verbose=False,toplot=False):
    """
    
    Makes a square image of a sum of Gaussian kernels, returned as a numpy two-dimensional array. 
    
    Parameters
    ----------
    size : int
        is the length of a side of the square
    xyz : array
        contains the (x,y,z) coordinates (in pixel units) of the centre of the Gaussian kernels
    fwhm : float
        is full-width-half-maximum, in pixel units, of the Gaussian kernels.
    verbose : bool
        if True, the routine writes out the info about the code. Default is False
    toplot : bool
        if True, plots the output array. Default is False

    Returns
    -------
    image : array
        2D array with the sum of Gaussian kernels
    
    """

    start_time = time.time()
    image = np.zeros((size,size))
    for i in range(xyz.shape[0]):
        image += xyz[i,2]*makeGaussian(size,fwhm=fwhm,center=xyz[i,0:2])
    if toplot:
        plt.figure()
        plt.imshow(image)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
    if verbose:
        print(' --- Sum of Gaussian templates generated in {0} seconds'.format(time.time() - start_time))
    return image

# %% Analytical expression for pixelized Gaussian

from scipy.special    import erf
from myutils          import fwhm2sigma,sigma2fwhm
from astropy.modeling import custom_model

@custom_model
def pixelized_Gaussian(x,y,amplitude=1.0,x_0=0.0,y_0=0.0,sigma_pix=1.0):
    """
    Two dimensional pixelixed Gaussian function. 
    
    Parameters
    ----------
    x : array
        x-coordinate of the pixel
    y : array
        y-coordinate of the pixel
    amplitude : float
        amplitude of the Gaussian
    x_0 : float
        x-coordinate of the center of the Gaussian
    y_0 : float
        y-coordinate of the center of the Gaussian
    sigma_pix : float
        standard deviation of the Gaussian in pixel units

    Returns
    -------
    r : array
        pixelized Gaussian
   
    """
    
    fwhm = sigma2fwhm*sigma_pix   
    return amplitude*analytical_pixelized_Gaussian(x,y,(x_0,y_0),fwhm_pix=fwhm)
    

def analytical_pixelized_Gaussian(i,j,center,fwhm_pix=3.0):
    """
    Analytical expression for a pixelized Gaussian function. 
    The Gaussian is normalized to have a maximum value of 1.0 
    at the center of the Gaussian.

    Parameters
    ----------
    i : array
        x-coordinate of the pixel
    j : array
        y-coordinate of the pixel
    center : tuple
        (x,y) coordinates of the center of the Gaussian
    fwhm_pix : float
        FWHM of the Gaussian in pixel units

    Returns
    -------
    r : array
        pixelized Gaussian

    """

    x = i-center[0]
    y = j-center[1]
    s = fwhm_pix*fwhm2sigma
    c = s*np.sqrt(2.0)
    
    r = erf(x/c)-erf((1+x)/c)
    r = r*(erf(y/c)-erf((1+y)/c))
    r = r*0.5*np.pi*(s**2)   

    return r    

def makeAnalyticalGaussian(size,fwhm_pix = 3.0,
                           center        = None):
    """
    Makes a square image of a pixelized Gaussian kernel, returned as a numpy
    two-dimensional array. The Gaussian takes a maximum value = 1
    at the position defined in 'center'

    Parameters
    ----------
    size : int
        is the length of a side of the square
    fwhm_pix : float
        FWHM of the Gaussian in pixel units
    center : tuple
        (x,y) coordinates of the center of the Gaussian. If center=None, the Gaussian is
        placed at the geometrical centre of the image (default is None)

    Returns
    -------
    m : array
        2D array with the pixelized Gaussian

    """ 
    
    y = np.arange(0, size, 1, float)
    x = y[:,np.newaxis]
    
    if center is None:
        centro = (size/2+0.5,size/2+0.5)
    else:
        centro = center
    
    m = analytical_pixelized_Gaussian(x,y, 
                                      center   = centro, 
                                      fwhm_pix = fwhm_pix)
    
    return m
    
    
    
    

