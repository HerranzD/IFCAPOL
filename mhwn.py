# -*- coding: utf-8 -*-
"""
Created on Feb 5 2024

@author: herranz
"""

import numpy             as np
import matplotlib.pyplot as plt
import time

from gauss_window        import makeGaussian
from myutils             import fwhm2sigma,sigma2fwhm
from scipy.optimize      import minimize_scalar


# %%    FAST 2D FILTERING THROUGH FFT

def fft_filter(image,kernel,timer=False):
    """
    Fast 2D filtering through FFT. Assuming that both image and kernel are
    Fourier space representations of the original image and the filter,
    respectively, obtained through numpy's fft2() routine, and that the
    filter kernel has been shifted with np.fft.fftshift, the filtered image
    preserves the same size, shape and axis orientation as the
    original image. The center of the filtered image is located at the same
    position as the center of the original image. The filterd image is real.

    Parameters
    ----------
    image : `array_like`
        A two-dimensional complex array (numpy or any compatible data type)
        containing the image, in Fourier space, to be filtered.
    kernel : `array_like`
        A two-dimensional complex array (numpy or any compatible data type)
        containing the kernel, in Fourier space, to be used as a filter.
    timer : bool, optional
        If True, the routine returns the time it took to perform the filtering.
        The default is False.

    Returns
    -------
    `array_like` or a tuple with two elements.
        If `timer` is False, the routine returns a two-dimensional array
        containing the filtered image. If `timer` is True, the routine returns
        a tuple with two elements:
            - The first element is a two-dimensional array containing the
                  filtered image.
            - The second element is a float number containing the time it
                took to perform the filtering.

    """

    if timer:
        start_time = time.time()

    fft_filtered_image = np.multiply(image,kernel)
    filtered_image     = np.fft.ifft2(fft_filtered_image)
    filtered_image     = np.fft.ifftshift(filtered_image)
    filtered_image     = np.fft.ifftshift(filtered_image)
    filtered_image     = np.real(filtered_image)

    if timer:
        run_time = time.time() - start_time
        return filtered_image,run_time
    else:
        return filtered_image

# %%    MEXICAN HAT WAVELET FAMILY (MHWn)

def mhwn(q,R,n):
    """
    Non-normalized radial Mexican Hat wavelet of order n function and
    scale R (in pixel units), in Fourier space.

    Parameters
    ----------
    q : `array_like`
        An array containing the wavenumber in Fourier space.
    R : float
        The scale of the wavelet in pixel units.
    n : int
        The order of the wavelet.

    Returns
    -------
    `array_like`
        An array containing the radial Mexican Hat wavelet
        of order n function in Fourier space.

    """

    return (q**(2*n))*np.exp(-0.5*(q*R)**2)

def mhwn_2D(image_size,R,n=2,timer=False,toplot=False):
    """
    Non-normalized 2D Mexican Hat wavelet of order n function and
    scale R (in pixel units), in Fourier space.

    Parameters
    ----------
    image_size : int
        The size of the image in pixels.
    R : float
        The scale of the wavelet in pixel units.
    n : int, optional
        The order of the wavelet. The default is 2.
    timer : bool, optional
        If True, the routine returns the time it took to create the
        wavelet. The default is False.
    toplot : bool, optional
        If True, the routine generates a plot of the wavelet.
        The default is False.

    Returns
    -------
    `array_like` or a tuple with two elements.
        If `timer` is False, the routine returns a two-dimensional array
        containing the Mexican Hat wavelet. If `timer` is True, the routine
        returns a tuple with two elements:
            - The first element is a two-dimensional array containing the
              Mexican Hat wavelet.
            - The second element is a float number containing the time it
              took to create the wavelet.

    """

    if timer:
        start_time = time.time()

    y  = np.arange(0, image_size, 1, float)
    x  = y[:,np.newaxis]
    x0 = image_size//2
    y0 = image_size//2
    r  = np.sqrt((x-x0)**2 + (y-y0)**2)

    w  = mhwn(r,R*2*np.pi/image_size,n)
    w  = np.fft.fftshift(w)

    if toplot:
        plt.figure()
        plt.pcolormesh(w)
        plt.axis('tight')
        plt.colorbar()
        plt.title('Mexican Hat Wavelet')

    if timer:
        run_time = time.time() - start_time
        return w,run_time
    else:
        return w

def real_mhwn_2D(image_size,R,n=2,timer=False,toplot=False):
    """
    Non-normalized 2D Mexican Hat wavelet of order n function and
    scale R (in pixel units), in real space.

    Parameters
    ----------
    image_size : int
        The size of the image in pixels.
    R : float
        The scale of the wavelet in pixel units.
    n : int, optional
        The order of the wavelet. The default is 2.
    timer : bool, optional
        If True, the routine returns the time it took to create the
        wavelet. The default is False.
    toplot : bool, optional
        If True, the routine generates a plot of the wavelet.
        The default is False.

    Returns
    -------
    `array_like` or a tuple with two elements.
        If `timer` is False, the routine returns a two-dimensional array
        containing the Mexican Hat wavelet. If `timer` is True, the routine
        returns a tuple with two elements:
            - The first element is a two-dimensional array containing the
              Mexican Hat wavelet.
            - The second element is a float number containing the time it
              took to create the wavelet.

    """

    if timer:
        start_time = time.time()
        w,t        = mhwn_2D(image_size,R,n=n,
                             timer=timer,toplot=toplot)
    else:
        w          = mhwn_2D(image_size,R,n=n,
                             timer=timer,toplot=toplot)

    if toplot:
        plt.title('FFT MHW')

    w_real = np.real(np.fft.ifftshift(np.fft.ifft2(w)))

    if toplot:
        plt.figure()
        plt.pcolormesh(w_real)
        plt.axis('tight')
        plt.colorbar()
        plt.title('Real MHW')

    if timer:
        run_time = time.time() - start_time
        return w_real,run_time
    else:
        return w_real


def normalized_mhwn_2D(image_size,R,fft_beam,n=2,timer=False):
    """
    Normalized 2D Mexican Hat wavelet of order n function and
    scale R (in pixel units), in Fourier space.

    Parameters
    ----------
    image_size : int
        The size of the image in pixels.
    R : float
        The scale of the wavelet in pixel units.
    fft_beam : `array_like` or `float`
        The Fourier space representation of the beam. If it is a scalar,
        it is assumed that the beam is a circularly symmetric Gaussian
        with a sigma width equal to the value of `fft_beam`.
        If it is an array, it is assumed that the beam in real space
        is normalized to one.
    n : int, optional
        The order of the wavelet. The default is 2.
    timer : bool, optional
        If True, the routine returns the time it took to create the
        wavelet. The default is False.

    Returns
    -------
    `array_like` or a tuple with two elements.
        If `timer` is False, the routine returns a two-dimensional array
        containing the Mexican Hat wavelet. If `timer` is True, the routine
        returns a tuple with two elements:
            - The first element is a two-dimensional array containing the
              Mexican Hat wavelet.
            - The second element is a float number containing the time it
              took to create the wavelet.

    """

    if timer:
        start_time = time.time()

    # checks if fft_beam is an scalar or an array
    if np.isscalar(fft_beam):
        real_beam   = makeGaussian(image_size,fwhm=fft_beam*sigma2fwhm)
        real_beam   = real_beam/real_beam.max()
        beam_kernel = np.fft.fft2(real_beam)
    else:
        beam_kernel = fft_beam  # it is assumed that the beam in real space is normalized to one

    fft_mwh = mhwn_2D(image_size,R,n=n,timer=False,toplot=False)

    fgauss  = fft_filter(beam_kernel,fft_mwh)

    if timer:
        run_time = time.time() - start_time
        return fft_mwh/fgauss.max(),run_time
    else:
        return fft_mwh/fgauss.max()


# %%    OPTIMAL SCALE MHWn FILTERING

def mhwn_filter(image,beam,order=2,toplot=False,timer=False):

    if timer:
        start_time = time.time()

    image_size = image.shape[0]

    # FFT of the image
    fft_image = np.fft.fft2(image)

    # checks if fft_beam is an scalar or an array
    if np.isscalar(beam):
        real_beam   = makeGaussian(image_size,fwhm=beam*sigma2fwhm)
        real_beam   = real_beam/real_beam.max()
        beam_kernel = np.fft.fft2(real_beam)
    else:
        beam_kernel = beam  # it is assumed that the beam in real space is normalized to one

    def snr(R):
        w = normalized_mhwn_2D(image_size,R,beam_kernel,n=order)
        f = fft_filter(fft_image,w)
        return f.std()

    optimal_R = minimize_scalar(snr, bounds=(0.01, 20), method='bounded')

    w = normalized_mhwn_2D(image_size,
                           optimal_R.x,
                           beam_kernel,
                           n=order,
                           timer=False)

    filtered_image = fft_filter(fft_image,w)

    if toplot:
        plt.figure()
        plt.pcolormesh(image)
        plt.axis('tight')
        plt.colorbar()
        plt.title('Original Image')

        plt.figure()
        plt.pcolormesh(filtered_image)
        plt.axis('tight')
        plt.colorbar()
        plt.title('Filtered Image')

    output_dict = {'filtered_image':filtered_image,
                   'optimal_R':optimal_R.x,
                   'optimization output':optimal_R,
                   'wavelet':w,
                   'gain':image.std()/filtered_image.std()}

    if timer:
        run_time = time.time() - start_time
        output_dict['run_time'] = run_time

    return output_dict



