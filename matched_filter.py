# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 21:15:49 2016

   This program filters an image stored in the array DATA
   and filters it with a matched fiter for a profile template
   TPROF or a Gaussian shaped template profile with full width
   at half maximum FWHM (in pixel units). It has been tested to
   give a similar performance to the Fortran90
   implementation by D. Herranz. Running time has not been compared.

   Optionally, images can be padded (with a reflection of the
   data inside) in order to reduce border effects, with a cost
   in running time. This option is activated by default.

@author: herranz
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import gauss2dfit as gfit
from gauss_window import makeGaussian,makeAnalyticalGaussian
from radial_profile import radial_profile
from skimage.feature import peak_local_max
from myutils import abs2,isscalar

# %% The matched filter itself:

def matched_filter(data0,
                   gprof                   = True,
                   lafwhm                  = 2.0,
                   tprof0                  = None,
                   resample_factor         = 1,
                   nbins                   = 50,
                   aplot                   = False,
                   topad                   = True,
                   kind                    = 'linear',
                   return_filtered_profile = False,
                   input_PS                = None,
                   output_PS               = None,
                   verbose                 = False):
    """
    Matched filter of a 2D data array (image)

    Parameters
    ----------
    data0 : `array_like`
        A two-dimensional array (numpy or any compatible data type) containing
        the image to be filtered with a matched filter.
    gprof : bool, optional
        If True, the source spatial profile is assumed to be a circularly 
        symmetric Gaussian, whose width parameter will be speficied in the
        `lafwhm` parameter. If False, the source spatial profile must be
        provided as a two-dimensional array (image), using the `tprof0` 
        parameter. The default is True.
    lafwhm : float, optional
        This parameters provides the source spatial profile Full 
        Width at Half Maximum (FWHM) in pixel units (i.e. FWHM = 3.5 pixels). 
        The source is assumed to have a Gaussian profile.
        The default is 2.0.
    tprof0 : `array_like`, optional
        This parameter contains an image of the source spatial profile. 
        It must have the same size and shape as `data0`. This parameter is only 
        used if `gprof` is set to False. If `gprof` is True, this parameter
        will be ignored. The default is None.
    resample_factor : int, optional
        The resampling factor for the evaluation of the source spatial profile. 
        This parameter makes it possible to create Gaussian profiles with a
        higher resolution than that of the `data0` image. This may be useful 
        in some occasions, particularly when the FHWM is not much larger than
        the pixel scale. The default is 1 (no resampling).
    nbins : int, optional
        The number of bins in wavenumber used for the averaging of the power
        spectrum of the signal. The default is 50.
    aplot : bool, optional
        If True, the routine generates a plot of the non-filtered and the 
        filtered image. The default is False.
    topad : bool, optional
        If True, the `data0` image is zero-padded in the borders for the purpose
        of Fourier Transform and power spectrum estimation. The default is True.
    kind : str, optional
        Mode of interpolation of the estimated power spectrum. The default is 'linear'.
    return_filtered_profile : bool, optional
        If True, the routine returns an image of the filtered source profile 
        (and only the source profile) instead of the filtered image. The 
        default is False.
    input_PS : `array_like`, optional
        If this parameter is provided (instead of having the default None value),
        the routine skips the part of estimating the power spectrum of the 
        `data0` image and it uses this array as an image of the power spectrum. 
        This is useful if we know in advance the statistical properties of the
        noise power spectrum (for example, white noise). This array must have
        the same size and shape as `data0`. The default is None.
    output_PS : bool, optional
        If True, the routine returns a two-dimensional array containing an image
        of the estimated noise power spectrum. The default is None.
    verbose : bool, optional
        If True, the code writes on screen some basic information during 
        execution. The default is False.

    Returns
    -------
    `array_like` or a dictionary with two `array_like` elements.
        If `output_PS` is False, the routine returns a two-dimensional array
        containing the matched filtered version of the data in `data0`. If 
        `output_PS` is True, the routine returns a dictionary with two
        elements:
            - **'Filtered'** contains the filtered version of `data0`.
            - **'PowerSpectrum'** contains a map of the (smoothed) estimated power spectrum of the noise.

    """

    start_time = time.time()

    s = data0.shape
    size0 = s[0]

    if verbose:
        print("--- Image size %d x %d pixels ---" % (size0,size0))

    if topad:
        l    = 2*int(lafwhm)
        padscheme = ((l,l),(l,l))
        data = np.pad(data0,padscheme,mode='reflect')
        if tprof0 is not None:
            tprof = np.pad(tprof0,padscheme,mode='reflect')
    else:
        l = 0
        data = np.copy(data0)

    s = data.shape
    size = s[0]

    if gprof:
        g  = makeGaussian(size,fwhm=lafwhm,
                          resample_factor=resample_factor)
    else:
        if not topad:
            tprof = tprof0.copy()
        g = tprof

    gf = np.abs(np.fft.fftshift(np.fft.fft2(g)))

    fdata = np.fft.fftshift(np.fft.fft2(data))

    if input_PS is not None:
        P = input_PS
    else:
        P = abs2(fdata)

    profmap,xp,yp = radial_profile(P,nbins,
                                   toplot=aplot,
                                   kind=kind,
                                   datype='real')

    filtro        = np.divide(gf,profmap)
    pc            = size//2
    filtro[pc,pc] = 0.0*filtro[pc,pc]

    fparanorm = np.multiply(gf,filtro)
    paranorm = np.real(np.fft.ifft2(np.fft.ifftshift(fparanorm)))
    normal = np.amax(paranorm)/np.amax(g)
    filtro = filtro/normal

    ffiltrada = np.multiply(fdata,filtro)

    filtrada_pad = np.real(np.fft.ifft2(np.fft.ifftshift(ffiltrada)))

    filtrada = filtrada_pad[l:l+size0,l:l+size0]

    if aplot:
        plt.figure()
        plt.pcolormesh(data[l:l+size0,l:l+size0])
        plt.axis('tight')
        plt.title('Original data')
        plt.colorbar()

        plt.figure()
        plt.pcolormesh(filtrada)
        plt.axis('tight')
        plt.title('Filtered data')
        plt.colorbar()

    if verbose:
        print("--- Matched filtered in %s seconds ---" % (time.time() - start_time))

    if return_filtered_profile:
        filtrada    = filtro
                # it is returned in Fourier space with padding

    if output_PS is not None:
        return {'Filtered':filtrada,
                'PowerSpectrum':P}
    else:
        return filtrada


def iterative_matched_filter(data0,
                             lafwhm=2.0,
                             resample_factor=1,
                             nbins=50,
                             snrcut=5.0,
                             aplot=False,
                             topad=True,
                             kind='linear',
                             return_filtered_profile=False,
                             verbose=False):
    """
    Basic iterative matched filter. This routine has the limitation of
    working only with ideal Gaussian, symmetric source profiles. This routine
    invokes `matched_filter`, identifies bright peaks above a certain 
    threshold, makes a LSQ fitting to these peaks, removes the best fit
    from the data and then re-computes the estimation power spectrum of the 
    image noise using the data from which the peaks have been removed.

    Parameters
    ----------
    data0 : `array_like`
        A two-dimensional array (numpy or any compatible data type) containing
        the image to be filtered with a matched filter.
    lafwhm : float, optional
        This parameters provides the source spatial profile Full 
        Width at Half Maximum (FWHM) in pixel units (i.e. FWHM = 3.5 pixels). 
        The source is assumed to have a Gaussian profile.
        The default is 2.0.
    resample_factor : int, optional
        The resampling factor for the evaluation of the source spatial profile. 
        This parameter makes it possible to create Gaussian profiles with a
        higher resolution than that of the `data0` image. This may be useful 
        in some occasions, particularly when the FHWM is not much larger than
        the pixel scale. The default is 1 (no resampling).
    nbins : int, optional
        The number of bins in wavenumber used for the averaging of the power
        spectrum of the signal. The default is 50.
    aplot : bool, optional
        If True, the routine generates a plot of the non-filtered and the 
        filtered image. The default is False.
    snrcut : float, optional
        The signal-to-noise ratio at which the data is thresholded 
        (after the first filtering) in order to look for bright peaks. 
        The default is 5.0.
    topad : bool, optional
        If True, the `data0` image is zero-padded in the borders for the purpose
        of Fourier Transform and power spectrum estimation. The default is True.
    kind : str, optional
        Mode of interpolation of the estimated power spectrum. The default is 'linear'.
    return_filtered_profile : bool, optional
        If True, the routine returns an image of the filtered source profile 
        (and only the source profile) instead of the filtered image. The 
        default is False.
    verbose : bool, optional
        If True, the code writes on screen some basic information during 
        execution. The default is False.

    Returns
    -------
    `array_like`, `array_like`.
    The first array is a version of the `data0` filtered by a standard, 
    non-iterative matched filter. The second array is a version of the `data0` 
    array, filtered by the iterative matched filter. If 
    `return_filtered_profile` is True, a filtered version of the source profile
    is returned instead.

    """
    

    start_time = time.time()

    lasigma = lafwhm/(2.0*np.sqrt(2.0*np.log(2.0)))

    s = data0.shape
    size0 = s[0]

    if verbose:
        print("--- Image size %d x %d pixels ---" % (size0,size0))

    if topad:
        l    = 2*int(lafwhm)
        padscheme = ((l,l),(l,l))
        data = np.pad(data0,padscheme,mode='reflect')
    else:
        l = 0
        data = np.copy(data0)

    s    = data.shape
    size = s[0]
    pc   = size//2

#   Gaussian profile (in real and Fourier space):

    g  = makeGaussian(size,fwhm=lafwhm,resample_factor=resample_factor)
    gf = np.abs(np.fft.fftshift(np.fft.fft2(g)))

#   FFT and power spectrum map of the original data:

    fdata            = np.fft.fftshift(np.fft.fft2(data))
    P0               = np.abs(fdata)**2
    profmap0,xp0,yp0 = radial_profile(P0,nbins,toplot=aplot,kind=kind)
#    profmap0[profmap0==0] = 1

#   First iteration of the normalised filter:

    filtro1        = np.divide(gf,profmap0)
    filtro1[pc,pc] = 0.0
    fparanorm      = np.multiply(gf,filtro1)
    paranorm       = np.real(np.fft.ifft2(np.fft.ifftshift(fparanorm)))
    normal         = np.amax(paranorm)/np.amax(g)
    filtro1        = filtro1/normal

#   First filtering:

    ffiltrada1     = np.multiply(fdata,filtro1)
    filtrada1_pad  = np.real(np.fft.ifft2(np.fft.ifftshift(ffiltrada1)))
    filtrada1      = filtrada1_pad[l:l+size0,l:l+size0]

#   Detection of the peaks above a cut in SNR,
#    fit to a Gaussian around that regions and substraction from the
#    input map

    picos  = peak_local_max(filtrada1,min_distance=int(lafwhm),
                            threshold_abs=snrcut*filtrada1.std(),
                            exclude_border=int(lafwhm))
    npicos = len(picos)

    if npicos<1:

        filtrada   = filtrada1  # It isn't necessary to iterate
        if verbose:
            print(' ')
            print(' ---- No peaks above threshold in the filtered image ')
            print(' ')
        f = 0

    else:

        data1 = np.copy(data0)

        if verbose:
            print(' ')
            print(' ---- {0} peaks above threshold '.format(npicos))
            print(' ')

        for pico in range(npicos):

#   We select a patch around that peak

            lugar = picos[pico,:]

            nx   = int(2.5*lafwhm)
            xinf = int(lugar[0])-nx
            if xinf<0:
                xinf = 0
            xsup = int(lugar[0])+nx
            if xsup>(size0-1):
                xsup = size0-1
            ny   = int(2.5*lafwhm)
            yinf = int(lugar[1])-ny
            if yinf<0:
                yinf = 0
            ysup = int(lugar[1])+ny
            if ysup>(size0-1):
                ysup = size0-1

#   We fit to a Gaussian profile with fixed width in that patch

            patch = data1[xinf:xsup,yinf:ysup]
            if resample_factor == 1:
                f = gfit.fit_single_peak(patch,toplot=aplot,
                                         fixwidth=True,fixed_sigma=lasigma)
            else:
                f = gfit.fit_single_pixelized_peak(patch,
                                                   resample_factor=resample_factor,
                                                   toplot=aplot,fixwidth=True,
                                                   fixed_sigma=lasigma)

#   We subtract the fitted Gaussian from a copy of the original data

            data1[xinf:xsup,yinf:ysup] = data1[xinf:xsup,yinf:ysup] - f.gaussmap

#   Plot the cleaned map:

        if aplot:
            plt.figure()
            plt.pcolormesh(data1)
            plt.axis('tight')
            plt.colorbar()
            plt.title('Original data with brightests sources removed')

#   Second interation of the filter:

        if topad:
            l    = 2*int(lafwhm)
            padscheme = ((l,l),(l,l))
            data2 = np.pad(data1,padscheme,mode='reflect')
        else:
            l = 0
            data2 = data1

#   FFT and power spectrum map of the original data:

        fdata2           = np.fft.fftshift(np.fft.fft2(data2))
        P1               = np.abs(fdata2)**2
        profmap1,xp1,yp1 = radial_profile(P1,nbins,toplot=aplot,kind=kind)

#   Second iteration of the normalised filter:

        filtro2        = np.divide(gf,profmap1)
        filtro2[pc,pc] = 0.0
        fparanorm      = np.multiply(gf,filtro2)
        paranorm       = np.real(np.fft.ifft2(np.fft.ifftshift(fparanorm)))
        normal         = np.amax(paranorm)/np.amax(g)
        filtro2        = filtro2/normal

#   Second filtering:

        ffiltrada2     = np.multiply(fdata,filtro2)
        filtrada2_pad  = np.real(np.fft.ifft2(np.fft.ifftshift(ffiltrada2)))
        filtrada       = filtrada2_pad[l:l+size0,l:l+size0]


    if aplot:
        plt.figure()
        plt.pcolormesh(data[l:l+size0,l:l+size0])
        plt.axis('tight')
        plt.title('Original data')
        plt.colorbar()

        plt.figure()
        plt.pcolormesh(filtrada)
        plt.axis('tight')
        plt.title('Filtered data')
        plt.colorbar()

    if verbose:
        print("--- Matched filtered in %s seconds ---" % (time.time() - start_time))

    if return_filtered_profile:
        filtrada = paranorm/np.amax(paranorm)

    return filtrada1,filtrada


# %% New iterative method (2021)

def get_PowerSpectrum(image,pad_size=0):

    if pad_size > 0:
        l         = 2*int(pad_size)
        padscheme = ((l,l),(l,l))
        data      = np.pad(image,padscheme,mode='reflect')
    else:
        l          = 0
        data      = np.copy(image)

#   FFT and power spectrum map of the original data:

    fdata            = np.fft.fftshift(np.fft.fft2(data))
    P0               = np.abs(fdata)**2

    return P0

def mf_step(imagen,perfil,PS,nbins):

    data          = imagen['Real']
    if 'Fourier' in imagen.keys():
        fft_data          = imagen['Fourier']
    else:
        fft_data          = np.fft.fftshift(np.fft.fft2(data))
        imagen['Fourier'] = fft_data


    profile       = perfil['Real']
    if 'Fourier' in perfil.keys():
        fft_profile       = perfil['Fourier']
    else:
        fft_profile       = np.fft.fftshift(np.fft.fft2(profile))
        perfil['Fourier'] = fft_profile

    psmap,xp,yp   = radial_profile(PS,nbins,
                                   toplot=False,
                                   kind='linear',
                                   datype='real')

    filtro        = np.divide(fft_profile,psmap)
    pc            = int(np.sqrt(filtro.size))//2
    filtro[pc,pc] = 0.0*filtro[pc,pc]

    fparanorm     = np.multiply(fft_profile,filtro)
    paranorm      = np.real(np.fft.ifft2(np.fft.ifftshift(fparanorm)))
    normal        = np.amax(paranorm)/np.amax(profile)
    filtro        = filtro/normal
    perfil['Filtered'] = paranorm/normal

    ffiltrada     = np.multiply(fft_data,filtro)
    filtrada      = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ffiltrada))))
    imagen['Filtered'] = filtrada

    return imagen.copy()

def new_iterative_matched_filter(imagen,fwhm,
                                 nbins  = 50,
                                 snrcut = 5.0,
                                 topad  = True,
                                 kind   = 'linear',
                                 toplot = None):

#   Size and padding

    n   = int(np.sqrt(imagen.size))
    img = {}

    if isscalar(fwhm):
        l = 2*int(fwhm)
    else:
        l = 6

    if topad:
        padscheme   = ((l,l),(l,l))
        img['Real'] = np.pad(imagen,padscheme,mode='reflect')
        m           = int(np.sqrt(img['Real'].size))
    else:
        img['Real'] = imagen.copy()
        m           = n

#    Profile

    profile = {}
    if isscalar(fwhm):
        profile['Real']     = makeAnalyticalGaussian(m,fwhm_pix=fwhm)
    else:
        if m == n:
            profile['Real'] = fwhm.copy()
        else:
            profile['Real'] = np.pad(fwhm,padscheme,mode='reflect')
    profile['Fourier'] = np.fft.fftshift(np.fft.fft2(profile['Real']))

#    Iterative process:

    npeaks = 100
    imiter = img.copy()
    niter  = 0
    PS     = get_PowerSpectrum(imiter['Real'])

    if toplot is not None:
        plt.figure()
        plt.imshow(imiter['Real'][l:m-l,l:m-l])
        plt.colorbar()
        plt.title('Iteration {0}'.format(niter))

    while npeaks > 0 and niter<20:

        niter  = niter+1

        imiter = mf_step(imiter,profile,PS,nbins)

        peaks  = peak_local_max(imiter['Filtered'],
                                min_distance=l,
                                threshold_abs=snrcut*imiter['Filtered'].std(),
                                exclude_border=l)

        npeaks1 = len(peaks)

        if npeaks1 > 0:
            z = imiter['Filtered'][peaks[:,0],peaks[:,1]]
            for i in range(npeaks1):
                imiter['Real'] = imiter['Real'] - z[i]*makeAnalyticalGaussian(m,fwhm_pix=fwhm,
                                                                              center=(peaks[i,0]+0.5,
                                                                                      peaks[i,1]+0.5))


        peaks  = peak_local_max(-imiter['Filtered'],
                                min_distance=l,
                                threshold_abs=snrcut*imiter['Filtered'].std(),
                                exclude_border=l)

        npeaks2 = len(peaks)

        if npeaks2 > 0:
            z = imiter['Filtered'][peaks[:,0],peaks[:,1]]
            for i in range(npeaks2):
                imiter['Real'] = imiter['Real'] - z[i]*makeAnalyticalGaussian(m,fwhm_pix=fwhm,
                                                                              center=(peaks[i,0]+0.5,
                                                                                      peaks[i,1]+0.5))

        PS                = get_PowerSpectrum(imiter['Real'])
        imiter.pop('Fourier', None)

        npeaks = npeaks1+npeaks2

        if toplot is not None:
            plt.figure()
            plt.imshow(imiter['Real'][l:m-l,l:m-l])
            plt.colorbar()
            plt.title('Iteration {0}'.format(niter))

    imout = mf_step(img,profile,PS,nbins)

    imout['Real']     = imout['Real'][l:m-l,l:m-l]
    imout['Filtered'] = imout['Filtered'][l:m-l,l:m-l]
    imout['Fourier']  = imout['Fourier'][l:m-l,l:m-l]

    return imout





