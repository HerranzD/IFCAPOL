#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:48:51 2022

@author: herranz
"""

import numpy             as np
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from astropy.stats   import sigma_clip
from astropy.table   import Column,Table,vstack
from myutils         import fwhm2sigma,table2skycoord
from catalogue_tools import clean_repetitions

def patch_analysis(sky_map,coord,fwhm,threshold=3.0,border=10,sclip=3.0):
    """

    Projects a square flat patch from a sky map around a given coordinate,
    filters the patch using a recursive matched filter, locates local
    peaks in the filtered image above a given number of sigmas (excluding
    a certain number of pixels at the border) a return a table of results.

    Parameters
    ----------
    sky_map : `Fitsmap`
        Input sky map.
    coord : `~astropy.SkyCoord`
        Coordinate around which the search is done.
    fwhm : `~astropy.units.quantity.Quantity`
        FWHM for the matched filter.
    threshold : float, optional
        Number of standard deviations over which peaks are looked for.
        The default is 3.0.
    border : int, optional
        Number of pixels to exclude in the border for both calculating
        statistics and searching local peaks. The default is 10.
    sclip : float, optional
        Sigma clipping level applied for the calculation of the matched
        filtered image rms. The default is 3.0.

    Returns
    -------
    t : `~astropy.Table`

        A table containing the following columns:

            - I: the i pixel index in the image data matrix
            - J: the j pixel index in the image data matrix
            - Intensity: the value of the matched filtered image at (i,j)
            - Intensity error: the rms of the matched filtered image
            - SNR: the signal to noise ratio of the peak
            - RA: the corresponding source RA (deg)
            - DEC: the corresponding source DEC (deg)


    """


    from IFCAPOL import define_npix,image_resampling

    # gnomonic projection of a patch around the coordinate `coord`

    plt.ioff()
    patch = sky_map.patch(coord,
                          define_npix(sky_map.nside),
                          resampling=image_resampling)
    plt.ion()

    # matched filtering of the patch (iterative)

    temp,mfpatch  = patch.iter_matched(fwhm=fwhm,toplot=False)

    # selection of a region without borders for the computation of statistics

    noborder = mfpatch.stamp_central_region(mfpatch.lsize-2*border).datos

    # sigma-clipped estimation of the standard deviation of the patch

    sigma    = sigma_clip(noborder,sigma=sclip,maxiters=10).std()

    # peak search on the whole image, above a given threshold

    detection_threshold = mfpatch.datos.mean()+threshold*sigma
    mindist             = np.max((1,int((fwhm2sigma*fwhm/patch.pixsize).si.value)))
    peaks               = peak_local_max(mfpatch.datos,
                                         min_distance   = mindist,
                                         threshold_abs  = detection_threshold,
                                         exclude_border = border,
                                         indices        = True)

    c1 = Column(data=[x[0] for x in peaks],name='I')
    c2 = Column(data=[x[1] for x in peaks],name='J')
    c3 = Column(data=[mfpatch.datos[x[0],x[1]] for x in peaks],name='Intensity')
    c4 = Column(data=[sigma for x in peaks],name='Intensity error')
    c5 = Column(data=[mfpatch.datos[x[0],x[1]]/sigma for x in peaks],name='SNR')
    co = patch.pixel_coordinate(c1,c2)
    c6 = Column(data=co.icrs.ra.deg,name='RA')
    c7 = Column(data=co.icrs.dec.deg,name='DEC')

    t  = Table([c1,c2,c3,c4,c5,c6,c7])

    return t

def get_parent_catalogue(sky_map,fwhm,fname,threshold=3.0):

    from fits_maps import Fitsmap

    nside0  = 8
    ltables = []
    vac     = Fitsmap.empty(nside0)

    for ic in range(vac.npix):

        print(ic,vac.npix)

        coord = vac.pixel_to_coordinates(ic)
        ltables.append(patch_analysis(sky_map,
                                      coord,
                                      fwhm,
                                      threshold=threshold))

    out_table = vstack(ltables)
    out_table.sort(keys='SNR',reverse=True)
    out_table.write(fname,overwrite=True)

    return out_table


def non_blind_survey(sky_map,blind_survey_fname,xclean=1.5,verbose=False):

    import IFCAPOL as pol

    blind   = Table.read(blind_survey_fname)
    dist    = sky_map.fwhm[0]*fwhm2sigma*xclean
    cleaned = clean_repetitions(blind,dist)
    coords  = table2skycoord(cleaned)

    outpl   = []

    for i in range(len(cleaned)):

        if verbose:
            print(' Analysing source {0} of {1}'.format(i,len(cleaned)))
        s = pol.Source.from_coordinate(sky_map,coords[i])
        outpl.append(s.info(include_coords=True,ID=i+1))

    out_tabl = Table(outpl)
    fname    = blind_survey_fname.replace('.fits','_IFCAPOL.fits')
    out_tabl.write(fname,overwrite=True)

    return out_tabl




