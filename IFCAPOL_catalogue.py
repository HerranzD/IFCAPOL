#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:48:51 2022

@author: herranz
"""

import numpy             as np
import matplotlib.pyplot as plt
import astropy.units     as u

from skimage.feature     import peak_local_max
from astropy.stats       import sigma_clip
from astropy.table       import Column,Table,vstack
from astropy.coordinates import SkyCoord
from myutils             import fwhm2sigma,table2skycoord


# %%  MATCHED FILTERING OF A PATCH AND EXTRACTION OF PEAKS ABOVE A CERTAIN THRESHOLD


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
            - Dist: the distance between the peak and the center of the patch.


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
    c8 = Column(data=patch.center_coordinate.separation(co).to(u.deg),name='Dist [deg]')
    t  = Table([c1,c2,c3,c4,c5,c6,c7,c8])

    return t

# %%  CATALOGUE OPERATIONS

def split_and_clean_catalogue(input_table,radius):
    """
    Given a table that contains a **RA** and **DEC** columns with the Right
    Ascension and Declination, both in degrees, of a number of objects,
    this routine returns two sub-tables:

        - The first row of the table.
        - A sub-table with all the rest of the objects that are not inside a circle of radius equal to the given `radius` parameter.

    Parameters
    ----------
    input_table : `~astropy.Table`
        A table of objects. This table should be already sorted in decreasing
        order of a given quantity with respect to which it is desirable to
        rank the sources (for example, the signal-to-noise ratio). In addition,
        the table must contain two columns, one with label **RA** with the
        Right Ascension coordinate of the sources (in degrees) and other with
        label **DEC** with the Declination coordinate (also in degrees).
    radius : `~astropy.Quantity`
        The cross-match radius to remove possible repetitions.

    Returns
    -------
    first : `~astropy.Table`
        The first object in the input table.
    rest : `~astropy.Table`
        All the rest of objects in the input table that are at an angular
        distance from the first object larger than `radius`.

    """

    work_table = input_table.copy()

    if len(work_table) == 1:
        first = work_table
        rest  = first[:0].copy()
    else:
        first  = work_table[0]
        cfirst = SkyCoord(first['RA'],first['DEC'],frame='icrs',unit=u.deg)
        last   = work_table[1:]
        clast  = SkyCoord(last['RA'],last['DEC'],frame='icrs',unit=u.deg)
        dist   = cfirst.separation(clast)
        rest   = last[dist>=radius]

    return first,rest

def remove_repeated_positions(input_table,radius):
    """
    Cleans a table of astronomical objects by cross-matching it with itself
    and removing all possible repetitions within a certain radius. Only the
    first element of any possible association group within that radius is
    kept. Therefore, the table should be alredady ranked in order of
    preference (for example, by signal-to-noise ratio or any other criterion).

    Parameters
    ----------
    input_table : `~astropy.Table`
        A table of objects. This table should be already sorted in decreasing
        order of a given quantity with respect to which it is desirable to
        rank the sources (for example, the signal-to-noise ratio). In addition,
        the table must contain two columns, one with label **RA** with the
        Right Ascension coordinate of the sources (in degrees) and other with
        label **DEC** with the Declination coordinate (also in degrees).
    radius : `~astropy.Quantity`
        The cross-match radius to remove possible repetitions.

    Returns
    -------
    `~astropy.Table`
        A Table cleaned from possible repetions of objects within the search
        radius.

    """

    lrows = []
    L     = 10
    r     = input_table.copy()

    while L > 0:

        f,r = split_and_clean_catalogue(r, radius)
        lrows.append(f)
        L   = len(r)

    return vstack(lrows)


# %%  BLIND SEARCH ACROSS ALL THE SKY

def blind_survey(sky_map,fwhm,fname,threshold=3.0,verbose=False):
    """
    Runs a blind search for sources of a given FWHM and over a certain
    signal-to-noise ratio on a given sky map.

    Parameters
    ----------
    sky_map : `Fitsmap`
        Input sky map (only temperature)
    fwhm : `~astropy.Quantity`
        The FWHM of the compact sources.
    fname : string
        File name for the output catalogue of detections.
    threshold : float, optional
        The sigma (signal-to-noise) detection threshold. The default is 3.0.
    verbose : bool, optional
        If True, the routine writes some basic information during runtime.
        The default is False.

    Returns
    -------
    out_table : `~astropy.Table`
        An ``~astropy.Table` containing astrometric and photometric information
        about the detections. See `patch_analysis` documentation for more
        information about the format of and columns of this table.

    """

    from fits_maps import Fitsmap

    nside0  = 8
    ltables = []
    vac     = Fitsmap.empty(nside0)

    for ic in range(vac.npix):

        if verbose:
            print('Blind search: ',ic,vac.npix)

        coord = vac.pixel_to_coordinates(ic)
        ltables.append(patch_analysis(sky_map,
                                      coord,
                                      fwhm,
                                      threshold=threshold))

    out_table = vstack(ltables)
    out_table.sort(keys='SNR',reverse=True)
    out_table.write(fname,overwrite=True)

    return out_table

# %%  NON-BLIND SEARCH ACROSS ALL THE SKY

def non_blind_survey(sky_map,blind_survey_fname,
                     xclean     = 2.0,
                     clean_mode = 'after',
                     snrcut     = 3.5,
                     verbose    = False):
    """
    Runs a non-blind polarization detection/estimation pipeline on a
    list of previously selected targets.

    Parameters
    ----------
    sky_map : `Fitsmap`
        Input sky map (I,Q,U)
    blind_survey_fname : str
        The file name of the table of targets. An ouput table is automatically
        generated using the same file name but adding the '_IFCAPOL' string
        at the end of the file name.
    xclean : float, optional
        The radius employed for cleaning the output catalogue of overlapping
        repetitions is defined as source profile sigma times `xclean`.
        The default is 2.0.
    clean_mode : str, optional
        Can take values in {'after','before'}. If 'before', the overlap cleaning
        is performed before the non-blind analysis. If 'after', the cleaning
        is performed after the non-blind analysis, using the new signal-to-noise
        ratio as reference. The default is 'after'.
    snr_cut : float, optional
        The signal to noise ratio (SNR) in intensity at which to cut the
        catalogue, that is, the effective sigma detection threshold.
    verbose : bool, optional
        If True, some basic info is written on screen during runtime.
        The default is False.

    Returns
    -------
    out_tabl : `~astropy.Table`
        A Table containing the non-blind catalogue. See the `IFCAPOL.Source.info`
        documentation for additional information on the format and columns
        of this output table. The Table is automatically saved to a file
        named as the `blind_survey_fname` parameter, but adding the '_cleaned'
        string at the end of the file name.

    """


    import IFCAPOL as pol

    blind   = Table.read(blind_survey_fname)
    dist    = sky_map.fwhm[0]*fwhm2sigma*xclean

    if clean_mode == 'before':
        cleaned = remove_repeated_positions(blind,dist)
    else:
        cleaned = blind.copy()

    coords  = table2skycoord(cleaned)
    seps    = cleaned['Dist [deg]']

    outpl   = []

    for i in range(len(cleaned)):

        if verbose:
            print(' Analysing source {0} of {1}'.format(i,len(cleaned)))
        s = pol.Source.from_coordinate(sky_map,coords[i])
        d = s.info(include_coords=True,ID=i+1)
        d['Separation from centre [deg]'] = seps[i]
        outpl.append(d)

    out_tabl = Table(outpl)
    if clean_mode == 'after':

        bln        = out_tabl.copy()
        bln['RA']  = bln['RA [deg]'].copy()
        bln['DEC'] = bln['DEC [deg]'].copy()
        bln.sort(keys='I SNR',reverse=True)

        temp       = remove_repeated_positions(bln, dist)
        out_tabl   = temp.copy()

    fname    = blind_survey_fname.replace('.fits','_{0}_cleaned.fits'.format(clean_mode))

    ttotal   = out_tabl.copy()
    out_tabl = ttotal[ttotal['I SNR']>=snrcut]

    out_tabl.write(fname,overwrite=True)

    return out_tabl




