# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import ligo.skymap.plot
import healpy            as hp
import astropy.units     as u
import matplotlib.colors as colors

from matplotlib import pyplot as plt

def skyview(data,
            coord,
            coordsys  = None,
            zoom_size = 4*u.deg,
            cmap      = 'rainbow',
            title     = None,
            tofile    = None):
    """
    This routine uses the ligo.skymap visualization tools to make a spherical
    plot of a healpix map and extract a square zoom around a given coordinate.

    Parameters
    ----------
    data : str or array
        If string, the data to be visualized is read from the file name given
        here. If an array, then the data is stored in the array. In this case,
        the data is assumed to be a Healpix map in equatorial coordinates.
        For arrays in Galactic coordinates, the coordys keyword below must be
        set to 'G'.
    coord : `~astropy.coordinates.SkyCoord`
        Sky coordinate where the plot will be centered. It is also used
        as center of the zooming region.
    coordsys : str or None
        Coordinate system. The coordinate system is automatically determined
        from the file header if `data` is a file name. If not, the coordinate
        system is determined by this keyword. The default is None, in which case
        the equatorial system is assumed. The other possibility is Galactic
        ('G','GAL','Gal','GALACTIC','Galactic','galactic') coordinates.
    zoom_size : `~astropy.units.quantity.Quantity`
        The size of the zoom. The default is 4*u.deg.
    cmap : src, optional
        The colormap. The default is 'rainbow'.
    title : str, optional
        Title of the plot. The default is None.
    tofile : str or None, optional
        If not None, the plot is saved to this file name. The default is None.

    Returns
    -------


    """

    res = isinstance(data,str)

    def get_normalization():
        if res:
            mapa  = hp.read_map(data)
        else:
            mapa  = data.copy()
        linth = mapa.std()/2.0
        return colors.SymLogNorm(linthresh=linth,
                                 linscale=0.8,
                                 vmin=mapa.min(),
                                 vmax=mapa.max(),
                                 base=10)

    norma = get_normalization()

    fig = plt.figure(figsize=(8, 8), dpi=100)

    ax = plt.axes(
        [0.05, 0.05, 0.9, 0.9],
        projection='astro globe',
        center=coord)

    ax_inset = plt.axes(
        [0.59, 0.3, 0.4, 0.4],
        projection='astro zoom',
        center=coord,
        radius=zoom_size)

    for key in ['ra', 'dec']:
        ax_inset.coords[key].set_ticklabel_visible(False)
        ax_inset.coords[key].set_ticks_visible(False)

    ax.grid()
    ax.mark_inset_axes(ax_inset)
    ax.connect_inset_axes(ax_inset, 'upper left')
    ax.connect_inset_axes(ax_inset, 'lower left')
    ax_inset.scalebar((0.1, 0.1), zoom_size/2).label()
    ax_inset.compass(0.9, 0.1, 0.2)

    if res:
        ax.imshow_hpx(data, cmap=cmap,norm=norma)
    else:
        if coordsys in ['G','GAL','Gal','GALACTIC','Galactic','galactic']:
            rot_gal2eq = hp.Rotator(coord="GC")
            mapa       = rot_gal2eq.rotate_map_pixel(data)
        else:
            mapa       = data.copy()
        ax.imshow_hpx(mapa, cmap=cmap,norm=norma)

    if title is not None:
        ax.set_title(title,fontsize=16)

    if res:
        ax_inset.imshow_hpx(data, cmap=cmap)
    else:
        ax_inset.imshow_hpx(mapa, cmap=cmap)

    ax_inset.plot(
        coord.icrs.ra.deg, coord.icrs.dec.deg,
        transform=ax_inset.get_transform('world'),
        marker=ligo.skymap.plot.reticle(),
        markersize=30,
        markeredgewidth=3)

    if tofile is not None:
        fig.savefig(tofile)


