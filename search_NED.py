#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 11:29:29 2017

@author: herranz
"""

from astropy.coordinates import SkyCoord
from astroquery.ned import Ned
from astropy.coordinates.name_resolve import NameResolveError
from astroquery.exceptions import TimeoutError,AstropyWarning
import astropy.units as u
import numpy as np

Ned.TIMEOUT = 120

def find_type(coord,radio,tipo,tofile=None):
    """
    Find objects of a given type in a given area in a NED search. The
    search is done using the astroquery package.

    Parameters
    ----------
    coord : SkyCoord
        Coordinates of the center of the search area
    radio : Quantity
        Radius of the search area
    tipo : string
        Type of object to search for
    tofile : string
        Name of the output file. If None, no file is written

    Returns
    -------
    x : astropy.table.Table
        Table with the results of the search

    """

    tabla = Ned.query_region(coord,radius=radio)
    N = len(tabla)
    print(' ')
    print(' --- There are {0} NED objects listed in the area '.format(N))
    x = tabla[tabla['Type']==tipo]
    print(' --- There are {0} NED objects of type {1} listed in the area '.format(len(x),tipo))
    x.sort('Separation')

    return x


def find_nearest_object_type(coord,radio,tipo,tofile=None):
    """
    Find the nearest object of a given type in a given area in a NED search. The
    search is done using the astroquery package.

    Parameters
    ----------
    coord : SkyCoord
        Coordinates of the center of the search area
    radio : Quantity
        Radius of the search area
    tipo : string
        Type of object to search for
    tofile : string
        Name of the output file. If None, no file is written

    Returns
    -------
    x : astropy.table.Table
        Table with the results of the search

    """

    tabla = Ned.query_region(coord, radius=radius)
    lista_nombres = np.array(tabla['Object Name'])
    lista_tipos = np.array(tabla['Type'])
    mask = lista_tipos == tipo
    M = np.count_nonzero(mask)
    print(f'--- There are {len(tabla)} NED objects listed in the area')
    print(f'--- There are {M} NED objects of type {tipo} listed in the area')
    if M == 0:
        retorno = []
        print(f'--- No objects of type {tipo} found in the search area')
    else:
        c = SkyCoord([SkyCoord.from_name(n) for n in lista_nombres[mask] if n])
        s = coord.separation(c)
        j = s.argmin()
        print(f'--- Nearest {tipo} object is {lista_nombres[mask][j]} separated {s[j].to(u.arcmin)} from the search coordinates')
        retorno = Ned.query_object(lista_nombres[mask][j])
        if tofile is not None:
            retorno.write(tofile)
    return retorno

def get_photometry_object(obj_name,from_year=2000):
    """
    Get photometry data from NED for a given object. The search is done using the astroquery package. 

    Parameters
    ----------
    obj_name : string
        Name of the object to search for
    from_year : float
        Minimum year of the photometry data to be returned

    Returns
    -------
    nu : astropy.units.Quantity
        Frequency of the photometry data
    S : astropy.units.Quantity
        Flux density of the photometry data
    ref : string
        Reference of the photometry data

    """

    
    try:
        tabla  = Ned.get_table(obj_name,output_table_format=3,from_year=from_year)
        if len(tabla)==0:
            nu,S,ref = [[],[],[]]
        else:
            unidad = u.Hz
            nu     = (unidad*tabla['Frequency'].data).to(u.GHz,equivalencies=u.spectral())
            S      = u.Jy*(tabla['NED Photometry Measurement'].data)
            ref    = tabla['Refcode']
    except (NameResolveError,TimeoutError,AstropyWarning):
        nu,S,ref = [[],[],[]]
    except Exception:
        nu,S,ref = [[],[],[]]

    return nu,S,ref

def get_possible_radio_z(coord,radio):
    """
    Get possible radio sources or QSOs in a given area in a NED search. The
    search is done using the astroquery package.

    Parameters
    ----------
    coord : SkyCoord
        Coordinates of the center of the search area
    radio : Quantity
        Radius of the search area

    Returns
    -------
    tb : astropy.table.Table
        Table with the results of the search

    """

    tipos    = ['QSO','RadioS','G']
    try:
        tabla    = Ned.query_region(coord,radius=radio)
        tipo     = np.array([x.decode() for x in tabla['Type'].data.data])
        redshift = tabla['Redshift'].data.data

        masca    = np.invert(np.isnan(redshift))
        masca2   = np.zeros(tipo.shape,dtype=bool)
        for i in range(masca2.size):
            if tipo[i] in tipos:
                masca2[i] = True
        masca *= masca2
        tb    = tabla[masca]
    except RemoteServiceError:
        tb = []
    except "Query failed":
        tb = []

    return tb


import numpy as np

def find_nearest(array, value):
    """
    Find the index of the element in the array that is closest to the given value.

    Parameters:
    array (numpy.ndarray): The array to search.
    value (float): The value to search for.

    Returns:
    int: The index of the element in the array that is closest to the given value.
         If the array is empty or the value is not a real number, returns -1.
    """
 
    idx = np.argmin(np.abs(array - value)) if array.size > 0 and np.isreal(value) else -1
    return idx