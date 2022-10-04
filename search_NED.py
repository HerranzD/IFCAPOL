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

    tabla = Ned.query_region(coord,radius=radio)
    N = len(tabla)
    print(' ')
    print(' --- There are {0} NED objects listed in the area '.format(N))
    x = tabla[tabla['Type']==tipo]
    print(' --- There are {0} NED objects of type {1} listed in the area '.format(len(x),tipo))
    x.sort('Separation')

    return x


def find_nearest_object_type(coord,radio,tipo,tofile=None):
    tabla = Ned.query_region(coord,radius=radio)
    N = len(tabla)
    print(' ')
    print(' --- There are {0} NED objects listed in the area '.format(N))
    lista_nombres = np.array([tabla['Object Name'][i] for i in range(N)])
    lista_tipos   = np.array([tabla['Type'][i] for i in range(N)])
    mask          = np.array(lista_tipos) == tipo
    M = np.count_nonzero(mask)
    print(' --- There are {0} NED objects of type {1} listed in the area '.format(M,tipo))
    if M == 0:
        retorno = []
    else:
        nombres = lista_nombres[mask]
        c = []
        for i in range(M):
            try:
                v = SkyCoord.from_name(nombres[i])
                c.append(v)
            except NameResolveError:
                pass
        c = SkyCoord(c)

        s = coord.separation(c)
        j = s.argmin()
        print(' --- Nearest {0} object is {1} separated {2} from the search coordinates'.format(tipo,nombres[j],s[j].to(u.arcmin)))
        retorno = Ned.query_object(nombres[j])
        if tofile is not None:
            retorno.write(tofile)
    return retorno

def get_photometry_object(obj_name,from_year=2000):
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







def find_nearest(array,value):
    if (len(array) == 0) or not np.isreal(value):
        idx = -1
    else:
        idx = (np.abs(array-value*np.ones(array.size))).argmin()
    return idx