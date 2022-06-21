# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 21:16:00 2016

   CMB mission data (WMAP, Planck)

@author: herranz
"""

import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import os
import pandas as pd
import matplotlib.pyplot as plt
from myutils import fwhm2sigma,sigma2fwhm
from astropy.table import Table
import healpy as hp
import ast

np.seterr(divide='ignore', invalid='ignore')

# %% Mission class

class Mission:

    def __init__(self,nfreq,bands,fwhm,extra_info):
        self.nfreq      = nfreq
        self.bands      = bands
        self.fwhm       = fwhm
        self.extra_info = extra_info

    def __getitem__(self,sliced):
        a = self.bands[sliced]
        b = self.fwhm[sliced]
        n = np.size(a)
        return Mission(n,a,b,self.extra_info)

    def append(self,newmission):
        n = self.nfreq
        m = newmission.nfreq
        b = []
        f = []
        for i in range(n):
            b.append(self.bands[i].to(u.GHz,equivalencies=u.spectral()).value)
            f.append(self.fwhm[i].to('arcmin').value)
        for i in range(m):
            b.append(newmission.bands[i].to(u.GHz,equivalencies=u.spectral()).value)
            f.append(newmission.fwhm[i].to('arcmin').value)
        if self.extra_info is None:
            if newmission.extra_info is None:
                lainfo = None
            else:
                lainfo = newmission.extra_info
        else:
            if newmission.extra_info is None:
                lainfo = self.extra_info
            else:
                lainfo = self.extra_info+newmission.extra_info
        return Mission(n+m,np.array(b)*u.GHz,np.array(f)*u.arcmin,lainfo)


    def fwhm_pixels(self,nside):
        p = hp.nside2resol(nside)*u.rad
        return (self.fwhm/p).si.value

    def sigma_pixels(self,nside):
        p = hp.nside2resol(nside)*u.rad
        return (self.sigma/p).si.value

    def index_frequency(self,freq,tol=1.e-6):
        if np.isscalar(freq):
            r = (freq-self.freq_GHz)/freq
        else:
            r = ((freq-self.freq)/freq).si.value
        return np.where(abs(r)<tol)[0][0]

    @property
    def freq(self):
        return self.bands.to(u.GHz,equivalencies=u.spectral())

    @property
    def wavelength(self):
        return self.bands.to(u.meter,equivalencies=u.spectral())

    @property
    def fwhm_arcmin(self):
        return self.fwhm.to('arcmin').value

    @property
    def sigma(self):
        return self.fwhm*fwhm2sigma

    @property
    def freq_GHz(self):
        return self.bands.to(u.GHz,equivalencies=u.spectral()).value

    @property
    def wavelenght_microns(self):
        return self.bands.to(u.micrometer,equivalencies=u.spectral()).value

    @property
    def wavelenght_angstrom(self):
        return self.bands.to(u.angstrom,equivalencies=u.spectral()).value

    @property
    def beam_area(self):
        beam_sigma = self.sigma
        omega_B    = (2*np.pi*beam_sigma**2).si
        return omega_B

    @property
    def Jy2K(self):
        omega_B = self.beam_area
        freq = self.freq
        return u.Jy.to(u.K, equivalencies=u.brightness_temperature(omega_B, freq))

    @property
    def K2Jy(self):
        omega_B = self.beam_area
        freq = self.freq
        return u.K.to(u.Jy, equivalencies=u.brightness_temperature(omega_B, freq))



class Band_filter:

    def __init__(self,name,x_axis,transmission,uncertainty,flag):
        self.name         = name
        self.x_axis       = x_axis
        self.transmission = transmission
        self.uncertainty  = uncertainty
        self.flag         = flag

    def plot_wavelenght(self,color=None,newplot=False):
        if newplot:
            plt.figure()
        if color is not None:
            plt.plot(self.wavelength.value,self.transmission,color=color)
        else:
            plt.plot(self.wavelength.value,self.transmission)
        if newplot:
            plt.xlabel(r'$\lambda$ [Angstrom]')
            plt.ylabel('transmission')

    def plot_frequency(self,color=None,newplot=False):
        if newplot:
            plt.figure()
        if color is not None:
            plt.plot(self.frequency.value,self.transmission,color=color)
        else:
            plt.plot(self.frequency.value,self.transmission)
        if newplot:
            plt.xlabel(r'$\nu$ [GHz]')
            plt.ylabel('transmission')

    def average_K2Jy(self,beam_area):
        x = self.x_axis
        y = self.transmission
        z = u.K.to(u.Jy,equivalencies=u.brightness_temperature(beam_area,x))
        return np.sum(z*y)/np.sum(y)

    @property
    def wavelength(self):
        return self.x_axis.to(u.angstrom,equivalencies=u.spectral())

    @property
    def frequency(self):
        return self.x_axis.to(u.GHz,equivalencies=u.spectral())

    @property
    def central_wavelength(self):
        return np.sum(self.wavelength*
                      self.transmission)/np.sum(self.transmission)

    @property
    def central_frequency(self):
        return np.sum(self.frequency*
                      self.transmission)/np.sum(self.transmission)




# %%                           PLANCK


planck_RIMO_froot = '/Users/herranz/Dropbox/Trabajo/Planck/Data/RIMO_files/'
planck_RIMO_files = [planck_RIMO_froot+'LFI_RIMO_R3.31.fits',
                     planck_RIMO_froot+'HFI_RIMO_R3.00.fits']

def Planck_bandpass(ifreq):

    sufix = ['030','044','070',
             'F100','F143','F217',
             'F353','F545','F857']
    if ifreq<3:
        fname  = planck_RIMO_files[0]
        unidad = u.GHz
    else:
        fname  = planck_RIMO_files[1]
        unidad = 1.0/u.cm
    extname = 'BANDPASS_'+sufix[ifreq]

    hdulist = fits.open(fname)
    n = hdulist.index_of(extname)
    t = hdulist[n]

    waven = t.data['WAVENUMBER']*unidad
    trans = t.data['TRANSMISSION']
    if ifreq<3:
        ebar  = t.data['UNCERTAINITY']
    else:
        ebar  = np.zeros(trans.size)
    flag  = t.data['FLAG']

    if ifreq>=3:
        l = 1.0/waven
        del(waven)
        waven = l.to(u.GHz,equivalencies=u.spectral())

    hdulist.close()
    return Band_filter(sufix[ifreq],waven,trans,ebar,flag)

planck_freqs  = np.array([30,44,70,100,143,217,353,545,857])*u.GHz
planck_fwhm   = np.array([32.65,27.00,13.01,9.94,7.04,4.66,4.41,4.47,4.23])*u.arcmin

planck_from_RIMOS = True
if planck_from_RIMOS:

    hdulist    = fits.open(planck_RIMO_files[0])
    n          = hdulist.index_of('FREQUENCY_MAP_PARAMETERS')
    map_params = hdulist[n]
    for i in range(3):
        planck_fwhm[i] = map_params.data['FWHM_EFF'][i]*u.arcmin
    hdulist.close()

    hdulist    = fits.open(planck_RIMO_files[1])
    n          = hdulist.index_of('MAP_PARAMS')
    map_params = hdulist[n]
    for i in range(6):
        planck_fwhm[i+3] = map_params.data['FWHM'][i]*u.arcmin
    hdulist.close()



planck_bands  = [Planck_bandpass(i) for i in range(9)]
Planck        = Mission(9,planck_freqs,planck_fwhm,{'bandpasses':planck_bands})

LFI           = Planck[0:3]
HFI           = Planck[3:]

planck_freqs2 = np.array([28.4,44.1,70.4,100,143,217,353,545,857])*u.GHz
planck_fwhm2  = np.array([32.293,27.00,13.213,9.94,7.04,4.66,4.41,4.47,4.23])*u.arcmin
Planck_RIMO   = Mission(9,planck_freqs2,planck_fwhm2,{'bandpasses':planck_bands})

LFI_RIMO      = Planck_RIMO[0:3]
HFI_RIMO      = Planck_RIMO[3:]


# %%                           WMAP

wmap_freqs    = np.array([23,33,41,61,94])*u.GHz
wmap_fwhm     = np.array([0.88,0.66,0.51,0.35,0.22])*u.degree
band_names    = 'band names: K,Ka,Q,V,W'
WMAP          = Mission(5,wmap_freqs,wmap_fwhm,band_names)


# %%                           QUIJOTE


quijote_freqs = np.array([11,13,17,19,30,40])*u.GHz
quijote_fwhm  = np.array([0.92,0.92,0.60,0.60,0.37,0.28])*u.degree
QUIJOTE       = Mission(6,quijote_freqs,quijote_fwhm,None)

MFI           = QUIJOTE[0:4]
TGI           = QUIJOTE[4]
FGI           = QUIJOTE[5]


# %%                        HERSCHEL


SPIRE_bands   = np.array([250,350,500])*u.micrometer
SPIRE_fwhm    = np.array([18.1,25.2,36.6])*u.arcsec
SPIRE_info    = 'H-ATLAS pixel size = 6, 8 and 12 arcsec at 250, 350 and 500 micron respectively. Standard pipeline SPIRE pixel sizes are 6, 10 and 14 arcsec for the same bands'
SPIRE         = Mission(3,SPIRE_bands,SPIRE_fwhm,SPIRE_info)

PACS_bands    = np.array([70,100,160])*u.micrometer
PACS_fwhm     = np.array([5.2,	7.7,	12])*u.arcsec
PACS          = Mission(3,PACS_bands,PACS_fwhm,None)

Herschel      = PACS.append(SPIRE)


# %%                           J-PAS

JPAS_transmission_dir   = '/Users/herranz/Dropbox/Trabajo/JPAS/Data/JPAS_Transmission_Curves_20170316/'

JPAS_transmission_names = [file for file
                           in os.listdir(JPAS_transmission_dir)
                           if file.endswith('.tab')]

JPAS_filter_names       = [nombre.replace('JPAS_','').replace('.tab','') for
                           nombre in JPAS_transmission_names]

def get_JPAS_filter(i):
    nombre = JPAS_transmission_dir+JPAS_transmission_names[i]
    filtro = pd.read_table(nombre,comment='#',
                           header=None,
                           delim_whitespace=True,
                           names=['Lambda[A]','T'])
    w = np.array([x for x in filtro['Lambda[A]']])*u.angstrom
    t = np.array([x for x in filtro['T']])
    ebar = np.zeros(t.size)
    flag = np.array(['F' for i in range(t.size)])
    return Band_filter(JPAS_filter_names[i],w,t,ebar,flag)



# %%                          PICO


PICO_bands = np.array([21,25,30,36,43.2,51.8,62.2,74.6,89.6,107.5,
                       129,154.8,185.8,222.9,267.5,321,385.2,462.2,
                       554.7,665.6,798.7])*u.GHz
PICO_fwhm  = np.array([38.4,32.0,28.3,23.6,22.2,18.4,12.8,10.7,9.5,7.9,7.4,
                       6.2,4.3,3.6,3.2,2.6,2.5,2.1,1.5,1.3,1.1])*u.arcmin
PICO_noise_rms = np.array([16.3,11.7,7.8,5.6,5.4,4.0,3.9,3.2,2.0,
                           1.7,1.6,1.4,2.5,3.1,2.0,3.0,3.3,7.8,
                           44.1,176.9,1260.7])*(10**(-6))*u.K*u.arcmin
#PICO_noise_rms = np.array([35.3553,23.3345,15.8392,10.6066,6.43467,
#                           4.94975,3.53553,2.82843,2.26274,2.05061,1.90919,
#                           1.83848,2.54558,3.74767,6.36396,11.3137,22.6274,
#                           53.0330,155.563,777.817,7071.07])*10**(-6)*u.K*u.arcmin

PICO_sims_location = 'http://www.jb.man.ac.uk/~cdickins/exchange/bpol_sims/'
PICO_sims_location += 'Mathieu/CMB-Probe/CMBprobe_intensity_sim/'
PICO_sims_location += 'skyinbands/CMB_PROBE_2017/'

PICO = Mission(21,PICO_bands,PICO_fwhm,
               {'noise_rms':PICO_noise_rms,'sims location':PICO_sims_location})



# %%                           LiteBird

IMo_file    = '/Users/herranz/Dropbox/Trabajo/LiteBird/IMo/IMo_V1_July2020.txt'
IMo         = Table.read(IMo_file,format='ascii',delimiter=';')

LiteBird    = Mission(len(IMo),
                      np.array(IMo['Center Frequency [GHz]'])*u.GHz,
                      np.array([ast.literal_eval(y)*u.arcmin for y in IMo['Beam size [arcmin]']],dtype='object'),
                      {'telescope':np.array(IMo['Telescope']),
                       'band ID'  :np.array(IMo['Band ID']),
                       'bandwidth':np.array([float(x.split(' ')[0])
                                             for x in IMo['Frequency Band [GHz] (Frac.)']])*u.GHz,
                       'pol rms'  :np.array(IMo['Sensitivity [uK arcmin]'])*1.e-6*u.K*u.arcmin})



# LFT_bands   = [40  ,50  ,60  ,68  ,78  ,89,  100, 119, 140]
# LFT_fwhm    = [69.2,56.9,49.0,40.8,36.1,32.3,27.7,23.7,20.7]
# LFT_pol_rms = [36.1,19.6,20.2,11.3,10.3,8.4 ,7.0 ,5.8, 4.7 ]


# HFT_bands   = [100, 119, 140, 166, 195, 235, 280, 337, 402]
# HFT_fwhm    = [37.0,31.6,27.6,24.2,21.7,19.6,13.2,11.2,9.7]
# HFT_pol_rms = [7.0 ,5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4,19.6]

# LiteBird    = Mission(18,np.array(LFT_bands+HFT_bands)*u.GHz,
#                       np.array(LFT_fwhm+HFT_fwhm)*u.arcmin,
#                       {'pol_rms':np.array(LFT_pol_rms+HFT_pol_rms)*(10**(-6))*u.K*u.arcmin})