#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:39:16 2020

@author: herranz
"""

import astropy.units as u
import numpy         as np

### Define custom units:

Kcmb  = u.def_unit('K_CMB')
try:
    u.add_enabled_units([Kcmb])
except ValueError:
    pass

mKcmb = u.def_unit('mK_CMB',1.e-3*Kcmb)
try:
    u.add_enabled_units([mKcmb])
except ValueError:
    pass

uKcmb = u.def_unit('uK_CMB',1.e-6*Kcmb)
try:
    u.add_enabled_units([uKcmb])
except ValueError:
    pass


### Parse units from strings

def parse_unit(string):

    try:
        x = u.Unit(string)
    except ValueError:
        if string.lower().replace(' ','') == 'jy/sr':
            x = u.Jy/u.sr
        elif string.lower().replace(' ','') == 'mjy/sr':
            x = u.MJy/u.sr
        elif string.lower().replace(' ','') in ['(jy/sr)^2','(jy/sr)**2']:
            x = (u.Jy/u.sr)**2
        elif string.lower().replace(' ','') in ['(mjy/sr)^2','(mjy/sr)**2']:
            x = (u.MJy/u.sr)**2
        elif string.lower().replace(' ','') in ['k_cmb','k(cmb)','kcmb']:
            x = Kcmb
        elif string.lower().replace(' ','') in ['k_cmb^2','k_cmb**2','(k_cmb)^2','(k_cmb)**2']:
            x = Kcmb**2
        elif string.lower().replace(' ','') in ['mk_cmb','mk(cmb)','mkcmb']:
            x = mKcmb
        elif string.lower().replace(' ','') in ['mk_cmb^2','mk_cmb**2','(mk_cmb)^2','(mk_cmb)**2']:
            x = mKcmb**2
        elif string.lower().replace(' ','') in ['uk_cmb','uk(cmb)','ukcmb']:
            x = uKcmb
        elif string.lower().replace(' ','') in ['uk_cmb^2','uk_cmb**2','(uk_cmb)^2','(uk_cmb)**2']:
            x = uKcmb**2
        else:
            print('unrecognized unit string')

    return x

def convert_factor(unit1_in,unit2_in,nu=None,beam_area=None):

    if unit1_in.to_string() == mKcmb.to_string():
        unit1 = Kcmb
        addf  = 1.e-3
    elif unit1_in.to_string() == uKcmb.to_string():
        unit1 = Kcmb
        addf  = 1.e-6
    else:
        unit1 = unit1_in
        addf  = 1.0

    if unit2_in.to_string() == mKcmb.to_string():
        unit2 = Kcmb
        addf  = addf*1.e3
    elif unit2_in.to_string() == uKcmb.to_string():
        unit2 = Kcmb
        addf  = addf*1.e6
    else:
        unit2 = unit2_in
        addf  = addf*1.0

    if unit1.to_string() == unit2.to_string():
        equiv = None
        fact  = 1
    elif unit1.is_equivalent(unit2):
        print(unit1,unit2)
        equiv = None
        fact  = unit1.to(unit2)

    # Block for beam areas:

    elif (unit1.is_equivalent(u.Jy) and unit2.is_equivalent(u.Jy/u.sr)):
        equiv = u.beam_angular_area(beam_area)
        fact  = (unit1/u.beam).to(unit2,equivalencies=equiv)
    elif (unit2.is_equivalent(u.Jy) and unit1.is_equivalent(u.Jy/u.sr)):
        equiv = u.beam_angular_area(beam_area)
        fact  = unit1.to(unit2/u.beam,equivalencies=equiv)

    # Block for thermodynamic temperature conversions

    elif (unit1.is_equivalent(Kcmb) and unit2.is_equivalent(u.Jy/u.sr)):
        equiv = u.thermodynamic_temperature(nu)
        fact  = (1)*u.K.to(unit2,equivalencies=equiv)
    elif (unit2.is_equivalent(Kcmb) and unit1.is_equivalent(u.Jy/u.sr)):
        equiv = u.thermodynamic_temperature(nu)
        fact  = unit1.to(u.K,equivalencies=equiv)*(Kcmb.to(unit2))
    elif (unit1.is_equivalent(Kcmb) and unit2.is_equivalent(u.Jy/u.beam)):
        equiv = u.thermodynamic_temperature(nu)
        fact  = (1)*(u.K).to(u.Jy/u.sr,equivalencies=equiv)
        fact2 = (u.Jy/u.sr).to(u.Jy/u.beam,equivalencies=u.beam_angular_area(beam_area))
        fact  = fact*fact2
    elif (unit2.is_equivalent(Kcmb) and unit1.is_equivalent(u.Jy/u.beam)):
        equiv = u.thermodynamic_temperature(nu)
        fact  = unit1.to(u.Jy/u.sr,equivalencies=u.beam_angular_area(beam_area))
        fact2 = (u.Jy/u.sr).to(u.K,equivalencies=equiv)
        fact  = fact*fact2*(1)
    elif (unit1.is_equivalent(Kcmb) and unit2.is_equivalent(u.Jy)):
        equiv = u.thermodynamic_temperature(nu)

        fact  = (1)*(u.K).to(u.Jy/u.sr,equivalencies=equiv)
        fact2 = (u.Jy/u.sr).to(u.Jy/u.beam,equivalencies=u.beam_angular_area(beam_area))
        fact  = fact*fact2*(u.Jy.to(unit2))
    elif (unit2.is_equivalent(Kcmb) and unit1.is_equivalent(u.Jy)):
        equiv = u.thermodynamic_temperature(nu)
        try:
            fact0 = unit1.to(u.Jy).value
        except AttributeError:
            fact0 = unit1.to(u.Jy)
        unitb = u.Jy/u.beam
        fact  = unitb.to(u.Jy/u.sr,equivalencies=u.beam_angular_area(beam_area))
        fact2 = (u.Jy/u.sr).to(u.K,equivalencies=equiv)
        fact  = fact0*fact*fact2*(1)

    # Block for squared thermodynamic temperature conversions

    elif (unit1.is_equivalent(Kcmb**2) and unit2.is_equivalent((u.Jy/u.sr)**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.thermodynamic_temperature(nu)
        fact   = (1)*u.K.to(unit2b,equivalencies=equiv)
        fact   = fact**2
    elif (unit2.is_equivalent(Kcmb**2) and unit1.is_equivalent((u.Jy/u.sr)**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.thermodynamic_temperature(nu)
        fact   = unit1b.to(u.K,equivalencies=equiv)*(Kcmb.to(unit2b))
        fact   = fact**2
    elif (unit1.is_equivalent(Kcmb**2) and unit2.is_equivalent((u.Jy/u.beam)**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.thermodynamic_temperature(nu)
        fact   = (1)*(u.K).to(u.Jy/u.sr,equivalencies=equiv)
        fact2  = (u.Jy/u.sr).to(u.Jy/u.beam,equivalencies=u.beam_angular_area(beam_area))
        fact   = fact*fact2
        fact   = fact**2
    elif (unit2.is_equivalent(Kcmb**2) and unit1.is_equivalent((u.Jy/u.beam)**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.thermodynamic_temperature(nu)
        fact   = unit1b.to(u.Jy/u.sr,equivalencies=u.beam_angular_area(beam_area))
        fact2  = (u.Jy/u.sr).to(u.K,equivalencies=equiv)
        fact   = fact*fact2*(1)
        fact   = fact**2
    elif (unit1.is_equivalent(Kcmb**2) and unit2.is_equivalent(u.Jy**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.thermodynamic_temperature(nu)
        fact   = (1)*(u.K).to(u.Jy/u.sr,equivalencies=equiv)
        fact2  = (u.Jy/u.sr).to(u.Jy/u.beam,equivalencies=u.beam_angular_area(beam_area))
        fact   = fact*fact2*(u.Jy.to(unit2b))
        fact   = fact**2
    elif (unit2.is_equivalent(Kcmb**2) and unit1.is_equivalent(u.Jy**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.thermodynamic_temperature(nu)
        try:
            fact0 = unit1b.to(u.Jy).value
        except AttributeError:
            fact0 = unit1b.to(u.Jy)
        unitb = u.Jy/u.beam
        fact  = unitb.to(u.Jy/u.sr,equivalencies=u.beam_angular_area(beam_area))
        fact2 = (u.Jy/u.sr).to(u.K,equivalencies=equiv)
        fact  = fact0*fact*fact2*(1)
        fact  = fact**2

    # Block for brightness (Rayleigh-Jeans) temperature conversions:

    elif (unit1.is_equivalent(u.K**2) and unit2.is_equivalent((u.Jy/u.sr))**2):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.brightness_temperature(nu,beam_area=beam_area)
        fact   = unit1b.to(u.Jy/u.beam,equivalencies=equiv)
        fact   = fact*((u.Jy/u.beam).to(unit2b,equivalencies=u.beam_angular_area(beam_area)))
        fact   = fact**2
    elif (unit2.is_equivalent(u.K**2) and unit1.is_equivalent((u.Jy/u.sr)**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.brightness_temperature(nu,beam_area=beam_area)
        fact   = unit1b.to(u.Jy/u.beam,equivalencies=u.beam_angular_area(beam_area))
        fact   = fact*((u.Jy/u.beam).to(unit2b,equivalencies=equiv))
        fact   = fact**2
    elif (unit1.is_equivalent(u.K**2) and unit2.is_equivalent((u.Jy/u.beam)**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.brightness_temperature(nu,beam_area=beam_area)
        fact   = unit1b.to(unit2b,equivalencies=equiv)
        fact   = fact**2
    elif (unit2.is_equivalent(u.K**2) and unit1.is_equivalent((u.Jy/u.beam)**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.brightness_temperature(nu,beam_area=beam_area)
        fact   = unit1b.to(unit2b,equivalencies=equiv)
        fact   = fact**2
    elif (unit1.is_equivalent(u.K**2) and unit2.is_equivalent(u.Jy**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.brightness_temperature(nu,beam_area=beam_area)
        fact   = unit1b.to(unit2b,equivalencies=equiv)
        fact   = fact**2
    elif (unit2.is_equivalent(u.K**2) and unit1.is_equivalent(u.Jy**2)):
        unit1b = np.power(unit1,1/2)
        unit2b = np.power(unit2,1/2)
        equiv  = u.brightness_temperature(nu,beam_area=beam_area)
        fact   = unit1b.to(unit2b,equivalencies=equiv)
        fact   = fact**2

    # Block for squared brightness (Rayleigh-Jeans) temperature conversions:

    elif (unit1.is_equivalent(u.K) and unit2.is_equivalent(u.Jy/u.sr)):
        equiv = u.brightness_temperature(nu,beam_area=beam_area)
        fact  = unit1.to(u.Jy/u.beam,equivalencies=equiv)
        fact  = fact*((u.Jy/u.beam).to(unit2,equivalencies=u.beam_angular_area(beam_area)))
    elif (unit2.is_equivalent(u.K) and unit1.is_equivalent(u.Jy/u.sr)):
        equiv = u.brightness_temperature(nu,beam_area=beam_area)
        fact  = unit1.to(u.Jy/u.beam,equivalencies=u.beam_angular_area(beam_area))
        fact  = fact*((u.Jy/u.beam).to(unit2,equivalencies=equiv))
    elif (unit1.is_equivalent(u.K) and unit2.is_equivalent(u.Jy/u.beam)):
        equiv = u.brightness_temperature(nu,beam_area=beam_area)
        fact  = unit1.to(unit2,equivalencies=equiv)
    elif (unit2.is_equivalent(u.K) and unit1.is_equivalent(u.Jy/u.beam)):
        equiv = u.brightness_temperature(nu,beam_area=beam_area)
        fact  = unit1.to(unit2,equivalencies=equiv)
    elif (unit1.is_equivalent(u.K) and unit2.is_equivalent(u.Jy)):
        equiv = u.brightness_temperature(nu,beam_area=beam_area)
        fact  = unit1.to(unit2,equivalencies=equiv)
    elif (unit2.is_equivalent(u.K) and unit1.is_equivalent(u.Jy)):
        equiv = u.brightness_temperature(nu,beam_area=beam_area)
        fact  = unit1.to(unit2,equivalencies=equiv)


    # Else

    else:
        fact  = 1
        print(' --- Warning: unknown unit conversion from {0} to {1}'.format(unit1.to_string(),unit2.to_string()))

    fact = addf*fact

    return fact


### Test


def testing():
    
    from astropy.table import Table
    from missions import Planck as exper
    rows = []
    
    for i in range(exper.nfreq):
        nu        = exper.freq[i]
        beam_area = exper.beam_area[i]
        rows.append({'Freq':nu.value,
                     'Beam area':beam_area.value,
                     'K_cmb to MJy/sr':convert_factor(Kcmb,u.MJy/u.sr,nu=nu,beam_area=beam_area),
                     'MJy/sr to K_cmb':convert_factor(u.MJy/u.sr,Kcmb,nu=nu,beam_area=beam_area),
                     'direct*inverse (K_cmb to MJy/sr)':convert_factor(Kcmb,u.MJy/u.sr,nu=nu,beam_area=beam_area)*
                                                        convert_factor(u.MJy/u.sr,Kcmb,nu=nu,beam_area=beam_area),
                     'K_cmb to Jy/beam':convert_factor(Kcmb,u.Jy/u.beam,nu=nu,beam_area=beam_area),
                     'Jy/beam to K_cmb':convert_factor(u.Jy/u.beam,Kcmb,nu=nu,beam_area=beam_area),
                     'direct*inverse (K_cmb to Jy/beam)':convert_factor(Kcmb,u.Jy/u.beam,nu=nu,beam_area=beam_area)*
                                                         convert_factor(u.Jy/u.beam,Kcmb,nu=nu,beam_area=beam_area),
                     'K to MJy/sr':convert_factor(u.K,u.MJy/u.sr,nu=nu,beam_area=beam_area),
                     'MJy/sr to K':convert_factor(u.MJy/u.sr,u.K,nu=nu,beam_area=beam_area),
                     'direct*inverse (K to MJy/sr)':convert_factor(u.K,u.MJy/u.sr,nu=nu,beam_area=beam_area)*
                                                    convert_factor(u.MJy/u.sr,u.K,nu=nu,beam_area=beam_area),
                     'K to Jy/beam':convert_factor(u.K,u.Jy/u.beam,nu=nu,beam_area=beam_area),
                     'Jy/beam to K':convert_factor(u.Jy/u.beam,u.K,nu=nu,beam_area=beam_area),
                     'direct*inverse (K to Jy/beam)':convert_factor(u.K,u.Jy/u.beam,nu=nu,beam_area=beam_area)*
                                                     convert_factor(u.Jy/u.beam,u.K,nu=nu,beam_area=beam_area),
                     'K to Jy':convert_factor(u.K,u.Jy,nu=nu,beam_area=beam_area),
                     'Jy to K':convert_factor(u.Jy,u.K,nu=nu,beam_area=beam_area),
                     'direct*inverse (K to Jy)':convert_factor(u.K,u.Jy,nu=nu,beam_area=beam_area)*
                                                convert_factor(u.Jy,u.K,nu=nu,beam_area=beam_area)})
    
    t = Table(rows)
    t['coc thermo/brightness']        = t['K_cmb to Jy/beam']/t['K to Jy/beam']
    t['coc brightness Jy/(Jy/beam))'] = t['K to Jy']/t['K to Jy/beam']
    
    t.write('/Users/herranz/Desktop/test_units.fits',overwrite=True)
    



