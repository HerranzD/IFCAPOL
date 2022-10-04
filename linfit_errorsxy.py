# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 12:33:51 2016

   Linear fit with errors in both x and y axis

@author: herranz
"""

import scipy.odr as odr
import numpy as np
import matplotlib.pyplot as plt

# Define a function (quadratic in our case) to fit the data with.
def linear_func(p, x):
    try:
        m, c = p
    except ValueError:
        m = p[0]
        c = p[1]
    except TypeError:
        m = p[0]
        c = p[1]
    return m*x + c

def linear0_func(p, x):
    m = p
    return m*x

# Create a model for fitting.
linmodel  = odr.Model(linear_func)
linmodel0 = odr.Model(linear0_func)

def linfit_errxy(x,y,x_err,y_err,verbose=False):

# Create a RealData object using our initiated data from above.
    data = odr.RealData(x, y, sx=x_err, sy=y_err)

# Set up ODR with the models and data

    odr1 = odr.ODR(data, linmodel, beta0=[1., 0.])
    odr2 = odr.ODR(data, linmodel0, beta0=[1.0])

    out1 = odr1.run()
    out2 = odr2.run()

    if verbose:
        out1.pprint()
        print(' ')
        out2.pprint()

    return out1,out2

def plotfit(x,y,x_err,y_err,outmodel,
            addunit   = False,
            addfit    = True,
            logscal   = False,
            x_label   = 'x',
            y_label   = 'y',
            newfig    = False,
            subplotn  = 111,
            linewidth = 0.5,
            capsize   = 2):

    if newfig:
        plt.figure()

    ax = plt.subplot(subplotn)

    if logscal:
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')

    plt.errorbar(x,y,xerr=x_err,yerr=y_err,
                 fmt='o',
                 linewidth=linewidth,
                 capsize=capsize)

    if x.min()>=0.0:
        xmi = 0.9*x.min()
    else:
        xmi = 1.1*x.min()

    if x.max()>=0.0:
        xma = 1.1*x.max()
    else:
        xmi = 0.9*x.max()

    x0 = np.linspace(xmi,xma,1000)

    if outmodel.beta.size == 2:
        m = outmodel.beta[0]
        c = outmodel.beta[1]
        y0 = m*x0+c
    else:
        m = outmodel.beta[0]
        y0 = m*x0

    if addfit:
        plt.plot(x0,y0,'r')

    if addunit:
        plt.plot(x0,x0,':k')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return x0,y0


