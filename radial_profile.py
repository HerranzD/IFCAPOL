import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import interp1d

def radial_profile(data,nbins=50,center=None,
                   toplot=False,kind='linear',datype='real',
                   equal_scale=False):

    s       = data.shape
    dmean   = data.mean()
    dstd    = data.std()
    size    = s[0]
    perfil  = [None] * nbins
    xperfil =  [None] * nbins

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    r = np.sqrt((x-x0)**2+(y-y0)**2)

    rbin = (nbins * r/r.max()).astype(np.int)

    if datype=='real':
        perfil  = ndimage.mean(data, labels=rbin, index=np.arange(0, rbin.max() ))
    else:
        perfilr = ndimage.mean(np.real(data), labels=rbin, index=np.arange(0, rbin.max() ))
        perfili = ndimage.mean(np.imag(data), labels=rbin, index=np.arange(0, rbin.max() ))
        perfil  = perfilr+np.complex(0,1)*perfili

    xperfil = [i for i in range(nbins)]
    xperfil = np.multiply(xperfil,r.max())
    xperfil = np.divide(xperfil,float(nbins))+r.max()/(2.0*nbins)

    perfil  = [data[x0,y0]] + np.array(perfil).tolist()
    xperfil = [0] + np.array(xperfil).tolist()

    if datype=='real':
        f = interp1d(xperfil, perfil, kind=kind,bounds_error=False,fill_value=perfil[nbins-1])
        if kind=='linear':
            pmap = np.interp(r,xperfil,perfil)
        else:
            pmap = f(r)
    else:
        f1 = interp1d(xperfil, np.real(perfil), kind=kind,bounds_error=False,fill_value=np.real(perfil[nbins-1]))
        f2 = interp1d(xperfil, np.imag(perfil), kind=kind,bounds_error=False,fill_value=np.imag(perfil[nbins-1]))
        if kind=='linear':
            pmap = np.interp(r,xperfil,np.real(perfil))+np.complex(0,1)*np.interp(r,xperfil,np.imag(perfil))
        else:
            pmap = f1(r) + np.complex(0,1)*f2(r)

    if toplot:
        plt.figure()
        xnew = np.linspace(0,np.max(xperfil),1001,endpoint=True)
        if datype=='real':
            plt.plot(xnew,f(xnew))
        else:
            plt.plot(xnew,f1(xnew))
        plt.show()
        plt.figure()
        plt.pcolormesh(pmap)
        plt.axis('tight')
        plt.colorbar()
        plt.show()

    if equal_scale:
        pmap = dstd*pmap/pmap.std()
        pmap = pmap-pmap.mean()+dmean

    return pmap,xperfil,perfil



