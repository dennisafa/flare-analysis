from appaloosa import aflare as ap
from lightkurve import KeplerTargetPixelFile, KeplerLightCurveFile
import matplotlib.pyplot as pl
import numpy as np
import wolf359.flaredetect as fd
import scipy as scipy
from scipy.optimize import minimize
from numpy import asarray
import george
from george import kernels
from scipy.signal import savgol_filter as sf
import celerite as cl
from celerite import terms
import copy
from astropy.io import fits
from scipy import integrate
import pandas as pd
from astropy.stats import LombScargle
import wolf359.wolfrotation as master


def computegeorge (flux, time):
    global gp
    global y

    y = flux

    kernel = kernels.CosineKernel(log_period=np.log(17), axes=0) * kernels.ExpSquaredKernel(metric=0.5)
    gp = george.GP(kernel)
    gp.compute(time, flux)
    print('Initial log likelihood', gp.log_likelihood(y))
    print('initial parameter vector', gp.get_parameter_vector())
    # res = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    # gp.set_parameter_vector(res.x)
    # print('Final log likelihood', gp.log_likelihood(y))
    # print('final parameter vector', res.x)
    # print(res)

    pred_mean, pred_var = gp.predict(flux, time, return_var=True)

    pl.fill_between(time, pred_mean - np.sqrt(pred_var), pred_mean + np.sqrt(pred_var), color='k', alpha=0.4,
                    label='Predicted variance')
    # pl.fill_between(time, flux - np.sqrt(gp._yerr2), flux + np.sqrt(gp._yerr2), color='k')
    pl.plot(time, pred_mean, color='Blue', label='Predicted mean')
    pl.plot(time, flux, alpha=0.6, label='Raw flux')
    pl.ylim(-0.1, 0.8)
    pl.legend(loc='best')
    pl.xlabel('Time')
    pl.ylabel('Normalized Flux')
    pl.show()
    # print(result)
    return pred_mean



def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y)

def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y)


#2457949.4982

'''Rotation modeling'''
file = np.genfromtxt("211828663.txt", dtype=float, usecols=(0, 1), delimiter=',')
y = file[:, 1]
x = file[:, 0]
print("Creating model...")
flare = master.Flare(y, x, 0, len(y))
y = master.clean(y, x)
g = computegeorge(y, x)

'''End rotation modeling'''



'''Flare analysis'''
# tpf = KeplerTargetPixelFile.from_archive('211828663', cadence='short')
# tpf.plot(frame=0)
# print(tpf.pipeline_mask)
# pl.show()
# pl.clf()
# aper = np.zeros(tpf.shape[1:])
# aper[2:6, 3:7] = 1
# tpf.plot(aperture_mask=aper)
# pl.show()
# lc = tpf.to_lightcurve(aperture_mask=aper.astype(bool))
# lc = lc.remove_nans().remove_outliers()
# flux = lc.flux
# time = lc.time
'''End flare analysis'''

