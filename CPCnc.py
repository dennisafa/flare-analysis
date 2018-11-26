from lightkurve import KeplerTargetPixelFile
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import wolf359.wolfrotation as master
import george
from george import kernels
from scipy.optimize import minimize


import warnings
warnings.filterwarnings("ignore")

import numpy as np
from matplotlib import pyplot as plt
from gatspy.periodic import LombScargleFast


from sklearn import linear_model
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, GridSearchCV

from wolf359.aflare import aflare1
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import simps

tpf = KeplerTargetPixelFile.from_archive('211817361', campaign=16)
tpf.plot(frame=6)
pl.show()
pl.clf()
aper = np.zeros(tpf.shape[1:])
aper[1:6, 2:7] = 1
lc = tpf.to_lightcurve(aperture_mask=aper.astype(bool))
lc = lc.remove_nans().remove_outliers().normalize().flatten()
tpf.plot(aperture_mask=aper)
pl.show()
y = lc.flux
x = lc.time
pl.plot(x, y)
pl.show()


def computegeorge (flux, time):
    global gp
    global y

    y = flux

    kernel = kernels.CosineKernel(log_period=np.log(20), axes=0) * kernels.ExpSquaredKernel(metric=0.5)
    gp = george.GP(kernel)
    gp.compute(time, flux)
    print('Initial log likelihood', gp.log_likelihood(y))
    print('initial parameter vector', gp.get_parameter_vector())
    res = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    gp.set_parameter_vector(res.x)
    print('Final log likelihood', gp.log_likelihood(y))
    print('final parameter vector', res.x)
    print(res)

    pred_mean, pred_var = gp.predict(flux, time, return_var=True)

    # pl.fill_between(time, pred_mean - np.sqrt(pred_var), pred_mean + np.sqrt(pred_var), color='k', alpha=0.4,
    #                 label='Predicted variance')
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

# '''Rotation modeling'''
flare = master.Flare(y, x, 0, len(y))
#y = master.removeRotation(flare.flux, flare.time)
pl.plot(x, y)
pl.show()
# g = computegeorge(flare.flux, flare.time)



'''End rotation modeling'''



'''Flare analysis'''

# flare = master.flatten(y, x)
# pl.plot(flare.time, flare.flux)
# pl.show()
#
# pl.plot(x, master.sub_flare_model(flare).flatten())
# pl.plot(x, flare.flux)
# pl.xlim(3270, 3290)
# pl.show()
# '''End flare analysis'''
