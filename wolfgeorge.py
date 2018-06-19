from lightkurve import KeplerLightCurveFile
from george import kernels
import matplotlib.pyplot as pl
import numpy as np
import george
from scipy.optimize import minimize
from astropy.stats import median_absolute_deviation\

def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y)

def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y)

wolf = KeplerLightCurveFile.from_archive(201885041)
wolfPDC = wolf.PDCSAP_FLUX.remove_nans()

y = wolfPDC.flux[:300]
x = wolfPDC.time[:300]
y = (y/np.median(y)) - 1
x = x[np.isfinite(y)]
y = y[np.isfinite(y)]

#pl.plot(x,y)
#pl.show()

def computeModel():
    global gp
    kernel = np.var(y) *  kernels.Matern52Kernel(metric=0.5) * kernels.CosineKernel(log_period=0.5)
    gp = george.GP(kernel)
    gp.compute(x,y)
    #pred = gp.predict(y, x, return_var=True)
    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    gp.set_parameter_vector(result.x)

def plotModel():
    pred, pred_var = gp.predict(y, x, return_var=True)
    pl.plot(x, pred)
    pl.plot(x,y)
    pl.show()