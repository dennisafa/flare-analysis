from lightkurve import KeplerLightCurveFile
from george import kernels
import matplotlib.pyplot as pl
import numpy as np
import george
from scipy.optimize import minimize
from astropy.stats import median_absolute_deviation


def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y)

def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y)


strLC = KeplerLightCurveFile.from_archive(206208968)
strPDC = strLC.PDCSAP_FLUX.remove_outliers()

y = strPDC.flux[:300]
x = strPDC.time[:300]
y = (y/np.median(y)) - 1 # sets the function to begin at 0
x = x[np.isfinite(y)]
y = y[np.isfinite(y)] # removes NaN values

pl.plot(x,y)
pl.show()
print(median_absolute_deviation(y))
print(np.var(y))

def computeModel():
    global gp
    kernel = np.var(y) * kernels.ExpSquaredKernel(0.5) * kernels.ExpSine2Kernel(log_period = 0.5, gamma=1)
    gp = george.GP(kernel)
    gp.compute(x, y)
    model = gp.predict(y, x, return_var=True)
    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    gp.set_parameter_vector(result.x)

def plotModel ():
    pred_mean, pred_var = gp.predict(y, x, return_var=True)
    #pred_std = np.sqrt(pred_var)
    #pl.errorbar(x,y, yerr=np.var(y) ** 0.5/10, fmt='k.')
    pl.plot(x, pred_mean, '--r')
    pl.plot(x, y, lw= 1, color = 'blue')
    pl.ylabel('Relative Flux')
    pl.xlabel('Time - BKJDC')
    pl.show()