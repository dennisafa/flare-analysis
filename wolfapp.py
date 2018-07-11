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
import os

class strPlot:

    def neg_ln_like(self, p):
        self.gp.set_parameter_vector(p)
        return -self.gp.log_likelihood(self.flux)

    def grad_neg_ln_like(self, p):
        self.gp.set_parameter_vector(p)
        return -self.gp.grad_log_likelihood(self.flux)


    def findflaremax(self, flux):
        return max(flux)

    def findflaretime(self, flarepeak, flux, time): # retrieves the time of the flare
        tof = time
        for i, flare in enumerate(flux):
            if flare == flarepeak:
                return tof[i]

    def getmodel(self, p, data): # computes the model of the flares using appaloosa's aflare1 function
        time, y, nflares = data
        p = np.exp(p)
        model = np.zeros_like([time])
        p = np.reshape(p, (nflares, 3))
        for i in range(nflares):
            model += ap.aflare1(time, tpeak=p[i, 0], fwhm=p[i, 1], ampl=p[i, 2], upsample=True, uptime=10)
        return model

    def ng_ln_like(self, p, data):
        _, y, _ = data
        model = self.getmodel(p, data)
        return np.sum((model - y) ** 2)

    def __init__(self, star, range1, range2): # cleans the list of flux, normalizes to 0
        sap = star.remove_outliers().remove_nans()
        self.flux = sap.flux[range1:range2]
        self.time = sap.time[range1:range2]
        self.flux = (self.flux/np.median(self.flux)) - 1
        self.flux = [number / scipy.std(self.flux) for number in self.flux]
        self.time = self.time[np.isfinite(self.flux)]
        self.flux = asarray(self.flux)
        self.flux = self.flux[np.isfinite(self.flux)]


    def guesspeaks(self): # gathers the peaks in the set of data, then returns a list of flare times, peaks, and fwhm
        self.detflares = fd.flaredetect(self.flux)
        self.flarecount = fd.getlength()
        self.nflares = np.shape(self.detflares)[0]
        self.params = np.zeros([self.nflares, 3])
        for i, flareVal in enumerate(self.detflares):
            self.flarepeak = flareVal
            self.flaretime = self.findflaretime(self.flarepeak, self.flux, self.time)
            p = [self.flaretime, 0.004, self.flarepeak]
            self.params[i, :] = p
        return np.log(self.params)


    def fit(self, p, bounds):
        result = minimize(self.ng_ln_like, p, args=[self.time, self.flux, self.nflares], method='L-BFGS-B', bounds=bounds)
        return result.x

    def min(self):
        result = minimize(self.ng_ln_like, np.log(self.params), args=[self.time, self.flux, self.nflares], method='L-BFGS-B', bounds=self.bounds)
        return result.x

    def setbounds (self, p):
        p = asarray(p).ravel()
        bounds = np.zeros([len(p), 2])
        for i in range(len(p)):
            for j in range(2):
                if j < 1:
                    bounds[i][j] = p[i]
                else:
                    bounds[i][j] = p[i] + p[i] ** 1/6
        self.bounds = bounds
        return bounds

    def computegeorge (self):
        testmodel = 1 * np.sin(2*np.pi/ 2.5) * self.flux

        kernel = kernels.ExpSine2Kernel(gamma=10, log_period=0.75) * kernels.Matern52Kernel(metric=0.5)  # a * cos(2pi/T * (t-3082) ) *
        self.gp = george.GP(kernel)
        self.gp.compute(self.time, self.flux)
        print(kernel)
        #result = minimize(self.neg_ln_like, self.gp.get_parameter_vector(), jac=self.grad_neg_ln_like)
        #self.gp.set_parameter_vector(result.x)
        pred_mean, pred_var = self.gp.predict(self.flux, self.time, return_var=True)
        return pred_mean


def checkzero(l):
    if len(l) == 0:
        return False
    else:
        return True


def remove_flares(flare):

    while len(fd.flaredetect(flare.flux)) > 0:# while flares are still being detected, compute its model and subtract flares
        tempmodel = sub_flare_model(flare)
        flare.flux = flare.flux-tempmodel.flatten()
    return flare.flux

def computegeorge (flux, time):

    kernel = kernels.ExpSine2Kernel(gamma=10, log_period=0.75) * kernels.Matern52Kernel(metric=0.5)  # a * cos(2pi/T * (t-3082) ) *
    gp = george.GP(kernel)
    gp.compute(time, flux)
    pred_mean, pred_var = gp.predict(flux, time, return_var=True)
    return pred_mean


def sign_change(model): # returns indices where the first derivative changes sign from positive to negative
    change_sign = []
    j = 0
    while j < len(model)-1:
        if model[j] > 0:
            while model[j] > 0 and j < len(model)-1:
                j+=1
            else:
                change_sign.append(j)
        else:
            j+=1
    return change_sign

def make_period_fit(time, flux):

    dx = np.diff(time) / np.diff(flux)
    period_change = sign_change(dx)
    return period_change


def sub_flares(flare, period, phase, range1, range2): # loops through a subset of flux values, detects flares, then subtracts and performs again
    global flare_orig
    flare_orig = flare.flux
    finalmodel = 0
    i = 0
    while len(fd.flaredetect(flare.flux)) > 0:# while flares are still being detected, compute its model and subtract flares
        tempmodel = sub_flare_model(flare)
        finalmodel += tempmodel
        flare.flux = flare.flux-tempmodel.flatten()
        i+=1

    if i > 1:

        pl.plot(flare.time, flare_orig,color = 'Blue', label = 'Original flux')
        pl.plot(flare.time, period[range1:range2], color='Grey', label = 'Period model')
        pl.plot(flare.time, finalmodel.flatten(),color = "Black", linestyle= '--', label = 'Flare model')
        pl.legend(loc='upper left')
        pl.xlabel('Time - BJD')
        pl.ylabel('Flux - Normalized 0')
        pl.show()
        print ("Number of flares in rotational period {}, = {}".format(phase, len(fd.flaredetect(finalmodel.flatten()))))
        #print(fd.getlength())
        return len(fd.flaredetect(finalmodel.flatten()))
    return 0


def sub_flare_model(flare):
    guessparams = flare.guesspeaks()  # returns the parameters of where and when each flare occured
    #print(flare.flarecount)
    notempty = checkzero(flare.detflares)  # checks for flares to exit in the data set before modeling
    if notempty:
        bounds = flare.setbounds(guessparams)
        fitparams = flare.fit(guessparams, bounds)  # fit the parameters with a minimization process
        model = flare.getmodel(fitparams, [flare.time, flare.flux,
                                           flare.nflares])
        return model


def run():
    # stars to test george on: 206208968, 201205469, 201885041

    fits = KeplerTargetPixelFile("/Users/Dennis/Desktop/fits/ktwo206208968-c03_lpd-targ.fits")
    #test = KeplerTargetPixelFile.from_fits_images("/Users/Dennis/Desktop/fits/ktwo201885041-c14_lpd-targ.fits")
    lc359 = fits.to_lightcurve(aperture_mask=fits.pipeline_mask)
    flare_star = strPlot(lc359, 0, len(lc359.flux))

    flare_count = []
    count = 300

    for i in range(0, len(lc359.flux), count):
        if i < len(lc359.flux) - 1:
            flare = strPlot(lc359, i, i+count)
            mod_flare = strPlot(lc359, i, i+count)
            clean_flare = remove_flares(mod_flare)
            george_model = computegeorge(clean_flare, flare.time)
            period = make_period_fit(mod_flare.time, george_model)
            for i, n in enumerate(period):
                if i < len(period) - 1:
                    flare = strPlot(lc359, period[i], period[i + 1])
                    sub_flares(flare, george_model, i, period[i], period[i+1])







    print("Average number of flares = {} for period = {}".format(np.average(flare_count), avgperiod))



def isdirtest():
    boolisTrue = os.path.isdir("/Users/Dennis/Desktop/fits/ktwo201885041-c14_lpd-targ.fits")
    print(boolisTrue)

def georgetest():
    fits = KeplerTargetPixelFile("/Users/Dennis/Desktop/fits/ktwo201885041-c14_lpd-targ.fits")
    lc359 = fits.to_lightcurve(aperture_mask=fits.pipeline_mask)

    period_flare = strPlot(lc359, 0, 500)
    model = period_flare.computegeorge()
    pl.plot(period_flare.time, model)
    pl.plot(period_flare.time, period_flare.flux)
    pl.show()

'''
To do: 
1) Use final model to calculate noise, modify flaredetect directly 
2) Don't run final model through flaredetect, it cancels out flares

@Update 
'''