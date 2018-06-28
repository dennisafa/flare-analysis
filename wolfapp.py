from appaloosa import aflare as ap
from lightkurve import KeplerTargetPixelFile, KeplerLightCurveFile
import matplotlib.pyplot as pl
import numpy as np
import wolf359.flaredetect as fd
import scipy as scipy
from scipy.optimize import minimize
import time
import os
from numpy import asarray
import george
from george import kernels

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
        sap = star.remove_outliers()
        self.flux = sap.flux[range1:range2]
        self.time = sap.time[range1:range2]
        self.flux = (self.flux/np.median(self.flux)) - 1
        self.flux = [number / scipy.std(self.flux) for number in self.flux]
        self.time = self.time[np.isfinite(self.flux)]
        self.flux = asarray(self.flux)
        self.flux = self.flux[np.isfinite(self.flux)]


    def guesspeaks(self, sliceNum): # gathers the peaks in the set of data, then returns a list of flare times, peaks, and fwhm
        self.detflares = fd.flaredetect(self.flux, sliceNum)
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

    def setbounds (self, p):
        p = asarray(p).ravel()
        bounds = np.zeros([len(p), 2])
        k = 0
        for i in range(len(p)):
            for j in range(2):
                if j < 1:
                    bounds[i][j] = p[i]
                else:
                    bounds[i][j] = p[i] + p[i] ** 1/6
        return bounds

    def computegeorge (self):
        kernel = np.var(self.flux) * kernels.ExpSquaredKernel(0.5) * kernels.ExpSine2Kernel(log_period = 0.5, gamma=1)
        self.gp = george.GP(kernel) # testing
        self.gp.compute(self.time, self.flux)
        result = minimize(self.neg_ln_like, self.gp.get_parameter_vector(), jac=self.grad_neg_ln_like)
        self.gp.set_parameter_vector(result.x)
        pred_mean, pred_var = self.gp.predict(self.flux, self.time, return_var=True)
        return pred_mean



def load():
    w359 = KeplerTargetPixelFile.from_archive(201885041, cadence='short')
    lc359 = w359.to_lightcurve(aperture_mask=w359.pipeline_mask)
    
def call(lc359, flux):
    # wolf359: 201885041
    # josh's star: 201205469
    # ran: 206208968
    w359 = KeplerTargetPixelFile.from_archive(201885041, cadence='short')
    lc359 = w359.to_lightcurve(aperture_mask=w359.pipeline_mask)
    track = 0
    steps = 100
    length = 1000
    for slice in range(0, 100, steps):

        flare = strPlot(lc359, slice, slice + steps)
        guessparams = flare.guesspeaks(track)
        notempty = checkzero(flare.detflares)
        if notempty:
            georgemodel = flare.computegeorge()
            bounds = flare.setbounds(guessparams)

            fitparams = flare.fit(guessparams, bounds)
            model = flare.getmodel(fitparams, [flare.time, flare.flux, flare.nflares]) + georgemodel
            plotflares(flare, model, track)

            slice += steps
            track += 1
            print("Success at range {}".format(slice))
        else:
            print("No flares in slice {}".format(slice))
            slice+=steps
            track+=1

def checkzero(l):
    if len(l) == 0:
        return False
    else:
        return True

def plotflares(flare, model, track):
    for it, flux in enumerate(flare.detflares):
        pl.plot(flare.params[it, 0], flux, marker='x', markersize=4, color="black")


    pl.plot(flare.time, model.flatten(), '--r')
    pl.plot(flare.time, flare.flux, color='Grey', lw=0.5)
    pl.xlabel('Time - BYJD')
    pl.ylabel('Flux - Normalized to 0')
    pl.show()
    pl.clf()

    while flare.guesspeaks(track, model.flatten())
    pl.plot(flare.time, (flare.flux-model.flatten()), color = 'Black', label = 'Subtracted flares')
    pl.plot(flare.time, flare.flux, color = 'Grey', linestyle = '--', label = 'Original')
    pl.legend(loc = 'upper-left')
    pl.xlabel('Time - BYJD')
    pl.ylabel('Flux - Normalized to 0')
    pl.show()


    savepath = os.path.join('/Users/Dennis/Desktop/plotswolf/test', 'wolf' + str(track) + '.png')
    pl.savefig(savepath)
    pl.clf()

def cleandata(star, range1, range2):
    sap =  star.remove_outliers()
    flux = sap.flux[range1:range2]
    time = sap.time[range1:range2]
    flux = (flux / np.median(flux)) - 1
    flux = [number / scipy.std(flux) for number in flux]
    time = time[np.isfinite(flux)]
    flux = asarray(flux)
    flux = flux[np.isfinite(flux)]