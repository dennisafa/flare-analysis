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

    def __init__(self, star, range1, range2, model=None): # cleans the list of flux, normalizes to 0
        sap = star.remove_outliers().remove_nans()
        if model is not None:
            self.flux = model[range1:range2]
        else:
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
            p = [self.flaretime, 0.05, self.flarepeak]
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

        kernel = kernels.ExpSine2Kernel(gamma=35, log_period=0.75) * kernels.Matern52Kernel(metric=0.5)  # a * cos(2pi/T * (t-3082) ) *
        self.gp = george.GP(kernel)
        self.gp.compute(self.time, self.flux)
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

    kernel = kernels.ExpSine2Kernel(gamma=15, log_period=0.75) * kernels.Matern52Kernel(metric=0.5)  # a * cos(2pi/T * (t-3082) ) *
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

def detect_period(flux, time):

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

    if i > 0:

        final_count = fd.model_peaks(list(finalmodel.flatten()))
        if final_count > 0:
            print("{} flares detected in period {}".format(final_count, phase))
            pl.plot(flare.time, flare_orig, color='Blue', label='Original flux')
            #pl.plot(flare.time, period[range1:range2], color='Grey', label='Period model')
            pl.plot(flare.time, finalmodel.flatten(), color="Black", linestyle='--', label='Flare model')
            pl.legend(loc='upper left')
            pl.xlabel('Time - BJD')
            pl.ylabel('Flux - Normalized 0')
            pl.show()

        #print(fd.getlength())
    else:
        print("No flares detected in period {}".format(phase))



def sub_flare_model(flare):
    guessparams = flare.guesspeaks()  # returns the parameters of where and when each flare occured
    #notempty = checkzero(flare.detflares)  # checks for flares to exit in the data set before modeling

    bounds = flare.setbounds(guessparams)
    fitparams = flare.fit(guessparams, bounds)  # fit the parameters with a minimization process
    model = flare.getmodel(fitparams, [flare.time, flare.flux,
                                           flare.nflares])
    return model


def run():
    # stars to test george on: 206208968, 201205469, 201885041

    #fits = KeplerTargetPixelFile("/Users/Dennis/Desktop/fits/ktwo206208968-c03_lpd-targ.fits")
    fits = KeplerTargetPixelFile("/Users/Dennis/Desktop/fits/ktwo206208968-c03_lpd-targ.fits")
    lc359 = fits.to_lightcurve(aperture_mask=fits.pipeline_mask)


    print("Creating model...")
    flare = strPlot(lc359, 0, len(lc359.flux))
    period = detect_period(flare.flux, flare.time)
    final_model = create_final_model(flare)
    flare.flux = final_model

    appaloosa_model = sub_flare_model(flare)
    pl.plot(appaloosa_model.flatten())
    pl.show()
    pl.clf()

    pl.plot(flare.time[:300], flare.flux[:300])
    pl.plot(flare.time[:300], appaloosa_model.flatten()[:300])
    pl.show()





    # avg_period = []
    #
    # for i, p in enumerate(period):
    #     if i < len(period) - 1:
    #         avg_period.append(period[i+1] - period[i])
    #
    # typ_period = np.average(avg_period)
    #
    # for i, n in enumerate(period):
    #     if i < len(period) - 1:
    #         if period[i+1] - period[i] > typ_period:
    #             flare = strPlot(lc359, period[i], period[i + 1], model=final_model)
    #             sub_flares(flare, period, i, period[i], period[i+1])

def create_final_model(flare):
    george_model = computegeorge(flare.flux, flare.time)  # create an initial model
    sub_model = flare.flux - george_model  # subtract the george model from the raw data
    george_model2 = computegeorge(sub_model, flare.time)  # create a model of the data with george model subbed
    clean_model2 = george_model2 + george_model  # plot the new model
    final_plot = flare.flux - clean_model2

    return final_plot


def georgetest():
    fits = KeplerTargetPixelFile("/Users/Dennis/Desktop/fits/ktwo206208968-c03_lpd-targ.fits")
    lc359 = fits.to_lightcurve(aperture_mask=fits.pipeline_mask)

    period_flare = strPlot(lc359, 0, 500)
    model = period_flare.computegeorge()
    pl.plot(period_flare.time, model)
    pl.plot(period_flare.time, period_flare.flux)
    pl.show()

def plot_final_model(flare, george_model, sub_model, george_model2, final_plot):
    figplot = pl.figure(figsize=(10, 10))
    figplot.subplots_adjust(hspace=0.4, wspace=0.4)
    raw_data = figplot.add_subplot(2, 2, 1)
    with_george = figplot.add_subplot(2, 2, 2)
    first_reduce = figplot.add_subplot(2, 2, 3)
    second_reduce = figplot.add_subplot(2, 2, 4)

    raw_data.set_ylabel("Normalized Flux")
    raw_data.set_xlabel("BJD")
    raw_data.plot(flare.time, flare.flux, color="Black", label="Raw flux")
    # raw_data.legend(loc='upper left')

    with_george.set_ylabel("Normalized Flux")
    with_george.set_xlabel("BJD")
    with_george.plot(flare.time, flare.flux, color="Black", label="Raw flux")
    with_george.plot(flare.time, george_model, color="Blue", linestyle='--', label="George Model")
    # raw_data.legend(loc='upper right')

    first_reduce.set_ylabel("Normalized Flux")
    first_reduce.set_xlabel("BJD")
    first_reduce.plot(flare.time, sub_model, color="Black", label="Flux w/ subtracted model")
    first_reduce.plot(flare.time, george_model2, color="Blue", linestyle='--')
    # raw_data.legend(loc='lower left')

    second_reduce.set_ylabel("Normalized Flux")
    second_reduce.set_xlabel("BJD")
    second_reduce.plot(flare.time, final_plot, color="Black")

    pl.show()

'''
To do: 
1) Use final model to calculate noise, modify flaredetect directly 
2) Don't run final model through flaredetect, it cancels out flares

@Update 
'''