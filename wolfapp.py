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
from wolf359.base_luminosity import base_lum as lum
from wolf359.flareenergy import energy_calc

import scipy.ndimage.filters as gausFilter


class Flare:

    def findflaremax(self, flux):
        return max(flux)


    def findfluxtime(self, flarepeak, flux, time):  # retrieves the time of the flare
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
            model += ap.aflare1(time, tpeak=p[i, 0], fwhm=p[i, 1], ampl=p[i, 2], upsample=False, uptime=10)
        return model

    def ng_ln_like(self, p, data):
        _, y, _ = data
        model = self.getmodel(p, data)
        return np.sum((model - y) ** 2)

    def __init__(self, flux, time, range1, range2, model=None): # cleans the list of flux, normalizes to 0
        self.flux = flux[range1:range2]
        self.time = time[range1:range2]
        # self.flux = self.flux[np.logical_not(np.isnan(self.flux))]
        #
        # self.time = time[:len(self.flux)]
        #
        # pl.plot(self.time, self.flux)
        # pl.show()
        # self.smo = pd.rolling_median(self.flux, 100, center=True)
        # self.flux = (self.flux - self.smo / np.median(self.flux)) + 0.5
        # pl.plot(self.time, self.flux)
        # pl.show()

    def guesspeaks(self): # gathers the peaks in the set of data, then returns a list of flare times, peaks, and fwhm
        self.detflares = fd.flaredetectpeak(self.flux)
        self.flarecount = fd.getlength()
        self.nflares = np.shape(self.detflares)[0]
        self.params = np.zeros([self.nflares, 3])
        for i, flareVal in enumerate(self.detflares):
            self.flarepeak = flareVal
            self.flaretime = self.findfluxtime(self.flarepeak, self.flux, self.time)
            p = [self.flaretime, 0.002, self.flarepeak]
            self.params[i, :] = p
        return np.log(self.params)


    def fit(self, p):
        result = minimize(self.ng_ln_like, p, args=[self.time, self.flux, self.nflares], method='L-BFGS-B')
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



'''George modeling'''
def computegeorge (flux, time):

    f = flux

    kernel = kernels.CosineKernel(log_period=np.log(40), axes=0) * kernels.ExpSquaredKernel(metric=0.5) + 1
    gp = george.GP(kernel)
    gp.compute(time, flux)
    pred_mean, pred_var = gp.predict(flux, time, return_var=True)
    #print(result)
    return pred_mean


'''Count the total flares'''
def flare_count(flat_flux, time):

    det_flares = fd.flaredetectpeak(flat_flux)
    flare_times = findflaretime(det_flares, flat_flux, time)
    print(len(det_flares))


def findflaretime(flarepeak, flux, time): # retrieves the time of the flare
    time = list(time)
    flux = list(flux)
    flare_times = []
    for i, flare in enumerate(flarepeak):
        t = flux.index(flare)
        flare_times.append(time[t])
    return flare_times


'''Count the total flares per period'''
def flares_per_period(flare_times, period, time):

    i = 0
    j = 0
    avg_period = []
    total = 0
    while i < len(period) - 1:
        count = 0
        avg_period.append(time[period[i + 1]] - time[period[i]])
        if j < len(flare_times):
            while flare_times[j] < time[period[i+1]] and j < len(flare_times) - 1:
                count += 1
                j += 1

            print("period {} had {} flares".format(i, count))
        total += count
        i+=1
    print("avg period is {} days".format(np.average(avg_period)))
    print ("total flare events {}".format(total))


'''Bounds for minimization'''
def setbounds(flux):
    flux = asarray(flux).ravel()
    bounds = np.zeros([len(flux),2])
    for i in range(len(flux)):
        for j in range(2):
            if j < 1:
                bounds[i][j] = flux[i] - (flux[i] ** 1/20 - flux[i])
            else:
                bounds[i][j] = flux[i] ** 1/20
    return bounds

'''Flare removal'''
def remove_flares(flare):

    model = sub_flare_model(flare)
    pl.plot(flare.time, model.flatten())
    pl.plot(flare.time, flare.flux)
    pl.show()

    while len(fd.flaredetectpeak(flare.flux)) > 0:# while flares are still being detected, compute its model and subtract flares
        tempmodel = sub_flare_model(flare)
        flare.flux = flare.flux-tempmodel.flatten()
        print("Flares subtracted!")
    return flare

'''Flare modeling and fitting'''
def sub_flare_model(flare):

    guessparams = flare.guesspeaks() # returns the parameters of where and when each flare occurred
    fitparams = flare.fit(guessparams)
    model = flare.getmodel(fitparams, [flare.time, flare.flux,
                                           flare.nflares])
    return model


'''Getting the period with a polynomial fit'''
def get_period_change(period):
    avg_period = []
    for i, p in enumerate(period):
        if i < len(period) - 1:
            avg_period.append(period[i + 1] - period[i])

    return np.average(avg_period)

def detect_period(flux, time):

    dx = np.diff(time) / np.diff(flux)
    period_change = sign_change(dx, time)
    return period_change


def sign_change(model, time): # returns indices where the first derivative changes sign from positive to negative
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


class Process:
    period = []
    flat_flux = []
    rotation = []
    clean_flux = []
    orig_flux = []
    flux_rotation = []


    def subtract_flares(self, flare, wl):

        # self.rotation = sf(flare.flux, wl, 3)
        # self.flat_flux = (flare.flux - self.rotation)
        # flare.flux -= self.rotation
        #
        # pl.plot(flare.time, flare.flux)
        # pl.show()
        #
        #
        # flare.flux = flare.flux[np.logical_not(np.isnan(flare.flux))]
        # flare.time = flare.time[:len(flare.flux)]

        model = sub_flare_model(flare)
        # pl.plot(flare.time, model.flatten())
        # pl.plot(flare.time, flare.flux)
        # pl.show()


        return energy_calc(model.flatten(), flare.time, flare.flux)



def wolf():
    ''' Wolf analysis'''
    wolf = KeplerTargetPixelFile.from_archive('201885041', cadence='short')
    lc359 = wolf.to_lightcurve(aperture_mask=wolf.pipeline_mask)
    # fits_file = fits.open('/Users/Dennis/Desktop/newwolfdata/files/ktwo201885041_01_kasoc-ts_slc_v1.fits')
    file = np.genfromtxt("/Users/Dennis/Desktop/newwolfdata/tweaked_SC_timeseries.txt", dtype=float, usecols=(0, 2),
                         delimiter=' ')

    pl.plot(lc359.time, lc359.flux)
    pl.show()


    smoothed2 = gausFilter.gaussian_filter1d(lc359.flux, 20)
    pl.plot(smoothed2[:400])
    pl.plot(lc359.flux[:400])
    pl.show()

    finalPlot = (lc359.flux - smoothed2) / np.median(lc359.flux)
    flare_count(finalPlot, lc359.time)

    for i in range(0, 10000, 1000):
        f1 = Flare(finalPlot, lc359.time, i, i+1000)
        model = sub_flare_model(f1)
        print(len(model.flatten()))
        if (len(model.flatten()) > 0):
            pl.plot(lc359.time[i:i+1000], finalPlot[i:i+1000])
            pl.plot(lc359.time[i:i+1000], model[i:i+1000].flatten())
            pl.show()
            pl.clf()
        # pl.plot(lc359.time[i:i+1000], lc359.flux[i:i+1000])
        # pl.show()
        # pl.clf()

    # y = file[:, 1]
    # x = file[:, 0]
    # y = y[np.logical_not(np.isnan(y))]
    # x = x[:len(y)]
    # y = (y - np.min(y)) / (np.max(y) - np.min(y))
    #
    # pl.plot(x, y)
    # pl.show()
    # flux = y
    # time = x
    # print("Creating model...")


    # duration = []
    # time_end = 0
    # points_lost = 0
    # for i in range(0, 50000, 1000):
    #     flare = Flare(finalPlot,lc359.time, i, i+1000)
    #     get = Process()
    #     duration += get.subtract_flares(flare, 101)
    #     time_end = flare.time[-1]
    #
    #
    # #print(duration)
    #
    # exptime = 1. / 24. / 80.
    #
    # pl.plot(np.log10(duration))
    # pl.show()
    #
    # totdur = float(len(lc359.time)) * exptime
    # duration = np.sort(duration)[::-1]
    #
    # ddx = np.log10(duration)
    # ddy = (np.arange(len(ddx))) / totdur
    # pl.plot(ddx, ddy, 'o', markersize=3, alpha=0.5)
    # pl.yscale('log')
    # pl.ylim(1e-2, 1e2)
    # pl.xlabel('log Equivalent Duration (seconds)')
    # pl.ylabel('Cumulative Flares per Day')
    # pl.show()
    # pl.clf()
    #
    # E_point = lum(12.840, 2, 4000)
    # print(E_point)
    # print(ddx)
    #
    # pl.plot(ddx + E_point, ddy, 'o', markersize=4, alpha=0.5)
    # pl.yscale('log')
    # pl.ylim(1e-2, 1e2)
    # pl.xlabel('log Flare Energy (erg)')
    # pl.ylabel('Cumulative Flares per Day')
    # pl.show()

    # flat_flux = get.flat_flux
    # clean_flux = get.clean_flux
    # orig_flux = get.orig_flux
    # period_list = get.period
    # period = detect_period(period_list, flare.time)
    # flare_detect(period, flat_flux, clean_flux, flat_flux, flare.time)


'''End wolf analysis'''


def dipper():
    '''Pixel file dipper star'''
    tpf = KeplerTargetPixelFile.from_archive('248432941')
    tpf.remove_outliers()
    aper = np.zeros(tpf.shape[1:])
    aper[:, :] = 1
    lc = tpf.to_lightcurve(aperture_mask=aper.astype(bool))
    lc.correct(windows=20, bins=10).bin(20).plot()
    pl.show()
    pl.clf()

    for j in range(0, 18, 4):

        for i in range(0, 18, 4):
            aper = np.zeros(tpf.shape[1:])
            aper[i:i+4, j:j+4] = 1
            tpf.plot(aperture_mask=aper)
            pl.show()
            lc = tpf.to_lightcurve(aperture_mask=aper.astype(bool))
            lc.correct(windows=20, bins=10).bin(20).plot()
            pl.ylim(0.9991, 1.0007)
            pl.show()
            pl.clf()

    '''End pixel file'''

# fits_file = fits.open('/Users/Dennis/Desktop/newwolfdata/files/ktwo201885041_01_kasoc-ts_slc_v1.fits')
# flux = fits_file[1].data['flux_raw']
# time = fits_file[1].data['time']
#
# print(flux[:500])
#
#
# pl.plot(time, flux)
#
# pl.show()

