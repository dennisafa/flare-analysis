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


class strPlot:

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

    def __init__(self, star, range1, range2, model=None): # cleans the list of flux, normalizes to 0
        sap = star.remove_outliers().remove_nans()
        self.flux = sap.flux[range1:range2]
        self.time = sap.time[range1:range2]
        self.flux = (self.flux / np.median(self.flux))

    def guesspeaks(self, std): # gathers the peaks in the set of data, then returns a list of flare times, peaks, and fwhm
        self.detflares = fd.flaredetectpeak(self.flux, std)
        self.flarecount = fd.getlength()
        self.nflares = np.shape(self.detflares)[0]
        self.params = np.zeros([self.nflares, 3])
        for i, flareVal in enumerate(self.detflares):
            self.flarepeak = flareVal
            self.flaretime = self.findfluxtime(self.flarepeak, self.flux, self.time)
            p = [self.flaretime, self.flarepeak/2, self.flarepeak]
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

def flare_detect(period, flat_flux, clean_flux, orig_flux, time):
    std = np.std(flare.flux) * 2
    det_flares = fd.flaredetectpeak(flat_flux, std)
    flare_times = findflaretime(det_flares, flat_flux, time)
    print(period)

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

def findflaretime(flarepeak, flux, time): # retrieves the time of the flare
    flare_times = []
    for i, flare in enumerate(flarepeak):
        t = flux.index(flare)
        flare_times.append(time[t])
    return flare_times


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

def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return gp.log_likelihood(f)

def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(f)

def remove_flares(flare):
    std = np.std(flare.flux) * 2
    first_model = sub_flare_model(flare, std).flatten()
    flare_times = flare_start_end(first_model, flare.time, flare.flux)
    flux_orig = flare.flux

    while len(fd.flaredetectpeak(flare.flux, std)) > 0:# while flares are still being detected, compute its model and subtract flares
        tempmodel = sub_flare_model(flare, std)
        flare.flux = flare.flux-tempmodel.flatten()
        print("Flares subtracted!")


    return flare


def flare_start_end(model, time, flux):
    i = 0
    flare_counter = 0
    duration = []

    while i < len(model):
        if model[i] > 0.001:
            start = i-1
            while model[i] - model[i+1] < 0 and i < len(model) - 1:
                i+=1

            peak = model[i]
            i+=1
            while model[i] > 0.001 and i < len(model) - 1:
                if model[i] - model[i+1] < 0 and model[i] < peak/2:
                    break
                else:
                    i+=1

            end = i
            flare_counter+=1
            print('Flare # ', flare_counter)
            ed = np.abs(np.trapz(flux[start:end], time[start:end] * 86400))
            duration.append(ed)
        i+=1





    # using the techniques davenport wrote to get energy/plot ED

    days = time[-1] - time[0]
    exptime = 1. / 24. / days # davenports methods

    totdur = float(len(time)) * exptime
    duration = np.sort(duration)[::-1]

    ddx = np.log10(duration)
    ddy = (np.arange(len(ddx)) + 1) / totdur
    pl.plot(ddx, ddy, 'o-', markersize=3)
    pl.yscale('log')
    pl.ylim(1e-2, 1e2)
    pl.xlabel('log Equivalent Duration (seconds)')
    pl.ylabel('Cumulative Flares per Day')
    pl.show()

    L_m6_most = 10 ** 30.61
    L_m8_m6 = L_m6_most / 3.
    E_point = L_m8_m6

    pl.plot(ddx + np.log10(E_point), ddy, 'o-', markersize=3)
    pl.yscale('log')
    pl.ylim(1e-2, 1e2)
    pl.xlabel('log Energy (erg)')
    pl.ylabel('Cumulative Flares per Day')
    pl.show()



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

def detect_period(flux, time):

    dx = np.diff(time) / np.diff(flux)
    period_change = sign_change(dx, time)
    return period_change

def sub_flare_model(flare, std):
    guessparams = flare.guesspeaks(std)  # returns the parameters of where and when each flare occurred

    bounds = flare.setbounds(guessparams)
    fitparams = flare.fit(guessparams, bounds)  # fit the parameters with a minimization process
    model = flare.getmodel(fitparams, [flare.time, flare.flux,
                                           flare.nflares])
    return model

def get_period_change(period):
    avg_period = []
    for i, p in enumerate(period):
        if i < len(period) - 1:
            avg_period.append(period[i + 1] - period[i])

    return np.average(avg_period)



class FinalModelGeorge:
    period = []
    flat_flux = []
    rotation = []
    clean_flux = []
    orig_flux = []
    flux_rotation = []


    def subtract_flares(self, flare):


        flux_orig = flare.flux


        sf_model = sf(flare.flux, 501, 3)
        self.rotation = sf(sf_model, 501, 3)
        for i in range(10):
            self.rotation = sf(self.rotation, 501, 3)
        self.flat_flux = flare.flux - self.rotation
        flare.flux = self.flat_flux

        #flare.flux = pd.rolling_median(flare.flux, 100, center=True)

        first_model = sub_flare_model(flare, std=np.std(flare.flux) * 2).flatten()
        pl.plot(flare.time, first_model, linestyle='--', color='Blue')
        pl.plot(flare.time, flare.flux, color='Black')
        pl.xlabel('BJD')
        pl.ylabel('Normalized Flux')
        pl.show()

        flare_times = flare_start_end(first_model, flare.time, flare.flux)
        #flare = remove_flares(flare)
        # flux_rotation = flare.flux + self.rotation
        # self.clean_flux = [x+1 for x in flare.flux]
        # self.flat_flux = [x+1 for x in self.flat_flux]
        # self.orig_flux = flux_orig
        #
        # self.period = sf(flux_rotation, 501, 3)
        # for i in range(2000):
        #     self.period = sf(self.period, 501, 3)

        #self.create_final_model(flare)

        #self.iter_model(flare)


def george_test():


    file = np.genfromtxt("248432941.txt", dtype=float, usecols=(0,1), delimiter=',')
    y = file[:, 1]
    x = file[:, 0]

wolf = KeplerTargetPixelFile.from_archive('201885041', cadence='short')
lc359 = wolf.to_lightcurve(aperture_mask=wolf.pipeline_mask)

# fits_file = fits.open('/Users/Dennis/Desktop/newwolfdata/files/ktwo201885041_01_kasoc-ts_slc_v1.fits')
# flux = fits_file[1].data['flux']
# time = fits_file[1].data['time']
# pl.plot(flux)
# pl.show()


#fits = KeplerTargetPixelFile('/Users/Dennis/Desktop/newwolfdata/files/ktwo201885041_02_kasoc-ts_llc_v1.fits')

print("Creating model...")
flare = strPlot(lc359, 0, len(lc359.flux))

get = FinalModelGeorge()


get.subtract_flares(flare)
# flat_flux = get.flat_flux
# clean_flux = get.clean_flux
# orig_flux = get.orig_flux
# period_list = get.period
# period = detect_period(period_list, flare.time)
# flare_detect(period, flat_flux, clean_flux, flat_flux, flare.time)


print("Finished")





