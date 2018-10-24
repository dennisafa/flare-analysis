import george
from george import kernels
import numpy as np
from lightkurve import KeplerTargetPixelFile
import matplotlib.pyplot as pl
from appaloosa import aflare as ap
import wolf359.flaredetect as fd
import pandas as pd
from astropy.io import fits
from scipy.signal import savgol_filter as sf
from scipy.optimize import minimize
from numpy import asarray


class Flare:
    flux = []
    time = []

    def __init__(self, flux, time, r1,  r2):

        self.flux = flux[r1:r2]
        self.flux = self.flux[np.logical_not(np.isnan(self.flux))]
        self.flux = (self.flux - min(self.flux)) / (max(self.flux) - min(self.flux))
        self.time = time[:len(self.flux)]


    def guesspeaks(self): # gathers the peaks in the set of data, then returns a list of flare times, peaks, and fwhm
        self.detflares = fd.flaredetectpeak(self.flux)
        self.flarecount = fd.getlength()
        self.nflares = np.shape(self.detflares)[0]
        self.params = np.zeros([self.nflares, 3])
        for i, flareVal in enumerate(self.detflares):
            self.flarepeak = flareVal
            self.flaretime = self.findfluxtime(self.flarepeak, self.flux, self.time)
            p = [self.flaretime, 0.05, self.flarepeak]
            self.params[i, :] = p
        return np.log(self.params)


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


'''George modeling'''
def computegeorge (flux, time):
    global gp
    global y

    y = flux

    kernel = kernels.CosineKernel(log_period=np.log(40), axes=0) * kernels.ExpSquaredKernel(metric=0.5)
    gp = george.GP(kernel)
    gp.compute(time, flux)
    print('Bounds', gp.get_parameter_bounds())
    print(gp.log_prior())
    print('Initial log likelihood', gp.log_likelihood(y))
    print('initial parameter vector', gp.get_parameter_vector())
    res = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    gp.set_parameter_vector(res.x)
    print('Final log likelihood', gp.log_likelihood(y))
    print('final parameter vector', res.x)
    print(res)

    pred_mean, pred_var = gp.predict(flux, time, return_var=True)

    pl.fill_between(time, pred_mean - np.sqrt(pred_var), pred_mean + np.sqrt(pred_var), color='k', alpha=0.4,label= 'Predicted variance')
    pl.fill_between(time, flux - np.sqrt(gp._yerr2), flux + np.sqrt(gp._yerr2), color='k')
    pl.plot(time, pred_mean, color='Blue', label='Predicted mean')
    pl.plot(time, flux, alpha = 0.6, label='Raw flux')
    pl.xlabel("BJD")
    pl.ylabel("Normalized Flux")
    pl.ylim(-0.1, 1)
    pl.legend(loc='best')
    pl.show()
    return pred_mean

def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.log_likelihood(y)

def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y)


def setbounds(flux):
    flux = asarray(flux).ravel()
    bounds = np.zeros([len(flux), 2])
    for i in range(len(flux)):
        for j in range(2):
            if j < 1:
                bounds[i][j] = flux[i]
            else:
                bounds[i][j] = flux[i] ** 1/20
    return bounds

'''End George Modeling'''

def remove_flares (flare):
    flare_count = 0


    while len(fd.flaredetectpeak(flare.flux)) > 0:# while flares are still being detected, compute its model and subtract flares
        tempmodel = sub_flare_model(flare)
        flare.flux = flare.flux-tempmodel.flatten()
        flare_count = flare_count + 1


    print('{} flares subtracted'.format(flare_count))


    return flare


def sub_flare_model(flare):
    guessparams = flare.guesspeaks()
    model = flare.getmodel(guessparams, [flare.time, flare.flux,
                                         flare.nflares])
    print(model)
    return model


def flatten(flux, time):
        # 113,008 points
        flare = Flare(flux, time, 0, len(flux))

        sav_gol_model = sf(flare.flux, 101, 3)
        flare.flux -= sav_gol_model

        return flare

def running_sum(list, window):
    j = 0
    new = 0
    new_list = []
    for i in range(len(list)):
        if j > window:
            new_list.append(new)
            new = 0
            j = 0
        else:
            new+=list[i]
            j+=1

    return new_list







def run (flux, time):
    # fits_file = fits.open('/Users/Dennis/Desktop/newwolfdata/files/ktwo201885041_01_kasoc-ts_slc_v1.fits')
    # flux = fits_file[1].data['flux']
    # time = fits_file[1].data['time']
    # flux = flux[np.logical_not(np.isnan(flux))]
    # flux = (flux/ np.median(flux)) - 1
    # flux = (flux-min(flux))/(max(flux)-min(flux))
    #
    # flux = running_sum(flux, 30)
    # flux = np.array(flux)
    # flux = pd.rolling_median(flux, 100, center=True)
    # flux = flux[np.logical_not(np.isnan(flux))]
    # flux = (flux/ np.median(flux)) - 1
    # time = time[:len(flux)]
    # g = computegeorge(flux, time)
    # pl.plot(time, flux)
    # pl.plot(time, g)
    # pl.show()

    # w = 30
    # flux = running_sum(flux, w)
    # time = time[:len(flux)]

    print("Creating model...")

    wolf = KeplerTargetPixelFile.from_archive('201885041')
    lc359 = wolf.to_lightcurve(aperture_mask=wolf.pipeline_mask)

    # flare = Flare(lc359, 0, len(lc359.flux))

    # flare = flatten(flare)
    # flare.flux = pd.rolling_median(flare.flux, 100, center=True)
    # flare.flux = flare.flux[np.logical_not(np.isnan(flare.flux))]
    # flare.time = flare.time[:len(flare.flux)]
    #
    # g = computegeorge(flare.flux, flare.time)
    # pl.plot(g)
    # pl.show()

    # flare.flux = remove_flares(flare)
    #
    # pl.plot(flare.time, flare.flux)
    # pl.show()
    # pl.clf()
    #
    # flare.flux += clean_model
    #
    # pl.plot(flare.time, flare.flux)
    # pl.show()
    # pl.clf()
    # george_model = computegeorge(flare.flux, flare.time)
    #
    #
    #
    #

def cleanGeorgeModel (flux, time):
    flare = Flare(flux, time, 0, len(flux))

    sav_gol_model = sf(flare.flux, 101, 3)
    flare.flux -= sav_gol_model
    flare = remove_flares(flare)
    flare.flux += sav_gol_model

    return flare.flux# now compute the GP with flares removed

def removeRotation (flux, time):
    flare = Flare(flux, time, 0, len(flux))

    sav_gol_model = sf(flare.flux, 101, 3)
    flare.flux -= sav_gol_model
    return flare.flux
