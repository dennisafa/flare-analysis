from appaloosa import aflare as ap
from lightkurve import KeplerTargetPixelFile
import matplotlib.pyplot as pl
import numpy as np
import wolf359.flaredetect as fd
import scipy as scipy
from scipy.optimize import minimize
import time
import os
from numpy import asarray

class strPlot:

    def findMaxFlux(self, flux):
        return max(flux)

    def findFluxTime(self, peakFlux, flux, time):
        tof = time
        i = 0
        for x in flux:
            if (x == peakFlux):
                return tof[i]
            i += 1

    def getModel(self, p, data):
        time, y, nflares = data
        p = np.exp(p)
        model = np.zeros_like([time])
        p = np.reshape(p, (nflares, 3))
        for i in range(nflares):
            model += ap.aflare1(time, tpeak=p[i, 0], fwhm=p[i, 1], ampl=p[i, 2], upsample=True, uptime=10)
        return model

    def ng_ln_like(self, p, data):
        _, y, _ = data
        model = self.getModel(p, data)
        return np.sum((model - y) ** 2)

    def __init__(self, star, range1, range2):
        sap = star.remove_nans()
        self.flux = sap.flux[range1:range2]
        self.time = sap.time[range1:range2]
        self.flux = (self.flux/np.median(self.flux)) - 1
        self.flux = [number / scipy.std(self.flux) for number in self.flux]


    def guesspeaks(self, sliceNum):
        self.detflares = fd.flaredetect(self.flux, sliceNum)
        self.nflares = np.shape(self.detflares)[0] # used to get the shape of the array, number of flares is the shape
        self.params = np.zeros([self.nflares, 3])
        for i, flareVal in enumerate(self.detflares):
            self.peakFlare = flareVal
            self.peakTime = self.findFluxTime(self.peakFlare, self.flux, self.time)
            p = [self.peakTime, 0.004, self.peakFlare]
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


w359 = KeplerTargetPixelFile.from_archive(201885041, cadence='short')
lc359 = w359.to_lightcurve(aperture_mask=w359.pipeline_mask)

i = 0
x=0
starttime = time.time()
while i < len(lc359.flux) - 300:
    flare = strPlot(lc359, i, i+300)
    guessparams = flare.guesspeaks(x)
    bounds = flare.setbounds(guessparams)
    if len(flare.detflares) == 0:
        i+=300
        continue
    fitparams = flare.fit(guessparams, bounds)
    model = flare.getModel(fitparams, [flare.time, flare.flux, flare.nflares])

    for it,flux in enumerate(flare.detflares):
            pl.plot(flare.params[it,0], flux, marker='x', markersize=4, color="black")
    pl.plot(flare.time, model.flatten(), '--r')
    pl.plot(flare.time, flare.flux, color = 'Grey', lw = 0.5)
    pl.xlabel('Time - BYJD')
    pl.ylabel('Flux - Normalized to 0')
    savepath = os.path.join('/Users/Dennis/Desktop/plotswolf/test', 'wolf' + str(x) + '.png')
    pl.savefig(savepath)
    pl.clf()
    i+=300
    x+=1
endtime = time.time()
print('Time taken = ' + str(endtime - starttime))