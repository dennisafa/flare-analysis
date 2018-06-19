from appaloosa import aflare as ap
from lightkurve import KeplerTargetPixelFile
import matplotlib.pyplot as pl
import numpy as np
import wolf359.flaredetect as fd
import scipy as scipy
from scipy.optimize import minimize


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


    def guesspeaks(self):
        multiFlares = fd.flaredetect(self.flux)
        self.nflares = np.shape(multiFlares)[0] # used to get the shape of the array, number of flares is the shape
        params = np.zeros([self.nflares, 3])
        for i, flareVal in enumerate(multiFlares):
            self.peakFlare = flareVal
            self.peakTime = self.findFluxTime(self.peakFlare, self.flux, self.time)
            p = [self.peakTime, 0.004, self.peakFlare]
            params[i, :] = p
        return np.log(params)


    def fit(self, p):
        result = minimize(self.ng_ln_like, p, args=[self.time, self.flux, self.nflares], method='Powell')
        return result.x


w359 = KeplerTargetPixelFile.from_archive(201885041, cadence='short')
lc359 = w359.to_lightcurve(aperture_mask=w359.pipeline_mask)
y = lc359.flux
x = lc359.time
flare1 = strPlot(lc359, 1800, 2200)

guessparams = flare1.guesspeaks() # getting the model parameters

fitparams = flare1.fit(guessparams) # now we fit the parameters through a minimization process
model = flare1.getModel(fitparams, [flare1.time,flare1.flux, flare1.nflares])
pl.plot(flare1.time, model.flatten())
pl.plot(flare1.time, flare1.flux)
pl.show()
