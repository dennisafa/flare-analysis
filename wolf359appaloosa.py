from appaloosa import aflare as ap
from lightkurve import KeplerLightCurveFile
import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import minimize

def findMaxFlux (flux):
    return max(flux)

def multiFlaresList(flux):
    listFlare = []
    for val in flux:
        if val > 0.02:
            listFlare.append(val)
    return listFlare


def findFluxTime(peakFlux, flux, time):
    tof = time
    i = 0
    for x in flux:
        if (x == peakFlux):
            return tof[i]
        i += 1

def ng_ln_like (p, data):
    time, y = data
    peakTime, fwhm, peakFlare = p
    model = ap.aflare1(time, tpeak=peakTime, fwhm=fwhm , ampl=peakFlare, upsample=True, uptime=10)
    return np.sum((model - y) ** 2)

class strPlot:

    global peakTime
    global peakFlare
    global time
    global flux

    def __init__(self, str, range1, range2, flarenum):

        sap = str.SAP_FLUX.remove_nans()
        self.flux = sap.flux[range1:range2]
        self.time = sap.time[range1:range2]
        self.flux = (self.flux/np.median(self.flux)) -1
        self.peakFlare = multiFlaresList(self.flux)[flarenum]
        self.peakTime = findFluxTime(self.peakFlare, self.flux, self.time)
        flareList = multiFlaresList(self.flux)
        print(flareList)



    def reduceandplot(self):

        p = [self.peakTime, self.peakFlare, 0.05]
        result = minimize(ng_ln_like, p, args=[self.time, self.flux])
        self.peakTime, fwhm, flarePeak = result.x
        pl.plot(self.time, ap.aflare1(self.time, tpeak=self.peakTime, fwhm=fwhm, ampl=self.peakFlare, upsample=True, uptime=10))
        pl.plot(self.time, self.flux)
        #pl.show()


w359 = KeplerLightCurveFile.from_archive(201885041)
flare1 = strPlot(w359, 250, 280, 2)
flare1.reduceandplot()
