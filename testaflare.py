from appaloosa import aflare as ap
from lightkurve import KeplerTargetPixelFile
import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import minimize

def findMaxFlux (flux):
    return max(flux)

def multiFlaresList(list):
    j = 0
    listFlare = []
    while j < len(list):
        if list[j] > 0.005:
            tempVar = list[j]
            if (list[j] - list[j + 1]) < 0:
                while j < len(list) - 2 and list[j] < list[j + 1]:
                    tempVar = list[j + 1]
                    j += 1
                else:
                    if j == len(list) - 2:
                        break
                    listFlare.append(tempVar)
            else:
                if list[j] - list[j-1] > 0:
                    listFlare.append(tempVar)
                j+=1
        else:
            j+=1
    return listFlare


def findFluxTime(peakFlux, flux, time):
    tof = time
    i = 0
    for x in flux:
        if (x == peakFlux):
            return tof[i]
        i += 1

def getModel(p, data):
    time, y = data
    peakTime, fwhm, peakFlare = p
    model = ap.aflare1(time, tpeak=peakTime, fwhm=fwhm, ampl=peakFlare, upsample=True, uptime=10)
    return model

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
    global fwhm
    global plot

    def __init__(self, str, range1, range2, flarenum):
        sap = str.remove_nans()
        self.flux = sap.flux[range1:range2]
        self.time = sap.time[range1:range2]
        self.flux = (self.flux/np.median(self.flux)) - 1


    def peaks(self):
        multiFlares = multiFlaresList(self.flux)
        print(multiFlares)
        model = 0
        for flareVal in multiFlares:
            self.peakFlare = flareVal
            self.peakTime = findFluxTime(self.peakFlare, self.flux, self.time)
            p = [self.peakTime, 0.004, self.peakFlare]
            data = [self.time, self.flux]
            model += getModel(p, data)

        print(model)

        pl.plot(self.time, model)
        pl.plot(self.time, self.flux)
        pl.show()

    def minimize(self):
        p = [self.peakTime, 0.004, self.peakFlare]
        result = minimize(ng_ln_like, p, args=[self.time, self.flux])
        self.peakTime, self.peakFlare  = result.x
        return result


w359 = KeplerTargetPixelFile.from_archive(201885041, cadence='short')
lc359 = w359.to_lightcurve(aperture_mask=w359.pipeline_mask)
y = lc359.flux
x = lc359.time
flare1 = strPlot(lc359, 700, 850, 0)
flare1.peaks()
