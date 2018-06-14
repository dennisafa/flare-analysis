from appaloosa import aflare as ap
from lightkurve import KeplerLightCurveFile
import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import minimize

def findMaxFlux (flux):
    return max(flux)

def multiFlaresList(flux):
    listFlare = []
    hold = 0
    for i in range(len(flux)):
        if flux[i] > 0.02:
            #print(flux[i])
            while flux[i] > flux[i+1]:
                hold = flux[i]
                i += 1
            listFlare.append(hold)
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
    global fwhm

    def __init__(self, str, range1, range2, flarenum):

        sap = str.SAP_FLUX.remove_nans()
        self.flux = sap.flux[range1:range2]
        self.time = sap.time[range1:range2]
        self.flux = (self.flux/np.median(self.flux)) -1
        self.peakFlare = multiFlaresList(self.flux)[flarenum]  #
        #print(self.peakFlare)
        self.peakTime = findFluxTime(self.peakFlare, self.flux, self.time)
        pl.plot(self.time, self.flux)
        pl.show()


        #print(flareList)

    # def peaks(self):
    #
    #     multiFlares = multiFlaresList(self.flux)
    #     for x in multiFlares:
    #         self.peakFlare = x #
    #         self.peakTime = findFluxTime(self.peakFlare, self.flux, self.time)
    #         tof = time
    #         i = 0
    #         for x in flux:
    #             if (x == self.peakFlare):
    #                 break
    #             i += 1
    #         print(time[i])



    def reduceandplot(self):

        p = [self.peakTime, 0.05, self.peakFlare]
        print('First values of peakTime, peakFlare, and fwhm' + str(self.peakTime) + " " + str(self.peakFlare) + " "  + str(0.05))
        result = minimize(ng_ln_like, p, args=[self.time, self.flux])
        print(result)
        self.peakTime, fwhm, flarePeak = result.x
        print(self.peakTime, fwhm, flarePeak)
        pl.plot(self.time, ap.aflare1(self.time, tpeak=self.peakTime, fwhm=fwhm, ampl=flarePeak, upsample=True, uptime=10))
        pl.plot(self.time, self.flux)
        pl.show()


w359 = KeplerLightCurveFile.from_archive(201885041)
flare1 = strPlot(w359, 250, 280, 2)
#flare1.peaks()
flare1.reduceandplot()
