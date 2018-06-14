import aflare as ap
from lightkurve import KeplerTargetPixelFile
import matplotlib.pyplot as plt
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
    time, y, nflares = data
    p = np.exp(p)
    model = np.zeros_like([time])
    p = p.reshape(3,nflares)
    for flare in range(nflares):
        model += ap.aflare1(time, tpeak=p[0,flare], fwhm=p[1,flare] , ampl=p[2,flare],
                            upsample=True, uptime=10)
    return model

def ng_ln_like (p, data):
    _, y, _ = data
    model = getModel(p, data)
    return np.sum((model - y) ** 2)


class strPlot:

    def __init__(self, star, range1, range2):
        sap = star.remove_nans()
        self.flux = sap.flux[range1:range2]
        self.time = sap.time[range1:range2]
        self.flux = (self.flux/np.median(self.flux)) - 1


    def guesspeaks(self):
        multiFlares = multiFlaresList(self.flux)
        # print(multiFlares)
        model = 0
        self.nflares = np.shape(multiFlares)[0]
        params = np.zeros([3, self.nflares])
        for i, flareVal in enumerate(multiFlares):
            self.peakFlare = flareVal
            self.peakTime = findFluxTime(self.peakFlare, self.flux, self.time)
            p = [self.peakTime, 0.004, self.peakFlare]
            params[:, i] = p
        return np.log(params)
            # data = [self.time, self.flux]
            # model += getModel(p, data)

#         print(model)

#         plt.plot(self.time, model)
#         plt.plot(self.time, self.flux)
#         plt.show()

    def fit(self, p):
        result = minimize(ng_ln_like, p, args=[self.time, self.flux, self.nflares], method='Powell')
        # self.peakTime, self.peakFlare  = result.x
        return result.x


# w359 = KeplerTargetPixelFile.from_archive(201885041, cadence='short')
# lc359 = w359.to_lightcurve(aperture_mask=w359.pipeline_mask)
# y = lc359.flux
# x = lc359.time
# flare1 = strPlot(lc359, 700, 850, 0)
# guessparams = flare1.guesspeaks()

# fitparams = flare1.fit(guessparams)


# def f(p, t, y):
#     p = np.exp(p)
#     flarePeak1, peakTime1, fwhm1, flarePeak2, peakTime2, fwhm2 = p
#     model1 = aflare.aflare1(t, tpeak=peakTime1, fwhm=fwhm1, ampl=flarePeak1)
#     model2 = aflare.aflare1(t, tpeak=peakTime2, fwhm=fwhm2, ampl=flarePeak2)
#     model = model1 + model2
#     return np.sum((y-model)**2)

# def get_model(p, t, y):
#     p = np.exp(p)
#     flarePeak1, peakTime1, fwhm1, flarePeak2, peakTime2, fwhm2 = p
#     model1 = aflare.aflare1(t, tpeak=peakTime1, fwhm=fwhm1, ampl=flarePeak1)
#     model2 = aflare.aflare1(t, tpeak=peakTime2, fwhm=fwhm2, ampl=flarePeak2)
#     model = model1 + model2
#     return model