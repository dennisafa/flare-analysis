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


class strPlot:

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
            p = [self.flaretime, 0.04, self.flarepeak]
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

    kernel = kernels.ExpSine2Kernel(gamma=5, log_period=0.36) * kernels.Matern52Kernel(metric=0.5)  # a * cos(2pi/T * (t-3082) ) *
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


# def sub_flares(flare, period, phase, range1, range2): # loops through a subset of flux values, detects flares, then subtracts and performs again
#     global flare_orig
#     flare_orig = flare.flux
#     finalmodel = 0
#     i = 0
#     while len(fd.flaredetect(flare.flux)) > 0:# while flares are still being detected, compute its model and subtract flares
#         tempmodel = sub_flare_model(flare)
#         finalmodel += tempmodel
#         flare.flux = flare.flux-tempmodel.flatten()
#         i+=1
#
#     if i > 0:
#
#         final_count = fd.model_peaks(list(finalmodel.flatten()))
#         if final_count > 0:
#             print("{} flares detected in period {}".format(final_count, phase))
#             pl.plot(flare.time, flare_orig, color='Blue', label='Original flux')
#             #pl.plot(flare.time, period[range1:range2], color='Grey', label='Period model')
#             pl.plot(flare.time, finalmodel.flatten(), color="Black", linestyle='--', label='Flare model')
#             pl.legend(loc='upper left')
#             pl.xlabel('Time - BJD')
#             pl.ylabel('Flux - Normalized 0')
#             pl.show()
#
#         #print(fd.getlength())
#     else:
#         print("No flares detected in period {}".format(phase))



def sub_flare_model(flare):
    guessparams = flare.guesspeaks()  # returns the parameters of where and when each flare occured
    #notempty = checkzero(flare.detflares)  # checks for flares to exit in the data set before modeling

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


def run():
    # stars to test george on: 206208968, 201205469, 201885041
    # testing 206208968 - 4:41 pm


    fits = KeplerTargetPixelFile.from_archive("206208968")
    lc359 = fits.to_lightcurve(aperture_mask=fits.pipeline_mask)

    print("Creating model...")
    flare = strPlot(lc359, 0, len(lc359.flux))

    get = FinalModelGeorge() # create an object to get model and smooth period curve from
    get.create_final_model(flare)
    final_plot = get.final_plot
    period_list = get.period
    period = detect_period(period_list, flare.time) # checks where the periods derivative changes sign, returns array of points
    flare.flux = final_plot

    appaloosa_model = np.ravel(sub_flare_model(flare)) # final model of the flares
    print("Period",period)
    print("Peaks model",appaloosa_model)

    avg_flare_count = []
    avg_period = []
    count = 1
    periodcount = []
    timeperiodcount = []

    print("Finished")

    #file = open('/Users/Dennis/Desktop/wolf359Final/plots10/FlareRate10%', 'a')
    for i, p in enumerate(period):
        if i < len(period) - 1:
             # check if the period is greater than 1.5 days
            flare_count = fd.model_peaks(appaloosa_model[period[i]:period[i+1]])
            avg_flare_count.append(flare_count)
            avg_period.append(flare.time[period[i+1]] - flare.time[period[i]])
            #file.write("period {} = {} from {:0.3f} to {:0.3f} \n".format(count,flare_count, flare.time[period[i]], flare.time[period[i+1]]))
            count += 1
            timeperiodcount.append(flare.time[period[i+1]] - flare.time[period[i]])
            periodcount.append(count)
    est_avg_flare = np.sum(avg_flare_count) / count
    est_avf_period = np.average(avg_period)

    pl.plot(periodcount, avg_flare_count, linestyle = '--', color='Black', marker='o')
    pl.xlabel("Period")
    pl.ylabel("# of flares")
    pl.show()
    pl.clf()
    pl.plot(periodcount, timeperiodcount, color = 'Blue', linestyle='--')
    pl.xlabel("Period")
    pl.ylabel("Time (BJD)")
    pl.show()
    pl.clf()
   # file.write("Average # flares = {}\nApproximate period = {}".format(est_avg_flare, est_avf_period))
    #file.close()


class FinalModelGeorge:
    period = []
    final_plot = []

    def create_final_model(self, flare):
        # 113,008 points
        flux_temp = flare.flux

        george_model = computegeorge(flux_temp, flare.time)# create an initial model
        sub_model = flux_temp - george_model  # subtract the george model from the raw data
        george_model2 = computegeorge(sub_model, flare.time)  # create a model of the data with george model subbed

        test_model = flux_temp - (george_model2 + george_model)
        sub_model2 = sub_model - george_model2
        george_model3 = computegeorge(sub_model2, flare.time)


        sub_model3 = sub_model2 - george_model3
        george_model4 = computegeorge(sub_model3, flare.time)

        clean_model = george_model2 + george_model + george_model3 + george_model4
        self.final_plot = flare.flux - clean_model# plot the new model
        self.period = sf(george_model, 31, 3)


        #flare, george_model, sub_model, george_model2, final_plot
        #plot_final_model(flare, george_model, sub_model, george_model2, self.final_plot)


        pl.ylabel("Normalized Flux")
        # second_reduce.set_xticklabels(flare.time.astype(int))
        pl.xlabel("Days")
        pl.plot(flare.time[:250], self.final_plot[:250], color="Black", label='Flux')
        pl.axhline(y=fd.get_std(flare.flux), color = 'Blue', label='2σ', linestyle = '--')
        pl.legend(loc='upper right')

        pl.show()
        print("Model complete")


class FinalModelSF:
    period = []
    final_plot = []

    def create_final_model(self, flare):
        # 113,008 points
        #flare.flux = [flare for flare in flare.flux if flare > -0.5]
        sf_model = sf(flare.flux, 501, 3)
        self.period = sf(sf_model, 501, 3)
        for i in range(1000):
            self.period = sf(self.period, 501, 3)
        self.final_plot = flare.flux - self.period


