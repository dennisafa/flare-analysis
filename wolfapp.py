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
            model += ap.aflare1(time, tpeak=p[i, 0], fwhm=p[i, 1], ampl=p[i, 2], upsample=True, uptime=10)
        return model

    def ng_ln_like(self, p, data):
        _, y, _ = data
        model = self.getmodel(p, data)
        return np.sum((model - y) ** 2)

    def __init__(self, flux, time, range1, range2, model=None): # cleans the list of flux, normalizes to 0
        self.flux = flux[range1:range2]
        self.time = time[range1:range2]
        #self.flux = (self.flux/np.median(self.flux)) - 1
        #self.flux = [number / scipy.std(self.flux) for number in self.flux]
        self.time = self.time[np.isfinite(self.flux)]
        #self.flux = asarray(self.flux)
        self.flux = self.flux[np.isfinite(self.flux)]


    def guesspeaks(self): # gathers the peaks in the set of data, then returns a list of flare times, peaks, and fwhm
        self.detflares = fd.flaredetect(self.flux)
        self.flarecount = fd.getlength()
        self.nflares = np.shape(self.detflares)[0]
        self.params = np.zeros([self.nflares, 3])
        for i, flareVal in enumerate(self.detflares):
            self.flarepeak = flareVal
            self.flaretime = self.findfluxtime(self.flarepeak, self.flux, self.time)
            p = [self.flaretime, 0.0005, self.flarepeak]
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

def flare_detect(period, flux, time, clean_flux):
    det_flares = fd.flaredetect(flux)
    flare_times = findflaretime(det_flares, flux, time)
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
        i += 1

        energyFlares = np.absolute(integrate.cumtrapz(flux[period[i]:period[i+1]], time[period[i]:period[i+1]], initial=0))
        print('energy=', energyFlares)
        pl.plot(energyFlares)
        pl.xlabel('Energy with flares')
        pl.show()
        pl.clf()

        energyNoFlares = np.absolute(integrate.cumtrapz(clean_flux[period[i]:period[i+1]], time[period[i]:period[i+1]], initial=0))
        pl.plot(time[period[i]:period[i+1]], clean_flux[period[i]:period[i+1]])
        pl.show()
        pl.clf()

        pl.plot(energyNoFlares)
        pl.xlabel('Energy without flares')
        pl.show()
        pl.clf()

        pl.plot(time[period[i]:period[i+1]], flux[period[i]:period[i+1]])
        pl.show()
        pl.clf()
        print('Without flares', np.sum(energyNoFlares))
        print('With flares', np.sum(energyFlares))

        print("period {} had {} flares".format(i, count))
        total += count
    print("avg period is {} days".format(np.average(avg_period)))
    print ("total flare events {}".format(total))

def findflaretime(flarepeak, flux, time): # retrieves the time of the flare
    flare_times = []
    flux = flux.tolist()
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
                bounds[i][j] = flux[i]
            else:
                bounds[i][j] = flux[i] ** 1/6
    return bounds

def neg_ln_like(p):
    gp.set_parameter_vector(p)
    return gp.log_likelihood(f)

def grad_neg_ln_like(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(f)

def remove_flares(flare):

    while len(fd.flaredetect(flare.flux)) > 0:# while flares are still being detected, compute its model and subtract flares
        tempmodel = sub_flare_model(flare)
        #sv_gol = sf(flare.flux, 501, 3)
        #tempmodel += sv_gol
        flare.flux = flare.flux-tempmodel.flatten()
        print("Flares subtracted!")

    return flare

def computegeorge (flux, time, sigma):
    global gp
    global f

    bounds = setbounds(flux)
    print(len(bounds))

    f = flux

   # kernel = kernels.CosineKernel(log_period=np.log(2.5)) * kernels.ExpSquaredKernel(metric=0.5)   # a * cos(2pi/T * (t-3082) ) *

    kernel = np.var(flux) * kernels.CosineKernel(log_period=0.36) * kernels.ExpSquaredKernel(metric=0.5)
    gp = george.GP(kernel)
    gp.compute(time, flux)
    #result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    #gp.set_parameter_vector(result.x)
    pred_mean, pred_var = gp.predict(flux, time, return_var=True)

    #print(result)
    return pred_mean


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

def sub_flare_model(flare):
    guessparams = flare.guesspeaks()  # returns the parameters of where and when each flare occurred

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
    final_plot = []
    rotation = []
    clean_flux = []


    def subtract_flares(self, flare):

        #flare.flux = flare.flux / np.mean(abs(flare.flux))
        flux_orig = flare.flux

        sf_model = sf(flare.flux, 501, 3)
        self.rotation = sf(sf_model, 501, 3)
        for i in range(10):
            self.rotation = sf(self.rotation, 501, 3)
        self.final_plot = flare.flux - self.rotation
        flare.flux = self.final_plot

        flare = remove_flares(flare)
        self.clean_flux = flare.flux
        flare.flux += self.rotation
        pl.plot(flare.time, flare.flux)
        pl.show()

        self.period = sf(flare.flux, 501, 3)
        for i in range(2000):
            self.period = sf(self.period, 501, 3)

        pl.plot(flare.time, self.period, linewidth = 2)
        #pl.plot(flare.time, flare.flux)
        pl.show()


        #self.create_final_model(flare)

        #self.iter_model(flare)


    def iter_model(self, flare):
        # 113,008 points

        george_model = computegeorge(flare.flux, flare.time, fd.get_std(flare.flux))  # create an initial model
        self.final_plot = flare.flux - george_model  # plot the new model

        pl.plot(flare.time, flare.flux)
        pl.plot(flare.time, george_model)
        pl.show()

        flare.flux = self.final_plot

        # sub_model = flux_temp - george_model  # subtract the george model from the raw data
        # george_model2 = computegeorge(sub_model, flare.time)  # create a model of the data with george model subbed
        # sub_model2 = sub_model - george_model2
        # george_model3 = computegeorge(sub_model2, flare.time)
        # sub_model3 = sub_model2 - george_model3
        # george_model4 = computegeorge(sub_model3, flare.time)
        # clean_model = george_model2 + george_model + george_model3 + george_model4

        print("Model complete")



    # stars to test george on: 206208968, 201205469, 201885041
    # testing 206208968 - 4:41 pm
fits_file = fits.open('/Users/Dennis/Desktop/newwolfdata/files/ktwo201885041_01_kasoc-ts_slc_v1.fits')
flux = fits_file[1].data['flux']
time = fits_file[1].data['time']



#fits = KeplerTargetPixelFile('/Users/Dennis/Desktop/newwolfdata/files/ktwo201885041_02_kasoc-ts_llc_v1.fits')

print("Creating model...")
flare = strPlot(flux, time, 0, 20000)

get = FinalModelGeorge()
get.subtract_flares(flare)
final_plot = get.final_plot
period_list = get.period
period = detect_period(period_list, flare.time)
flare_detect(period, final_plot, flare.time, get.clean_flux)
flare.flux = final_plot

print("Finished")

    #flare.flare_detect(period)




