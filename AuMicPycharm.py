import george
from george import kernels
import numpy as np
import matplotlib.pyplot as pl
import aflare as ap
import flaredetect as fd
import pandas as pd
from astropy.io import fits
from scipy.signal import savgol_filter as sf
from numpy import asarray
from flareenergy import energy_calc
from astropy.stats import LombScargle
from astropy.modeling.blackbody import blackbody_lambda



class Flare:
    flux = []
    time = []

    def __init__(self, x, y, r1, r2):

        self.flux = y[r1:r2]
        self.flux = self.flux[np.logical_not(np.isnan(self.flux))]
        self.flux = self.flux[np.logical_not(np.isinf(self.flux))]
        self.flux = self.flux.flatten()
        self.time = x[r1:r2]
        self.time = self.time[np.logical_not(np.isnan(self.time))]
        self.flux = self.flux.flatten()
        self.time = self.time[:len(self.flux)]


    def guesspeaks(self):  # gathers the peaks in the set of data, then returns a list of flare times, peaks, and fwhm
        self.detflares = fd.flaredetectpeak(self.flux)
        self.flarecount = fd.getlength()
        self.nflares = np.shape(self.detflares)[0]
        self.params = np.zeros([self.nflares, 3])
        for i, flareVal in enumerate(self.detflares):
            self.flarepeak = flareVal
            self.flaretime = self.findfluxtime(self.flarepeak, self.flux, self.time)
            p = [self.flaretime, 0.004, self.flarepeak]
            self.params[i, :] = p
        return np.log(self.params)

    def findfluxtime(self, flarepeak, flux, time):  # retrieves the time of the flare
        tof = time
        for i, flare in enumerate(flux):
            if flare == flarepeak:
                return tof[i]

    def getmodel(self, p, data):  # computes the model of the flares using appaloosa's aflare1 function
        time, y, nflares = data
        p = np.exp(p)
        model = np.zeros_like([time])
        p = np.reshape(p, (nflares, 3))
        for i in range(nflares):
            model += ap.aflare1(time, tpeak=p[i, 0], fwhm=p[i, 1], ampl=p[i, 2], upsample=False, uptime=10)
        return model


def lnprob(p):
    # Trivial uniform prior.
    if np.any((-100 > p[1:]) + (p[1:] > 100)):
        return -np.inf

    # Update the kernel and compute the lnlikelihood.
    gp.set_parameter_vector(p)
    return gp.log_likelihood(y, quiet=True)


'''George modeling'''


def computegeorge(flux, time):
    global gp
    global y

    y = flux
    x = time

    kernel = kernels.CosineKernel(log_period=np.log(3), axes=0) + kernels.ExpSquaredKernel(metric=0.5)
    gp = george.GP(kernel, mean=np.mean(y), fit_mean=True)
    gp.compute(x, y)
    print('Bounds', gp.get_parameter_bounds())
    print(gp.log_prior())
    print('Initial log likelihood', gp.log_likelihood(y))
    print('initial parameter vector', gp.get_parameter_vector())
    # res = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, method="L-BFGS-B")
    # gp.set_parameter_vector(res.x)
    # print('Final log likelihood', gp.log_likelihood(y))
    # print('final parameter vector', res.x)
    # print(res)

    '''Emcee sampling'''
    # nwalkers, ndim = 36, len(gp)
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
    #
    # # Initialize the walkers.
    # p0 = gp.get_parameter_vector() + 1e-4 * np.random.randn(nwalkers, ndim)
    #
    # print("Running burn-in")
    # p0, _, _ = sampler.run_mcmc(p0, 100)
    #
    # print("Running production chain")
    # sampler.run_mcmc(p0, 100)
    # print(sampler.flatchain[0][1])
    #
    # for i in range(ndim):
    #     pl.figure()
    #     pl.hist(sampler.flatchain[0, i], 100, color="k", histtype="step")
    #
    # pl.show()

    # for i in range(50):
    #     # Choose a random walker and step.
    #     w = np.random.randint(sampler.chain.shape[0])
    #     n = np.random.randint(sampler.chain.shape[1])
    #     gp.set_parameter_vector(sampler.chain[w, n])
    #     pl.plot(x, gp.sample_conditional(y, x))
    #
    #
    # pl.show()

    '''End emcee'''

    pred_mean, pred_var = gp.predict(y, x, return_var=True)

    pl.fill_between(time, pred_mean - np.sqrt(pred_var), pred_mean + np.sqrt(pred_var), color='k', alpha=0.4,
                    label='Predicted variance')

    # pl.plot(flare.time, pred_mean, color='Blue', label='Predicted mean')
    # pl.plot(flare.time, flare.flux, alpha=0.6, label='Raw flux')
    # pl.ylim(0.0, 1)
    # pl.ylabel("Relative Flux")
    # pl.xlabel("BJD")
    # pl.legend(loc='best')
    # pl.show()
    # pl.clf()
    #
    # # pred_mean, pred_var = gp.predict(y, x, return_var=True)
    #
    # pl.fill_between(time, pred_mean - np.sqrt(pred_var), pred_mean + np.sqrt(pred_var), color='k', alpha=0.4,
    #                 label='Predicted variance')
    #
    # pl.plot(flare.time, pred_mean, color='Blue', label='Predicted mean')
    # # pl.plot(flare.time, flare.flux, alpha=0.6, label='Raw flux')
    # pl.ylim(0.0, 1)
    # pl.ylabel("Relative Flux")
    # pl.xlabel("BJD")
    # pl.legend(loc='best')
    # pl.show()
    # pl.clf()

    # x = np.linspace(max(x), 3120, 3095)
    # mu, var = gp.predict(flare.flux, x, return_var=True)
    #
    # std = np.sqrt(var)
    #
    # pl.plot(time, y, color="Blue")
    # pl.fill_between(x, mu + std, mu - std, color="k", alpha=0.5)
    #
    # pl.xlim(min(time), 3180)
    #
    # pl.show()

    # print(result)

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
                bounds[i][j] = flux[i] ** 1 / 20
    return bounds


'''End George Modeling'''


def remove_flares(flare):
    while len(fd.flaredetectpeak(
            flare.flux)) > 0:  # while flares are still being detected, compute its model and subtract flares
        tempmodel = sub_flare_model(flare)
        pl.plot(flare.time, flare.flux)
        pl.plot(flare.time, tempmodel.flatten())
        pl.show()
        flare.flux = flare.flux - tempmodel.flatten()

        print("Flares subtracted!")

    return flare


def sub_flare_model(flare):
    guessparams = flare.guesspeaks()
    model = flare.getmodel(guessparams, [flare.time, flare.flux,
                                         flare.nflares])
    return model


def flatten(flare):
    # 113,008 points

    sf_model = sf(flare.flux, 501, 3)
    rotation = sf(sf_model, 501, 3)
    for i in range(10):
        rotation = sf(rotation, 501, 3)
    flat_flux = flare.flux - rotation
    flare.flux = flat_flux

    # smo = pd.rolling_median(flare.flux, 100, center=True)
    # smo2 = pd.rolling_median(flare.flux - smo, 2, center=True)

    pl.plot(flare.time, flare.flux)
    pl.show()
    remove_flares(flare)
    flare.flux += rotation

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
            new += list[i]
            j += 1

    return new_list


def planck(wav, T):
    h = 6.62 * 10**-27 # plancks constant, j/s
    c = 3.00 * 10**10 # speed of light, cm/s
    k = 1.38 * 10**-16 # boltzmanns constant
    e_power  = (c*h)/(k*wav*T)
    b_lambda = 2*(c**2)*h/(wav**5) * 1/((2.71828**e_power) - 1)
    return b_lambda



class Process:
    count = 0

    def subtract_flares(self, flare):
        model = sub_flare_model(flare)

        params = pl.gcf()
        params.set_size_inches(12, 6)

        # pl.plot(flare.time, model.flatten(), 'b--', label = 'Appaloosa model')
        # pl.plot(flare.time, flare.flux, alpha=0.3, color='Black', label = 'Flattened Flux')
        # pl.xlabel('Barycentric Julian Date')
        # pl.ylabel('Flux')
        # pl.legend(loc='best')
        # pl.show()
        # pl.clf()
        #
        # pl.plot(flare.time, model.flatten(), 'b--', label = 'Appaloosa model')
        # pl.xlabel('Barycentric Julian Date')
        # pl.ylabel('Flux')
        # pl.legend(loc='best')
        # pl.show()
        # pl.clf()

        self.count += flare.nflares

        return energy_calc(model.flatten(), flare.time, flare.flux)


tess_function = np.genfromtxt("tess-function.txt", dtype=float, usecols=(0, 1))
wav = tess_function[:, 0]
tess_R = tess_function[:, 1]
b_R_star = []
b_R_flare = []
test_function = []
temp = 3742 # kelvin
wav = wav/1e7


solar_l = 3.8 * 10**33 # erg
boltz = 5.67 * 10**-5
pi_over_temp = (np.pi)/(boltz * (temp**4))
prod_func = tess_R * planck(wav, temp)
conv_integral = np.trapz(prod_func, wav)
l_prime = solar_l * 0.0727 * (conv_integral) * pi_over_temp
print(l_prime)

# wavelengths = np.arange(1e-9, 3e-6, 1e-9)
# intensity5000 = planck(wavelengths, 5000.)
# pl.plot(wavelengths * 1e9, intensity5000)
# pl.show()


prod_star = []
prod_flare = []
# pl.plot(wav, tess_R)
# pl.show()
planck_file_star = open("planck_response_aumic.txt", 'w')
RB_product_star = open("RB_product.txt", 'w')

#integral_star = np.trapz(b_R_star * tess_R, wav) # b lambda * tess response over d lambda
# integral_flare = np.trapz(b_R_flare * tess_R, wav)
# print(integral_star)
# print(b_R_star)
# print(tess_R)
# print(integral_star)



# print(prod_star)
# pl.plot(wav, prod_star)
# pl.show()

wavelength = wav * 10 # convert to angstroms
flux_lam = blackbody_lambda(wavelength, temp) # erg/Angstrom/cm^2
radius = 0.763 * (6.957 * 10**8)
radius = radius * 100
conv_integral = np.trapz(flux_lam * tess_R, wavelength) # convolution integral over wavelength
tot_lum = np.asarray(np.pi * radius**2 * conv_integral) # total lum in the tess bandpass


test_wav = np.arange(0, 40000, 1)
test_lam = blackbody_lambda(test_wav, temp)
pl.plot(test_wav, test_lam, color="Green")
pl.show()
#pl.plot(wavelength, tess_R, color='Red')
pl.plot(wavelength,flux_lam * tess_R,  color='Black')
pl.plot(wavelength, flux_lam, color='Blue')
pl.xlabel("Wavelength, angstroms")
pl.ylabel("erg/s")
pl.show()


# equation is pi * Radius_star^2 * integral(planck_function * response_function d(wavelength))



# print(L_star)

fits_file = fits.open('/Users/dennisafa/nasa/tess/tess2018206045859-s0001-0000000441420236-0120-s_lc.fits')

y = fits_file[1].data.field("PDCSAP_FLUX")[:]
x = fits_file[1].data.field("TIME")[:]
# pl.plot(x,y)
# pl.show()
# pl.clf()


# flare = Flare(lc359, 0, len(lc359.flux))
flare = Flare(x, y, 0, len(y))
pl.plot(flare.time, flare.flux)
pl.show()
frequency, power = LombScargle(flare.time, flare.flux, center_data=True).autopower()
pl.plot(frequency, power)
period_days = 1/frequency
period_hours = period_days * 24
pl.plot(period_days, power, color='Black')
pl.xlim(0, 10)
pl.xlabel('Period days')
pl.ylabel('Power')
pl.show()

smo1 = pd.rolling_median(flare.flux, 100, center=True)
smo2 = pd.rolling_median(flare.flux - smo1, 2, center=True)
y = np.isfinite(smo2)
flare.flux = ((flare.flux[y] - smo1[y]) / np.median(flare.flux))
flare.time = flare.time[:len(flare.flux)]
pl.plot(flare.time, flare.flux)
pl.show()
pl.clf()


# pl.plot(flare.time, flare.flux)
# pl.show()
# pl.clf()
# smoothedFilter = gausFilter.gaussian_filter1d(flare.flux, 70)
# pl.plot(flare.time,smoothedFilter)
# pl.plot(flare.time, flare.flux)
# pl.show()

# flare.flux = flare.flux - smoothedFilter
pl.plot(flare.time, flare.flux)
pl.show()
pl.clf()
# pl.plot(flare.time,flare.flux)
# pl.show()

# get = Process()
# ed = get.subtract_flares(flare)


file = np.genfromtxt("aumicflare_times.txt", dtype=float, usecols=(0, 1))
start = file[:, 0]
end = file[:, 1]
# print(end)
tracker = 0
ed = []
loc_mean = []
amp = []
l_flare_p = []
delta_f = []
a_flare_prime = []
final_l_flare = []
# print("Time per interval", flare.time[1] - flare.time[0])

for n, i in enumerate(flare.time):
    if tracker < len(start):
        if flare.time[n] > start[tracker]:
            temp = n
            peak = flare.flux[n]
            while flare.time[temp] < end[tracker] and temp < len(flare.time):
                if flare.flux[temp] > peak:
                    peak = flare.flux[temp]
                temp += 1
                loc_mean.append(flare.flux[temp])

            a_flare = np.trapz(flare.flux[n:temp], flare.time[n:temp] * 86400)
            # l_prime_flare = a_flare * integral_flare
            # l_flare_p.append(l_prime_flare)
            # delta_f = l_prime_flare / L_star
            # a_flare_t = (delta_f*np.pi*(radius**2)) * (integral_star/integral_flare)
            # a_flare_prime.append(a_flare_t)
            # final_l_flare.append(5.670367*10**-8 * 9000**4 * a_flare_t) # ergs


            ed.append(a_flare)  # 2 min cadence
            # print(ed)

            local = np.sum(loc_mean) / len(flare.flux[n:temp])
            # loc_mean.clear()
            # print("Flare amp =", (peak - local) / local)
            amp.append(peak)

            # print(len(flare.flux[n:temp]))
            # print("Loc mean = ", np.cumsum(loc_mean) / len(flare.flux[n:temp]))
            # print(amp)
            # pl.plot(flare.time[n:temp], flare.flux[n:temp], color = 'Black', linestyle='-')
            # pl.legend(loc='best')
            # pl.ylabel('Normalized flux')
            # pl.show()
            # pl.clf()
            # print("Start = : ", flare.time[n])
            # print("End = : ", flare.time[temp])
            # print(temp)
            n = temp
            tracker += 1

print("L_flare' = ", l_flare_p)
print("Delta f/f =", delta_f)
print("A_flare(t)", a_flare_prime)
print("L_flare", final_l_flare)

exptime = 2. / 24. / 27.
# print(flare.time[len(flare.time)-1] - flare.time[-1])

totdur = float(len(flare.time)) * exptime
duration = np.sort(ed)[::-1]
print("Mean flares,", np.mean(amp))

ddx = np.log10(duration)
ddy = (np.arange(len(ddx))) / totdur
pl.plot(ddx, ddy, 'o--', markersize=2, alpha=0.5)
pl.yscale('log')
pl.xlabel('log Equivalent Duration (seconds)')
pl.ylabel('Cumulative Flares per Day')
pl.show()
pl.clf()

E_point = np.log10(l_prime)

print("Rough energy calc", E_point)

pl.plot(ddx + E_point, ddy, 'o', color='Black', markersize=3, alpha=0.7)
pl.title("Au Mic Flare Frequency Distribution")
pl.yscale('log')
#pl.ylim(1e-20, 1e3)
pl.xlabel('log Flare Energy (erg)')
pl.ylabel('Cumulative Flares per Day')
pl.show()
pl.clf()
print(ddy[-1], ddx[-1]+E_point)
print(ddx+E_point)

pl.scatter(ddx + E_point, amp)
pl.show()

#
#
#
# # a_flare = ed[0] *
#
#
# # g = computegeorge(flare.flux, flare.time)
# # pl.plot(flare.time, g, label='George model')
# # pl.show()
#
# # flare.flux -= g # or sav gol model
#
# # model = sub_flare_model(flare)
# # pl.plot(model.flatten())
# # pl.show()
# # count = flare.nflares
# # print(count)
#
# # flare = remove_flares(flare)
# # pl.plot(flare.time, flare.flux, label='Rotation and flare subbed flux')
# # pl.xlabel("BJD")
# # pl.ylabel("Relative Flux")
# # pl.legend(loc='best')
# # pl.show()
