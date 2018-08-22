from appaloosa import aflare as ap
from lightkurve import KeplerTargetPixelFile, KeplerLightCurveFile
import matplotlib.pyplot as pl
import wolf359.wolfapp as master
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
import pandas as pd
from astropy.stats import LombScargle




#2457949.4982

'''Rotation modeling'''
# file = np.genfromtxt("211828663.txt", dtype=float, usecols=(0, 1), delimiter=',')
# y = file[:, 1]
# x = file[:, 0]
# print("Creating model...")
# flare = strPlot(y, x, 0, len(y))
# print(len(y))
# g = computegeorge(flare.flux, flare.time)
#get = FinalModelGeorge()
#get.subtract_flares(flare)
# flat_flux = get.flat_flux
# clean_flux = get.clean_flux
# orig_flux = get.orig_flux
# period_list = get.period
# period = detect_period(period_list, flare.time)
# flare_detect(period, flat_flux, clean_flux, flat_flux, flare.time)
'''End rotation modeling'''



'''Flare analysis'''
# tpf = KeplerTargetPixelFile.from_archive('211877371', cadence='short')
#
# tpf.plot(frame=0)
# print(tpf.pipeline_mask)
# pl.show()
# pl.clf()
#
# aper = np.zeros(tpf.shape[1:])
# aper[0:3, 4:6] = 1
# tpf.plot(aperture_mask=aper)
# pl.show()
# lc = tpf.to_lightcurve(aperture_mask=aper.astype(bool)).flatten()
# lc = lc.remove_nans().remove_outliers()
# flux = lc.flux
# time = lc.time
# flux = pd.rolling_median(flux, 400, center=True)
# pl.plot(flux)
# pl.show()

'''End flare analysis'''