from lightkurve import KeplerTargetPixelFile
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import wolf359.wolfapp as master


#2457949.4982

'''Rotation modeling'''
# file = np.genfromtxt("211828663.txt", dtype=float, usecols=(0, 1), delimiter=',')
# y = file[:, 1]
# x = file[:, 0]
# print("Creating model...")
# flare = master.strPlot(y, x, 0, len(y))
# get = master.FinalModelGeorge()
# get.subtract_flares(flare)
# g = computegeorge(flare.flux, flare.time)
# flat_flux = get.flat_flux
# clean_flux = get.clean_flux
# orig_flux = get.orig_flux
# period_list = get.period

'''End rotation modeling'''



'''Flare analysis'''
tpf = KeplerTargetPixelFile.from_archive('211817361', cadence='short')

tpf.plot(frame=0)
print(tpf.pipeline_mask)
pl.show()
pl.clf()
aper = np.zeros(tpf.shape[1:])
aper[1:6, 2:7] = 1
tpf.plot(aperture_mask=aper)
pl.show()
lc = tpf.to_lightcurve(aperture_mask=aper.astype(bool))
lc = lc.remove_nans().remove_outliers()
flux = lc.flux
flux = pd.rolling_median(flux, 1001, center=True)
pl.plot(flux)
pl.show()


time = lc.time

pl.plot(time, flux)
pl.show()
# '''End flare analysis'''
