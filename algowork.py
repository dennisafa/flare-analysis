import numpy as np
import scipy as scipy
import matplotlib.pyplot as pl
from lightkurve import KeplerTargetPixelFile

def flaredetect(flux):
    j = 0
    listFlare = []
    baseval = np.abs(np.average(flux) * 1.15)
    noise = noisecalc(flux)
    print(baseval)
    while j < len(flux)-1:
        if flux[j] > baseval:
            peak = flux[j]
            firstval = flux[j]
            if (flux[j] - flux[j + 1]) < 0:
                while j < len(flux) - 1 and flux[j] < flux[j + 1]:
                    peak = flux[j + 1]
                    j += 1
                else:
                    if peak - firstval > noise:
                        listFlare.append(peak)
                    j+=1
            else:
                if (flux[j] - flux[j - 1]) > 0 and (flux[j] - flux[j-1]) > noise:
                    listFlare.append(peak)
                j += 1
        else:
            j += 1
    return listFlare

def noisecalc (flux):
    j = 0
    jumps = []
    baseval = np.abs(np.average(flux) * 1.15)
    while j < len(flux) - 1:
        if flux[j] < baseval:
            firstval = flux[j]
            peakval = 0
            if (flux[j] - flux[j+1]) < 0:
                while j < len(flux) - 1 and flux[j] < flux[j + 1]:
                    peakval = flux[j+1]
                    j+=1
                else:
                    jumps.append(np.abs(peakval - firstval))
                    j+=1
            else:
                j+=1
        else:
            j+=1
    return np.average(jumps)



w359 = KeplerTargetPixelFile.from_archive(201885041, cadence='short')
lc359 = w359.to_lightcurve(aperture_mask=w359.pipeline_mask)
y = lc359.flux
x = lc359.time
y = (y/np.median(y)) - 1
y = [number/scipy.std(y) for number in y]

print(flaredetect(y))


pl.plot(x, y)
pl.show()