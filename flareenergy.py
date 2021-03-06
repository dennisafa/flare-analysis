import matplotlib.pyplot as pl
import numpy as np

def energy_calc(model, time, flux):
    i = 0
    flare_counter = 0
    duration = []

    while i < len(model) - 1:
        if model[i] > 0.001 and i < len(model) - 1:
            if (i-3 > 0):
                start = i-3
            else:
                start = i
            while model[i] - model[i+1] < 0 and i < len(model) - 2:
                i+=1

            peak = model[i]
            while model[i] > 0.001 and i < len(model) - 2:
                i+=1

            end = i

            flare_counter+=1
            #model[start:end] += np.median(flux)
            ed = np.trapz(flux[start:end], time[start:end] * 86400)
            duration.append(ed)
            i+=1
            pl.plot(time[start-100:end+100], model[start-100:end+100])
            pl.plot(time[start-100:end+100], flux[start-100:end+100])
            pl.show()
            pl.clf()
        i+=1

   # model += np.median(flux)
   #  pl.plot(time, model)
   #  pl.plot(time, flux)
   #  pl.show()
   #  pl.clf()
    # using the techniques davenport wrote to get energy/plot ED
    return duration