import matplotlib.pyplot as pl
import numpy as np

def energy_calc(model, time, flux):
    i = 0
    flare_counter = 0
    duration = []

    while i < len(model) - 1:
        if model[i] > 0.005 and i < len(model) - 1:
            start = i-2
            while model[i] - model[i+1] < 0 and i < len(model) - 2:
                i+=1

            peak = model[i]
            i+=1
            while model[i] > 0.005 and i < len(model) - 1:
                if model[i] - model[i+1] < 0 and model[i] < peak/3:
                    break
                else:
                    i+=1

            end = i-2
            flare_counter+=1
            ed = np.trapz(flux[start:end], time[start:end] * 86400)
            duration.append(ed)
        i+=1
    # using the techniques davenport wrote to get energy/plot ED
    return duration