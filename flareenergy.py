import matplotlib.pyplot as pl
import numpy as np

def energy_calc(model, time, flux):
    i = 0
    flare_counter = 0
    duration = []

    while i < len(model):
        if model[i] > 0.01:
            start = i-1
            while model[i] - model[i+1] < 0 and i < len(model) - 1:
                i+=1

            peak = model[i]
            i+=1
            while model[i] > 0.005 and i < len(model) - 1:
                if model[i] - model[i+1] < 0 and model[i] < peak/2:
                    break
                else:
                    i+=1

            end = i
            flare_counter+=1
            print('Flare # ', flare_counter)
            #pl.plot(time[start:end], flux[start:end])
            #pl.plot(time[start:end], model[start:end])
            pl.show()
            pl.clf()
            ed = np.trapz(model[start:end], time[start:end] * 86400)
            duration.append(ed)
        i+=1
    # using the techniques davenport wrote to get energy/plot ED

    days = time[-1] - time[0]
    print(days)
    exptime = 1. / 24. / days

    totdur = float(len(time)) * exptime
    print(totdur)
    duration = np.sort(duration)[::-1]

    ddx = np.log10(duration)
    ddy = (np.arange(len(ddx)) + 1) / totdur
    pl.plot(ddx, ddy, 'o-', markersize=3)
    pl.yscale('log')
    pl.ylim(1e-2, 1e2)
    pl.xlabel('log Equivalent Duration (seconds)')
    pl.ylabel('Cumulative Flares per Day')
    pl.show()

    L_m6_most = 10 ** 30.61
    E_point = L_m6_most/3

    pl.plot(ddx + np.log10(E_point), ddy, 'o-', markersize=3)
    pl.yscale('log')
    pl.ylim(1e-2, 1e2)
    pl.xlabel('log Flare Energy (erg)')
    pl.ylabel('Cumulative Flares per Day')
    pl.show()


