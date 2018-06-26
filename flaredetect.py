import numpy as np
def flaredetect(flux, sliceNum):
    j = 0
    listFlare = []
    baseval = np.abs(np.average(flux) * 1.5)
    noise = noisecalc(flux)
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
                if (flux[j] - flux[j - 1]) > 0 and (flux[j] - flux[j-1]) > noise**1/3:
                    listFlare.append(peak)
                j += 1
        else:
            j += 1
    print('Flare detect successful, number of flares: ' + str(len(listFlare)) + ' at slice number ' + str(sliceNum))
    return listFlare

def noisecalc (flux):
    return np.var(flux)

