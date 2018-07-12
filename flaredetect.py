import numpy as np
def flaredetect(flux, slicenum=0):
    global listFlare
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
                    if peak - firstval > noise * 2:
                        listFlare.append(peak)
                    j+=1
            else:
                if (flux[j] - flux[j - 1]) > 0 and (flux[j] - flux[j-1]) > noise * 2:
                    listFlare.append(peak)
                j += 1
        else:
            j += 1
    #print('Flare detect successful, number of flares: ' + str(len(listFlare)) + ' at slice number ' + str(slicenum))
    return listFlare

def noisecalc (flux):
    return np.var(flux)

def getlength():
    return len(listFlare)

def model_peaks(flux):
    flux = np.asarray(flux)
    listFlare = []
    j = 0
    while j < len(flux) - 1:
        if (flux[j] - flux[j + 1]) < 0:
           while j < len(flux) - 1 and flux[j] < flux[j + 1]:
               j += 1
           listFlare.append(flux[j])
        else:
            j+=1
    return len(listFlare)

