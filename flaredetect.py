import numpy as np
def flaredetectpeak(flux, std=None):
    global listFlare
    global firstval
    j = 0
    listFlare = []

    if std is None:
        baseval = get_std(flux) * 2 # 2 sigma deviation # 1 sigma for rotation
    else:
        baseval = std
    noise = get_noise(flux, baseval)
    while j < len(flux)-1:
        if flux[j] > baseval:
            peak = flux[j]
            firstval = flux[j]
            if (flux[j] - flux[j + 1]) < 0:
                while j < len(flux) - 1 and flux[j] < flux[j + 1]:
                    peak = flux[j + 1]
                    j += 1
                else:
                    temp = flux[j]
                    base = flux[j]
                    h = j
                    while h > 1 and flux[h] > flux[h - 1]:
                        base = flux[h - 1]  # peak will be point before it rises
                        h -= 1
                    else:
                        if temp - base > noise:
                            listFlare.append(peak)
                    j += 1
            else:
                temp = flux[j]
                base = flux[j]
                h = j
                while h > 1 and flux[h] > flux[h-1]:
                    base = flux[h - 1] # peak will be point before it rises
                    h -= 1
                else:
                    if temp - base > noise:
                        listFlare.append(peak)
                j += 1
        else:
            j += 1
    return listFlare


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

           if j + 1 < len(flux):
            listFlare.append(flux[j])
        else:
            j+=1

    return len(listFlare)

def get_std(flux):
    noise = np.std(flux)
    return noise

def get_noise(flux, baseval):
    list_flare = [fl for fl in flux if fl < baseval]
    noise_check = []
    for i, val in enumerate(list_flare):
        if i < len(list_flare) - 1:
            dist = np.absolute(list_flare[i] - list_flare[i+1])
            noise_check.append(dist)
    return np.average(noise_check)

def flaredetecttime(time, flux):
    global listFlare
    global firstval
    j = 0
    listFlare = []

    baseval = get_std(flux) # 2 sigma deviation
    noise = get_noise(flux, baseval)
    while j < len(flux)-1:
        if flux[j] > baseval:
            peak = flux[j]
            firstval = flux[j]
            if (flux[j] - flux[j + 1]) < 0:
                while j < len(flux) - 1 and flux[j] < flux[j + 1]:
                    peak = flux[j + 1]
                    j += 1
                else:
                    temp = flux[j]
                    base = flux[j]
                    h = j
                    while h > 1 and flux[h] > flux[h - 1]:
                        base = flux[h - 1]  # peak will be point before it rises
                        h -= 1
                    else:
                        if temp - base > noise:
                            listFlare.append(time[j])
                    j += 1
            else:
                temp = flux[j]
                base = flux[j]
                h = j
                while h > 1 and flux[h] > flux[h-1]:
                    base = flux[h - 1] # peak will be point before it rises
                    h -= 1
                else:
                    if temp - base > noise:
                        listFlare.append(time[j])
                j += 1
        else:
            j += 1
    return listFlare

