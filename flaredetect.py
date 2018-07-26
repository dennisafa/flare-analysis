import numpy as np
def flaredetect(flux):
    global listFlare
    global firstval
    j = 0
    listFlare = []
    baseval = get_std(flux) * 2 # 2 sigma deviation
    noise = get_noise(flux)
    print("Noise is",noise)
    print("Baseval is",baseval)
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
                        firstval = flux[j]
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
                        listFlare.append(temp)
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

           if j + 1 < len(flux):
            listFlare.append(flux[j])
        else:
            j+=1

    return len(listFlare)

def get_std(flux):
    list_flare = [flare for flare in flux if flare < 1] # this also has to change depending on how noisy the data is
    noise = np.std(list_flare)
    return noise

def get_noise(flux):
    list_flare = [flare for flare in flux if flare < 1]
    noise_check = []
    for i, val in enumerate(list_flare):
        if i < len(list_flare) - 1:
            dist = list_flare[i] - list_flare[i+1]
            if dist > 0:
                noise_check.append(dist)
            else:
                noise_check.append(list_flare[i+1] - list_flare[i])
    return np.average(noise_check)
