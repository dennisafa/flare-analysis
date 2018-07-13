import numpy as np
def flaredetect(flux):
    global listFlare
    global firstval
    j = 0
    listFlare = []
    baseval = np.abs(np.average(flux) * 4)
    print(baseval)
    noise = get_noise(flux)
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
                        print(flux)
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

def get_noise(flux):
    list_flare = [flare for flare in flux if flare < 0.5]
    noise = np.var(list_flare)
    print(noise)
    return noise