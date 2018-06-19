import numpy as np
def flaredetect(flux):
    j = 0
    listFlare = []
    baseval = np.average(flux) *4
    print(baseval)
    while j < len(flux):
        if flux[j] > baseval:
            tempvar = flux[j]
            if (flux[j] - flux[j + 1]) < 0:
                while j < len(flux) - 2 and flux[j] < flux[j + 1]:
                    tempvar = flux[j + 1]
                    j += 1
                else:
                    if j == len(flux) - 2:
                        break
                    listFlare.append(tempvar)
            else:
                if flux[j] - flux[j - 1] > 0:
                    listFlare.append(tempvar)
                j += 1
        else:
            j += 1
    return listFlare

