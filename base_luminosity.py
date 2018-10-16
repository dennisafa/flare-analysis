import numpy as np

def base_lum(kp_mag, dist, band):

    '''
    Halle et al 2014

    kp_mag: Kepler Magnitude
    dist: Distance in parsecs
    band: Angstrom length of bandpass
    returns: Log luminosity of the star
    '''

    dist_cm = dist * 3.086e18
    str_flux = 10**((kp_mag + 20.24)/-2.5)
    lum = str_flux*((4*np.pi) * (dist_cm)**2)*(band)
    return np.log10(lum)