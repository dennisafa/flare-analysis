import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits

fits_file = fits.open('/Users/Dennis/Desktop/LP-95-58/hlsp_tess-data-alerts_tess_phot_00307210830-s02_tess_v1_tp (1).fits')

y = fits_file[1].data['FLUX']
x = fits_file[1].data['TIME']

print(x)
print(y)
pl.plot(x, y)
pl.show()


print(y)