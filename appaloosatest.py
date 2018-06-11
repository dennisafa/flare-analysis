from appaloosa import aflare as ap
from lightkurve import KeplerLightCurveFile
import matplotlib.pyplot as pl
import numpy as np

flux = KeplerLightCurveFile.from_archive(201885041)
sap = flux.SAP_FLUX.remove_nans()
p = sap.flux[250:280] # ?
t = sap.time[250:280]
model = ap.aflare(t,p)
print(model)
pl.plot(t, p)
pl.show()
