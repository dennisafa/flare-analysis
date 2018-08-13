import george
from george import kernels
import numpy as np
from lightkurve import targetpixelfile
import matplotlib.pyplot as pl


def computegeorge (flux, time):

    f = flux

    kernel = np.var(flux) * kernels.CosineKernel(log_period=np.log(2.5), axes=0) * kernels.ExpSquaredKernel(metric=0.5)
    gp = george.GP(kernel)
    gp.compute(time, flux)
    pred_mean, pred_var = gp.predict(flux, time, return_var=True)
    #print(result)
    return pred_mean

def iter_model(self, flare):
    flux_temp = flare.flux

    george_model = computegeorge(flare.flux, flare.time)  # create an initial model
    self.final_plot = flare.flux - george_model  # plot the new model

    flare.flux = self.final_plot

    sub_model = flux_temp - george_model  # subtract the george model from the raw data
    george_model2 = computegeorge(sub_model, flare.time)  # create a model of the data with george model subbed
    sub_model2 = sub_model - george_model2
    george_model3 = computegeorge(sub_model2, flare.time)
    sub_model3 = sub_model2 - george_model3
    george_model4 = computegeorge(sub_model3, flare.time)
    clean_model = george_model2 + george_model + george_model3 + george_model4
    pl.plot(clean_model)
    pl.show()
    flare.flux = flare.flux - clean_model

    return flare

