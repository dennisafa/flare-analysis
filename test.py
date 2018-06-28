class Detect:


    def __init__(self):
        w359 = KeplerTargetPixelFile.from_archive(201885041, cadence='short')
        lc359 = w359.to_lightcurve(aperture_mask=w359.pipeline_mask)

    def run(self, lc359, f):

    # wolf359: 201885041
    # josh's star: 201205469
    # ran: 206208968

        track = 0
        steps = 100
        length = len(f)
        for slice in range(0, length, steps):

            flare = strPlot(lc359, slice, slice + steps)
            georgemodel = flare.computegeorge()
            guessparams = flare.guesspeaks(track)
            bounds = flare.setbounds(guessparams)
            notempty = checkzero(flare.detflares)

            if notempty:
                fitparams = flare.fit(guessparams, bounds)
                model = flare.getmodel(fitparams, [flare.time, flare.flux, flare.nflares]) + georgemodel
                plotflares(flare, model, track)

                slice += steps
                track += 1
                print("Success at range {}".format(slice))
            else:
                print("No flares in slice {}".format(slice))
                slice += steps
                track += 1

        def checkzero(self, l):

            if len(l) == 0:
                return False
            else:
                return True

        def plotflares(self, flare, model, track):

            for it, flux in enumerate(flare.detflares):
                pl.plot(flare.params[it, 0], flux, marker='x', markersize=4, color="black")

            pl.plot(flare.time, model.flatten(), '--r')
            pl.plot(flare.time, flare.flux, color='Grey', lw=0.5)
            pl.xlabel('Time - BYJD')
            pl.ylabel('Flux - Normalized to 0')
            savepath = os.path.join('/Users/Dennis/Desktop/plotswolf/test', 'wolf' + str(track) + '.png')
            pl.savefig(savepath)
            pl.clf()
