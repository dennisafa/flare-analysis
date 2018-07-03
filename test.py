from scipy import stats
import matplotlib.pyplot as pl
y = stats.poisson.pmf(range(9), [1,3,2,1,2,3,5,6,7])

pl.plot(y)
pl.show()