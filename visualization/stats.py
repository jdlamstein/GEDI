import numpy as np
import scipy.odr as odr
import scipy.stats as st
from matplotlib import pyplot as plt
import random # for testing


# put into flat array
def flatten_np(a): return np.reshape(a, (np.prod(np.shape(a))))


# wrapper for pearson coefficient
def pcc(x, y): return st.pearsonr(x, y)[0]


def odr_linear(x, y, b_guess=(0, 0), pprint=False, plot=False):
    '''

    Args:
        x: data points
        y: data points
        b_guess: necessary prelim. estimate for coefficients
        pprint: print regression estimate data
        plot: plot data points against regression line

    Returns:

    '''

    assert len(x) == len(y)

    linear = lambda b, p: p * b[0] + b[1]
    model = odr.Model(linear)
    data = odr.RealData(x, y)

    res = odr.ODR(data, model, b_guess).run()
    m, b = res.beta

    if pprint: res.pprint()
    if plot:
        plt.plot(x, y, 'b+', x, list(map(lambda p: linear((m, b), p), x)))
        plt.show()

    return m, b


'''x = np.arange(10)
y = np.array(list(map(lambda x: 2 * x + random.random() - .5, range(10))))
print(x, y)

print(odr_linear(x, y, plot=True))'''