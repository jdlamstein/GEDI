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

class threshold:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.temp_x, self.temp_y = map(np.copy, (x, y))
        self.m, self.b = odr_linear(x, y, [0] * 2)
        self.lin = lambda x: self.m * x + self.b

    def points_above(self, t):
        pairs_gt = [pair for pair in zip(self.x, self.y) if pair[0] >= t or pair[1] >= self.lin(t)]
        return zip(*pairs_gt)

    def points_below(self, t):
        pairs_lt = [pair for pair in zip(self.x, self.y) if pair[0] < t and pair[1] < self.lin(t)]
        return zip(*pairs_lt)

    def iter_below(self, t):
        pairs_lt = [pair for pair in zip(self.temp_x, self.temp_y) if pair[0] < t and pair[1] < self.lin(t)]
        self.temp_x, self.temp_y = zip(*pairs_lt)
        return pcc(self.temp_x, self.temp_y)

'''x = np.arange(10)
y = np.array(list(map(lambda x: 2 * x + random.random() - .5, range(10))))
print(x, y)

print(odr_linear(x, y, plot=True))'''