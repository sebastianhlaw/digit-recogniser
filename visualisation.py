# Author:   Sebastian Law
# Date:     22-Nov-2016
# Revised:  22-Nov-2016

import numpy as np
from matplotlib.pylab import plt


def plot(Xi, yi=None):
    plt.close('all')
    side = np.sqrt(Xi.size).astype(int)
    data = np.array(Xi).reshape((side, side))
    plt.imshow(data, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("y = "+str(yi))
    plt.show()
