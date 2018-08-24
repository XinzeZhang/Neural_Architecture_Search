# from pandas import DataFrame
# from pandas import Series
# from pandas import concat
# from pandas import read_csv
# from pandas import datetime

# # from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler

# import numpy as np
# from numpy import concatenate

import math

# import matplotlib
# import matplotlib.ticker as ticker
# import matplotlib.pyplot as plt

import time

# from matplotlib import animation

# convert an array of values into a dataset matrix



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def asMinutesUnit(s):
    m = math.floor(s / 60)
    s -= m * 60
    m+=s/60.0
    return '%.3f' % (m)
# time-count


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))