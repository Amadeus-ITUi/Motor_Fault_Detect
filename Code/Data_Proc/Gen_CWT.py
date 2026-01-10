import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from Read_CWRU_mat import get_cwru_path, load_and_preprocess

data_list3 = load_and_preprocess(get_cwru_path('DE', 1730, '0.007', 'InnerRace'))
data_array3 = data_list3[0:512]
sampling_period = 1.0/12000
totalscale = 128
wavename = 'cmor1-1'

fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscale
scales = cparam / np.arange(totalscale, 0, -1)

coefficients, frequencies = pywt.cwt(data_array3, scales, wavename, sampling_period)
amp = abs(coefficients)
frequ_max = frequencies.max()

t = np.linspace(0, sampling_period, 512, endpoint=False)

plt.contourf(t, frequencies, amp, cmap='jet')
plt.title('滚珠-512-128-cmor1-1')
plt.show()
