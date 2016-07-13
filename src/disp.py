import numpy as np
import scipy, sys
import scipy.io.wavfile as sc
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import math

output_name = "TrainBatch2.mat"
matdata2 = scipy.io.loadmat(output_name)['arr']
maxi = 0.
mini = float("inf")
for vec in matdata2:
    tmp_max=max(vec)
    if tmp_max>maxi:
        maxi=tmp_max
    tmp_min=min(vec)
    if tmp_min<=min:
        mini=tmp_min

print maxi
print mini
