import numpy as np
import scipy, sys
import scipy.io.wavfile as sc
from scipy.fftpack import dct
import matplotlib.pyplot as plt

output_name = "tst.mat"
matdata2 = scipy.io.loadmat(output_name)['out']
print matdata2.shape
