import matplotlib.pyplot as plt
import numpy as np

x = np.cumsum(np.random.random(1000) - 0.5)

fig, (ax1, ax2) = plt.subplots(nrows=2)
data, freqs, bins, im = ax1.specgram(x)
print freqs
print bins
ax1.axis('tight')

# "specgram" actually plots 10 * log10(data)...
ax2.pcolormesh(bins, freqs, 10 * np.log10(data))
ax2.axis('tight')

plt.show()
