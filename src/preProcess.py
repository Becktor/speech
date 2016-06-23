import numpy as np
import scipy
import scipy.io.wavfile as sc
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def process(s, signal_size, o):
    filters = []
    for tmp in range(signal_size):
        #print s
        sample_rate, signal = sc.read(s)
        #print sample_rate
        #print len(signal)/sample_rate
        signal_size=tmp+1

        signal = signal[int((signal_size-1) * sample_rate):int(signal_size * sample_rate)]

        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        frame_size = 0.025 # mili seconds
        frame_stride = 0.01 # mili seconds
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate #Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length-frame_length)) / frame_step))

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z)

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        frames *= np.hamming(frame_length)
        # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))  # Explicit Implementation **

        NFFT= 512
        mag_frames = np.absolute(np.fft.rfft(frames,NFFT))
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

        nfilt = 41

        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
                for k in range(f_m, f_m_plus):
                    fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
                    filter_banks = np.dot(pow_frames, fbank.T)
                    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
                filter_banks = 20 * np.log10(filter_banks)  # dB

        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
        output=o[:-4]+'_p_'+str(tmp)+'.mat'
        scipy.io.savemat(output, mdict = {'out':filter_banks}, oned_as='row')
        ### print filter_banks
        #print output
        filters.append(filter_banks)

    return filters

def plot_spec(z,title):
    k,j = z.shape
    print z.shape
    k=np.arange(0.,k)
    j=np.arange(0.,j)
    fig, ax = plt.subplots()
    ax.pcolormesh(z.T)
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.show()
   # plt.savefig('tst.png')



if __name__ == "__main__":
    output_name = 'tst.mat'
    filter_banks = process('audio/tst.wav', 4, output_name)
    matdata2 = scipy.io.loadmat(output_name+'_'+str(1))['out']
    #for tmp in range(4):
     #   plot_spec(filter_banks[tmp],str(tmp))
   # print matdata2.shape
    # And just to check if the data is the same:
    print np.all(filter_banks[1] == matdata2)
