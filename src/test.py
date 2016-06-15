from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("audio/s1_an_1.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

print fbank_feat[1:3,:]
