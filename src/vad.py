import webrtcvad
from scipy.io import wavfile
import scipy
source1 = "/home/Jobe/Git/speech/audio/tst1.wav"
sample_rate, audio = wavfile.read(source1)
audio_n = audio/float(2**15)

vad = webrtcvad.Vad(3)

def audioSlice(x, fs, frames, hop):
    framesamp = int(frames*fs)
    hopsamp = int(hop*fs)
    audio_slice = scipy.array([x[i:i+framesamp] for i in range(0, len(x)-framesamp, hopsamp)])
    return audio_slice

frames=10./1000 #10 ms
hop = 1.0*frames
Z = audioSlice(audio_n, sample_rate, frames, hop)
vad.is_speech(Z[100], sample_rate)
