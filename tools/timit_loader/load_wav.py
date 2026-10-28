import librosa
import librosa.display
import sklearn
import numpy as np
import os

def read_file(root,name):
    return read_wav(root,name)

def read_wav(root,name):
    """
    Read the wav file and return:
    name::string
    wav::np.array
    length::duration
    sr::sampling rate

    """
    tmp_path = os.path.join(root, name)
    tmp_wav, sr = librosa.load(tmp_path)
    tmp_wav = normalize_vow(tmp_wav)
    t=np.arange(len(tmp_wav))/sr
    return name[:-4], tmp_wav, t, sr

def normalize(x, axis=0,amp_range = (-1,1)):
    norm = sklearn.preprocessing.minmax_scale(x, feature_range=amp_range,axis=axis)
    norm = norm - np.mean(norm[0:8820]) #i.e the first 0.3 seconds if sampling at 22050
    norm /= max(abs(norm))
    return norm

#it's fundsmentally the same as above  but it cut some electronic generated noise from my bad recordings
def normalize_vow(x, axis=0,amp_range = (-1,1)):
    norm = normalize(x, axis, amp_range)
    return librosa.effects.trim(norm,top_db=10,frame_length=len(norm), hop_length=64)[0]

