import os
import pathlib
DIR = os.path.join(pathlib.Path(__file__).parent.parent.absolute(),"TIMIT")
# from timit_parser import *
# from __init__ import RATE_FACTOR
# RATE =1.3799112426035502 ## This ratio is the 0.5 quantile of the rate distribution. THe rate distribution is computed by the audio-end signal #h and the length of the spectrogram
# SAMPLE_RATE = 22050 # the wav file are sampled with this rate
#
#
#
# import torch.nn as nn
# import torchaudio

train_audio_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_mels=128),
)
"""
Create a hierarchy of files with single word 

"""
def get_dataset(dtype="train", sent_type="SX"):
    """
    Get all the standard (SX) sentences in the dataset, ordered by name
    """
    if dtype=="train":
        path = os.path.join(DIR,"TRAIN")
    if dtype=="test":
        path = os.path.join(DIR,"TEST")
    for (dirname, dirpath,files) in  os.walk(path):
        for _file in files:
            if _file.endswith("TXT"):
                S_id = _file.split(".")[0][2:]
                S_type = _file.split(".")[0][:2]
                if S_type ==sent_type:
                    data = os.path.join(dirname, _file.split(".")[0])
                    oscillogram, sr = librosa.load(data+".WAV")
                    sentence  = read_utterance(data+".TXT")
                    yield data
                    # yield oscillogram, sr, sentence, S_id, S_type


generate_data = get_dataset()


def split_words(sentence_path):
    def get_word(word):
        start, end, word = word.split()
        word = word.replace("\n","")
        return (int(start)*RATE, int(end)*RATE, word)

    words  = []
    with open(sentence_path+".WRD","r") as fp:
        for word in fp.readlines():
            words.append(get_word(word))
    waveform, sr = librosa.load(sentence_path + ".WAV")
    spec = train_audio_transform(waveform).squeeze(0).transpose(0, 1)
    return spec
    ## In order to not create too many artifacts in the data we do the spectrogram and then chunk over it, otherwise there will be a lot of high frequency noise at the end of each word chunk, and the need of filtering each signal with some hamming window.
    # chunks = []
    # for word in words:
    #     oscillogram[]
    # with open(sentence_path+".PHN","r") as fp:
    #     print(fp.readlines())
    # return words, oscillogram

words, oscillogram = split_words(sentence)
oscillogram = split_words(sentence)


def rate_measure(sentence_path):
    oscillogram, sr = librosa.load(sentence_path + ".WAV")
    with open(sentence_path+".PHN","r") as fp:
        end_time = fp.readlines()[-1].split()[1]
    return len(oscillogram)/int(end_time)

rates = []
for sentence in generate_data:
    rates.append(rate_measure((sentence)))

import numpy as np

# class WORD(np.Array):


import torch
specs = []
for z in range(50):
    oscillogram, sr = librosa.load(sentence_path + ".WAV")
    with open(sentence_path+".PHN","r") as fp:
        end_time = fp.readlines()[-1].split()[1]
    return len(oscillogram)/int(end_time)
    sentence = generate_data.__next__()
    waveform, sr = librosa.load(sentence + ".WAV")
    spec = train_audio_transform(torch.from_numpy(waveform)).squeeze(0).transpose(0, 1)
    specs.append(spec)
    print(spec.size())

spectrograms = nn.utils.rnn.pad_sequence(specs, batch_first=True).unsqueeze(1).transpose(2, 3)

spectrograms.size()
