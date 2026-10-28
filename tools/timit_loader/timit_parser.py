import pandas as pd
import os
import librosa
import pathlib
from .load_wav import normalize
import numpy as np
from nltk.tokenize import word_tokenize



RATE_FACTOR = 1.378125
SR = 22050
TRANSCRIPT_SR = 16000

vowels = [
    "iy",
    "ih",
    "eh",
    "ey",
    "ae",
    "aa",
    "aw",
    "ay",
    "ah",
    "ao",
    "oy",
    "ow",
    "uh",
    "uw",
    "ux",
    "er",
    "ax",
    "ix",
    "axr",
    "ax-h"]


def create_spkrdata(path):
    spkrsen  = pd.read_fwf(os.path.join(path,"DOC/SPKRSENT.TXT"),comment=";")
    spkrinfo = pd.read_fwf(os.path.join(path,"DOC/SPKRINFO.TXT"),comment=";")
    columns_sent =["ID","SA","SA",'SX', 'SX', 'SX', 'SX', 'SX',"SI","SI","SI"]
    columns_info = ['ID',
     'Sex',
     'DR',
     'Use',
     'RecDate',
     'BirthDate',
     'Ht',
     'Race',
     'Edu',
     'Comments']


    spkrsen.columns = columns_sent
    spkrinfo.columns = columns_info
    spkrinfo.ID = spkrinfo.ID.str.lower()
    spkrinfo.Sex = spkrinfo.Sex.str.lower()

    spkrinfo["index"]= spkrinfo.apply(lambda row: row["Sex"]+row["ID"], axis=1)
    spkrinfo.set_index("index", inplace=True)
    return spkrinfo, spkrsen

########################
 #Create the dataset
########################
def count_dataset(path, dtype="train"):
    for (dirname, dirpath, files) in os.walk(path):
        for _file in files:
            if _file.endswith("TXT"):
                S_id = _file.split(".")[0][2:]
                S_type = _file.split(".")[0][:2]
                if S_type =="SX":
                    yield 1

def get_dialect(path):
    return int(path.split("/")[-3][-1])
    # | > x->parse(Int, filter(startswith("dr"), x)[1][end])


def yield_dataset(path):
    """
    Get all the sentences in the dataset, by name order
    """
    print("Import dataset from: {}".format(path))
    for (dirname, dirpath,files) in  os.walk(path):
        for _file in files:
            if _file.endswith("txt"):
                S_id = _file.split(".")[0]
                Speaker_ID = dirname.split("/")[-1]
                path = os.path.join(dirname, _file.split(".")[0])
                sentence  = read_utterance(path+".txt")
                words = (sentence.lower()).split()
                gender = Speaker_ID[0]
                a, b = get_word_times(path)
                words = [[w,t[0],t[1]] for w, t in zip(a,b)]
                sentence = a
                a,b =get_phones_times(path)
                phones = [[w,t[0],t[1]] for w, t in zip(a,b)]
                yield Speaker_ID, S_id, get_dialect(path), gender, path, words, phones, sentence

#%%



def create_dataset(path):
    rows=[]
    for sent in yield_dataset(path):
        rows.append(sent)
    return pd.DataFrame(rows, columns=["speaker","senID", "dialect", "gender", "path", "words", "phones" , "sentence"])

class Word():
    word = ""
    t0 = 0.
    t1 = 0.
    phones = []
    duration = 0.
    def __init__(self, word, times, phs):
        self.word=word
        self.phones=phs
        self.duration = times[1] - times[0]
        self.t0=times[0]
        self.t1=times[1]

class Phone():
    ph = ""
    t0 = 0.
    t1 = 0.
    osc = None
    db = None
    def __init__(self, ph, osc, db, times):
        self.ph=ph
        self.db=db
        self.osc=osc
        self.t0=times[0]
        self.t1=times[1]

#####################################################

def clean_phone(input):
    phs = []
    for ph in input:
        ph = ph.split()[-1].replace("\n","")
        if ph in vowels:
            ph = "_"
        phs.append(ph)
    phs= " ".join(phs)
    return phs



def get_phone(path, start, end):
    path_wav = path + ".wav"
    phn  = path + ".phn"
    phone = np.genfromtxt(phn, dtype=None, delimiter =" ")
    selected = phone[start:end]
    wav, sr = librosa.load(path_wav)
    wav = normalize(wav)
    scale_t=len(wav)/sr/phone[-1][1]
    tr0 = int(selected[0][0]*RATE_FACTOR)
    trf = int(selected[-1][1]*RATE_FACTOR)
    audio = wav[tr0: trf]
    t=np.arange(len(audio))/sr +tr0/sr
    return audio, t, (tr0/sr, trf/sr), selected


def read_utterance(path):
    utterance = open(path).read().replace(".","")
    return " ".join(utterance.split()[2:])

## Find word
def find_word(word, dataset):
    return dataset[[word in sent for sent in dataset.sentence]]

def find_words(words, dataset):
        return dataset[[any([word in sent for word in words]) for sent in dataset.sentence]]

def get_word_times(path):
    words = []
    times = []
    for word in open(path + ".wrd").read().strip().split("\n"):
        t0, t1, w = word.split()
        t0, t1 = float(t0), float(t1)
        words.append(w)
        times.append([t0, t1])
    return words, times


def find_phone(my_ph, dataset):
    def get_phones(path):
        phone = []
        for word in open(path + ".phn").read().strip().split("\n"):
            t0, t1, w = word.split()
            phone.append(w)
        return phone
    return dataset[dataset.apply(lambda row: my_ph in get_phones(row.path), axis=1)]

def get_phones_times(path):
    phones = []
    times = []
    for phone in open(path + ".phn").read().strip().split("\n"):
        t0, t1, w = phone.split()
        t0, t1 = float(t0), float(t1)
        phones.append(w)
        times.append([t0, t1])
    return phones, times



def happens_within(subinterval, interval):
    if subinterval[0] >= interval[0] and subinterval[1] <= interval[1]:
        return True
    else:
        return False

#

BAE = {"b": int(np.ceil(1 / np.log2(238.3 / 200.3))), "hop": 16, "fmin": 200, "n_bins": 20}  # 4


def get_spectra(dataframe, target_words=[], cqt_p=BAE):
    def scale_times(times, scaling):
        t0, t1 = times
        return int(t0 * scaling), int(t1 * scaling)

    words_list = []
    paths = dataframe.path
    if isinstance(dataframe.path,str):
        paths = [dataframe.path]
    print(paths)

    for my_path in paths:
        oscillogram, sr = librosa.load(my_path + ".wav")
        _words, _word_times = get_word_times(my_path)
        phone, ph_times = get_phones_times(my_path)
        final_time = ph_times[-1][-1]

        ## Use the BAE encoding
        cqt = librosa.cqt(oscillogram, sr=sr, hop_length=cqt_p["hop"], fmin=cqt_p["fmin"], n_bins=cqt_p["n_bins"],
                          bins_per_octave=cqt_p["b"])
        db = librosa.amplitude_to_db(abs(cqt))

        ## Use the oscillogram as the correct time length
        duration = len(oscillogram) / sr
        osc_sr = len(oscillogram) / duration
        db_sr = cqt.shape[1] / duration
        print(final_time/duration, TRANSCRIPT_SR)

        # %%
        words, word_times = [], []
        if target_words:
            for w,t in zip(_words, _word_times):
                if w in target_words:
                    words.append(w)
                    word_times.append(t)
        else:
            words = _words
            word_times = _word_times

        ph_times = np.array(ph_times) /TRANSCRIPT_SR
        word_times = np.array(word_times) / TRANSCRIPT_SR
        for (word, interval) in zip(words, word_times):
            phs = []
            for (ph, ph_interval) in zip(phone, ph_times):
                if happens_within(ph_interval, interval):
                    t0_db, t1_db = scale_times(ph_interval, db_sr)
                    t0_osc, t1_osc = scale_times(ph_interval, osc_sr)
                    phs.append(Phone(ph,  oscillogram[t0_osc: t1_osc - 1],db[:, t0_db:t1_db - 1],ph_interval))
            words_list.append(Word(word, interval, phs))
    return words_list
# def word_spectrum(word, path, cqt_p=BAE):
#     def scale_times(times, scaling):
#         t0, t1 = times
#         return round(t0 * scaling), int(t1 * scaling)
#
#     cqt_p = {"b": int(np.ceil(1 / np.log2(238.3 / 200.3))), "hop": 16, "fmin": 200, "n_bins": 20}  # 4
#
#     words= []
#     words_ph = []
#     accepted_intervals=[]
#
#     oscillogram, sr = librosa.load(path + ".wav")
#     words, word_times = get_word_times(path)
#     phone, ph_times, final_time = get_phones_times(path)
#
#     ## Use the BAE encoding
#     cqt = librosa.cqt(oscillogram, sr=sr, hop_length=cqt_p["hop"], fmin=cqt_p["fmin"], n_bins=cqt_p["n_bins"],
#                       bins_per_octave=cqt_p["b"])
#     db = librosa.amplitude_to_db(abs(cqt))
#
#     ## Use the oscillogram as the correct time length
#     duration = len(oscillogram) / sr
#     osc_sr = len(oscillogram) / duration
#     db_sr = cqt.shape[1] / duration
#
#     ph_times = np.array(ph_times) * duration / final_time
#     word_times = np.array(word_times) * duration / final_time
#     # %%
#
#     for (word, times) in zip(words, word_times):
#         t0_db, t1_db = scale_times(times, db_sr)
#         t0_osc, t1_osc = scale_times(times, osc_sr)
#         words.append(Word(word, db[:, t0_db:t1_db - 1],oscillogram[t0_osc: t1_osc - 1],times))
#         accepted_times.append(times)
#
#     for intervals in accepted_intervals:
#         phs = []
#         for (ph, times) in zip(phone, ph_times):
#             if
#             t0_db, t1_db = scale_times(times, db_sr)
#             t0_osc, t1_osc = scale_times(times, osc_sr)
#             phs.append(Phone(ph, db[:, t0_db:t1_db - 1],oscillogram[t0_osc: t1_osc - 1],times))
#     return words
#
# def get_word_spectra(my_word, dataset):
#     samples = []
#     for path in find_word(my_word,dataset).path:
#         words, words_db, phone, ph_db = chunked_spectrum(path)
#         matches = [n for n,match in enumerate(words) if my_word in match]
#         for n in matches:
#             samples.append(words_db[n])
#     return samples
#
# def get_ph_spectra(my_word, dataset):
#     samples = []
#     for path in find_word(my_word,dataset).Path:
#         words, words_db,_,_, phone, ph_db,_,_= chunked_spectrum(path)
#         matches = [n for n,match in enumerate(words) if my_word in match]
#         for n in matches:
#             samples.append(words_db[n])
#     return samples

# def get_sentence_paths(sentence_number):
#     """
#     Get all the paths to any occurrence of the sentence.
#
#     Return:
#          sentences:: [path1, path2, ..., pathn]
#     """
#     paths = []
#     for (id, sen_type) in get_speakers_id(sentence_number):
#         use = spkrinfo.loc[spkrinfo["ID"] == id]["Use"].values[0]
#         dr = spkrinfo.loc[spkrinfo["ID"]  == id]["DR"].values[0]
#         sex = spkrinfo.loc[spkrinfo["ID"] == id]["Sex"].values[0]
#         use = "TRAIN" if use=="TRN" else "TEST"
#         id = sex+id
#         paths.append(os.path.join(DIR,os.path.join(use,"DR"+str(dr),id,sen_type)))
#     return paths

# def get_speakers_id(sentence_number):
#     """
#     Return the ID of all the speakers that pronounce that sentence
#     """
#     ids = spkrsen.loc[spkrsen.eq(sentence_number).any(1)]["ID"].values
#     sentences = []
#     for id in ids:
#         x = spkrsen.set_index("ID").loc[id]
#         s= x.loc[x == sentence_number].index.values[0]
#         sentences.append((id,s))
#     return sentences

# def first_speaker_id(sentence_number):
#     return get_speakers_id(sentence_number)[1]
