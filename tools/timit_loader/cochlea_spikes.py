import os

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
                gender = Speaker_ID[0]
                yield Speaker_ID, S_id,  path

#%%



def create_dataset(path):
    rows=[]
    for sent in yield_dataset(path):
        rows.append(sent)
    return pd.DataFrame(rows, columns=["speaker","senID", "dialect", "gender", "path", "words", "phones" , "sentence"])
