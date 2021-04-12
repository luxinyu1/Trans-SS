from pathlib import Path
from paths import DATA_DIR
import pickle
from nltk.corpus import stopwords

ppdb_path = DATA_DIR / 'ppdb-2.0-tldr'
pickle_path = DATA_DIR / 'ppdb.pickle'
ppdb = {}

stopset = set(stopwords.words('english'))

with open(ppdb_path, 'r', encoding='utf-8') as f_ppdb:
    for line in f_ppdb:
        line = line.strip().split("|||")
        phrase, paraphrase = line[1].strip(), line[2].strip()
        if phrase not in stopset and paraphrase not in stopset:
            if phrase in ppdb:
                ppdb[phrase].append(paraphrase)
            else:
                ppdb[phrase] = [paraphrase]
        else:
            continue

with open(pickle_path, 'wb') as f_pickle:
    pickle.dump(ppdb, f_pickle)