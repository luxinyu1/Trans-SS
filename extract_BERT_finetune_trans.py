from access.utils.paths import DATA_DIR, DATASETS_DIR
import pickle

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from tqdm import tqdm

complex_data = DATA_DIR / 'complex_trans.csv'
simple_data = DATA_DIR / 'simple_trans.csv'
finetune_bert_data = DATASETS_DIR / 'finetune_BERT.tsv'
ppdb_pickle = DATA_DIR / 'ppdb.pickle'

lenfile = sum([1 for i in open(complex_data, "r")])

with open(ppdb_pickle, 'rb') as f_pickle:
    ppdb = pickle.load(f_pickle)
    
with open(complex_data, 'r', encoding='utf-8') as f_complex:
    with open(simple_data, 'r', encoding='utf-8') as f_simple:
        with open(finetune_bert_data, 'w', encoding='utf-8') as f_finetune:
            for complex, simple in tqdm(zip(f_complex, f_simple), total=lenfile):
                tokenized_complex = word_tokenize(complex.strip())
                tokenized_simple = word_tokenize(simple.strip())
                for word in tokenized_complex:
                    if word in ppdb:
                        paraphrases = ppdb[word]
                        for paraphrase in paraphrases:
                            if paraphrase in simple:
                                f_finetune.write("{}\t{}\t{}\t{}\n".format(complex, simple, word, paraphrase))
                                break
                            else:
                                continue
                    else:
                        continue