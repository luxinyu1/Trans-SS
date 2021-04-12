from access.utils.paths import DATA_DIR
from access.utils.calc import sentence_fkf, count_words_in_sentence
from easse.fkgl import corpus_fkgl
from sacrebleu import corpus_bleu
from tqdm import tqdm
import linecache

from nltk.tokenize import word_tokenize

def verify(pair_generated):
    bleu = corpus_bleu(pair_generated['ds'], pair_generated['ts'], force=True, lowercase=True).score
    ts_tokenized = word_tokenize(pair_generated['ts'], language="english")
    ds_tokenized = word_tokenize(pair_generated['ds'], language="english")
    try:
        ts_fkf = sentence_fkf(pair_generated['ts'])
        ds_fkf = sentence_fkf(pair_generated['ds'])
    except:
        return False
    if pair_generated['ts'].lower() == pair_generated['ds'].lower():
        return False
    elif len(ts_tokenized)<=3 or len(ds_tokenized)<=3:
        return False
    elif '<unk>' in pair_generated['ds']:
        return False
    elif bleu <= 15.0:
        return False
    else:
        return True

para_data = DATA_DIR / 'de-en_trans_result' / 'generate-test.txt'
simple_data = DATA_DIR / 'simple_trans-wofkgl.csv'
complex_data = DATA_DIR /'complex_trans-wofkgl.csv'
en_row = DATA_DIR / 'wmt_en_de' / 'train.en'

lenfile = sum([1 for i in open(para_data, "r")])

with open(para_data, 'r', encoding='utf-8') as f:
    with open(simple_data, 'w', encoding='utf-8') as f_s:
        with open(complex_data, 'w', encoding='utf-8') as f_c:
            for line in tqdm(f, total=lenfile):
                if line.startswith('S'):
                    pair_generated = {}
                elif line.startswith('T'):
                    no = int(line.strip().split('\t')[0][2:])
                    row_line = linecache.getline(str(en_row), no+1).strip()
                    pair_generated['ts'] = row_line
                elif line.startswith('H'):
                    pass
                elif line.startswith("D"):
                    pair_generated['ds'] = line.strip().split('\t')[-1]
                    if verify(pair_generated):
                        f_s.write(pair_generated['ts'] + '\n')
                        f_c.write(pair_generated['ds'] + '\n')
                elif line.startswith("P"):
                    pass
                else:
                    pass
