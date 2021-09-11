from utils.paths import DATA_DIR
from utils.calc import sentence_fkf, count_words_in_sentence
from easse.fkgl import corpus_fkgl
from sacrebleu import corpus_bleu
from tqdm import tqdm
import linecache

def verify(pair_generated):
    try:
        ts_fkf = sentence_fkf(pair_generated['ts'])
        ds_fkf = sentence_fkf(pair_generated['ds'])
    except:
        return False
    if abs(ts_fkf-ds_fkf) <= 10.0:
        return False
    else:
        return True

def extract(pair_generated):
    if sentence_fkf(pair_generated['ts']) < sentence_fkf(pair_generated['ds']):
        return pair_generated['ds'], pair_generated['ts']
    else:
        return pair_generated['ts'], pair_generated['ds']

para_data = DATA_DIR / 'de-en_trans_result' / 'generate-test.txt'
simple_data = DATA_DIR / 'simple_trans-wobleu.csv'
complex_data = DATA_DIR /'complex_trans-wobleu.csv'
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
                        try:
                            ss, ds = extract(pair_generated)
                        except:
                            continue
                        f_s.write(ss + '\n')
                        f_c.write(ds + '\n')
                elif line.startswith("P"):
                    pass
                else:
                    pass
