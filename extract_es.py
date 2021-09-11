from utils.paths import DATA_DIR
from utils.calc import sentence_fkf_es
from easse.fkgl import corpus_fkgl
from sacrebleu import corpus_bleu
from tqdm import tqdm
import linecache
import logging
import os
import sys
from nltk.tokenize import word_tokenize

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def verify(pair_generated):
    bleu = corpus_bleu(pair_generated['ds'], pair_generated['ts'], lowercase=True).score
    ts_tokenized = word_tokenize(pair_generated['ts'], language="spanish")
    ds_tokenized = word_tokenize(pair_generated['ds'], language="spanish")
    try:
        ts_fkf = sentence_fkf_es(pair_generated['ts'])
        ds_fkf = sentence_fkf_es(pair_generated['ds'])
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
    elif abs(ts_fkf-ds_fkf) <= 10.0:
        return False
    else:
        return True

def extract(pair_generated):
    if sentence_fkf_es(pair_generated['ts']) < sentence_fkf_es(pair_generated['ds']):
        return pair_generated['ds'], pair_generated['ts']
    else:
        return pair_generated['ts'], pair_generated['ds']

para_data = DATA_DIR / 'en-es_trans_result' / 'generate-test.txt'
simple_data = DATA_DIR / 'simple_trans_es.csv'
complex_data = DATA_DIR / 'complex_trans_es.csv'
bridge_data = DATA_DIR / 'bridge_trans_es.csv'

lenfile = sum([1 for i in open(para_data, "r")])

with open(para_data, 'r', encoding='utf-8') as f:
    with open(simple_data, 'w', encoding='utf-8') as f_s:
        with open(complex_data, 'w', encoding='utf-8') as f_c:
            with open(bridge_data, 'w', encoding='utf-8') as f_bridge:
                for line in tqdm(f, total=lenfile):
                    pair_generated = {}
                    cells = line.strip().split('\t')
                    pair_generated["ds"] = cells[0]
                    pair_generated["ts"] = cells[1]
                    if verify(pair_generated):
                        try:
                            ss, ds = extract(pair_generated)
                        except:
                            continue
                        try:
                            f_bridge.write(cells[2] + '\n')
                            f_s.write(ss + '\n')
                            f_c.write(ds + '\n')
                        except:
                            logger.warning("Empty data exists. Skipping one sample.")