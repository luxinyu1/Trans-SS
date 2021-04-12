from access.utils.paths import DATA_DIR
from access.utils.calc import sentence_fkf, count_words_in_sentence
from sacrebleu import corpus_bleu
from tseval.feature_extraction import get_wordrank_score, get_compression_ratio

from tqdm import tqdm

para_data = DATA_DIR / 'parabank2.tsv'
simple_data = DATA_DIR / 'simple_parabank.csv'
complex_data = DATA_DIR / 'complex_parabank.csv'

lenfile = sum([1 for i in open(para_data, "r")])

with open(para_data, 'r', encoding='utf-8') as f:
    with open(simple_data, 'w', encoding='utf-8') as f_s:
        with open(complex_data, 'w', encoding='utf-8') as f_c:
            for line in tqdm(f, total=lenfile):
                tabs = line.strip().split('\t')
                sentences = tabs[1:]
                fkf = []
                for sentence in sentences:
                    try:
                        fkf.append(sentence_fkf(sentence))
                    except:
                        break
                s = sentences[fkf.index(max(fkf))]
                c = sentences[0]
                if count_words_in_sentence(c) <= 9:
                    continue
                elif max(fkf)-fkf[0] <= 5.0:
                    continue
                elif get_compression_ratio(c, s) > 1.1:
                    continue
                else:
                    f_s.write(s+'\n')
                    f_c.write(c+'\n')