from access.utils.paths import DATA_DIR
import linecache

from tqdm import tqdm

para_data = DATA_DIR / 'de-en_trans_result' / 'generate-test.txt'
simple_data = DATA_DIR / 'simple_para.csv'
complex_data = DATA_DIR / 'complex_para.csv'
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
                    f_s.write(pair_generated['ts'] + '\n')
                    f_c.write(pair_generated['ds'] + '\n')
                elif line.startswith("P"):
                    pass
                else:
                    pass