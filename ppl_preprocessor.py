import math
import torch
import pickle
import linecache
from tqdm import tqdm
from access.utils.paths import DATA_DIR, MODELS_DIR
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

path_to_model = MODELS_DIR / 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(path_to_model).to("cuda")
tokenizer = GPT2TokenizerFast.from_pretrained(path_to_model)

para_data = DATA_DIR / 'generate-test.txt'
en_row = DATA_DIR / 'wmt_en_de' / 'train.en'
ppl_pickle = DATA_DIR / 'ppls.pickle'

lenfile = sum([1 for i in open(para_data, "r")])

ppls = []

def get_ppl(ts, ds):
    encodings = tokenizer(ts+' '+ds, return_tensors='pt')
    _encodings = tokenizer(ts, return_tensors='pt')
    start_pos = _encodings["input_ids"].size(1) - 1
    input_ids = encodings["input_ids"].to("cuda")
    target_ids = input_ids.clone()
    target_ids[0,start_pos::] = -100
    with torch.no_grad():
        loss = model(input_ids, labels=target_ids)[0]
        ppl = math.exp(loss)
    return ppl

with open(para_data, 'r', encoding='utf-8') as f:
    for line in tqdm(f, total=lenfile):
        if line.startswith('S'):
            pass
        elif line.startswith('T'):
            pair_generated = {}
            no = int(line.strip().split('\t')[0][2:])
            row_line = linecache.getline(str(en_row), no+1).strip()
            pair_generated['ts'] = row_line
        elif line.startswith('H'):
            pass
        elif line.startswith("D"):
            pair_generated['ds'] = line.strip().split('\t')[-1]
            ppls.append(get_ppl(pair_generated['ts'], pair_generated['ds']))
        elif line.startswith("P"):
            pass
        else:
            pass
        
f = open(ppl_pickle, 'wb')
pickle.dump(ppls, f)
f.close()