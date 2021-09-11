from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import sys
from tqdm import tqdm
import torch
import pickle
import logging
import argparse
from pathlib import Path
from torch.utils.data import SequentialSampler, DataLoader

from utils.paths import REPO_DIR, DATA_DIR

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--test-batch-size",
                    type=int,
                    required=True,
                    help="The batch size when executing translating.")
parser.add_argument("--no-cuda",
                    action='store_true',
                    help="Whether not to use CUDA when available.")
parser.add_argument("--empty-cache-freq",
                    type=int,
                    default=10,
                    help="The frequency of cache cleaning in pytorch.")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

tokenizer = AutoTokenizer.from_pretrained("./models/mbart-large-finetuned-opus-en-es-translation/", src_lang="en_XX", tgt_lang="es_XX")
model = AutoModelForSeq2SeqLM.from_pretrained("./models/mbart-large-finetuned-opus-en-es-translation/").to(device)
model.eval()

bin_path = REPO_DIR / "translate_en-es-bin"
pickle_path = bin_path / "inputs.pickle"
dataset_dir = DATA_DIR / "europarl_en_es"
en_path = dataset_dir / "europarl-v7.es-en.en"
es_path = dataset_dir / "europarl-v7.es-en.es"
res_dir = DATA_DIR / "en-es_trans_result"

with open(en_path, "r", encoding="utf-8") as f_input:
    with open(es_path, "r", encoding="utf-8") as f_para:
        i = 0
        inputs = []
        batch = []
        para = []
        for en, es in zip(f_input, f_para):
            batch.append(en.strip())
            para.append(es.strip())
            i += 1
            if i % args.test_batch_size == 0:
                inputs.append([batch, para])
                batch = []
                para = []
        
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
with open(res_dir / "generate-test.txt", "w+", encoding="utf-8") as f_res:
    for step, batch in enumerate(tqdm(inputs, desc="Iteration")):
        encodes = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")
        input_ids = encodes["input_ids"]
        input_ids = input_ids.to(device)
        with torch.no_grad():
            try:
                outputs = model.generate(input_ids, max_length=1024, num_beams=5, early_stopping=True).to("cpu")
                decodes = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for d, p, i in zip(decodes, batch[1], batch[0]):
                    f_res.write(d+"\t")
                    f_res.write(p+"\t")
                    f_res.write(i+"\n")
            except RuntimeError as e:
                if "CUDA out of memory." in str(e):
                    logger.warning("OOM detected. Skipping batch.")
                    torch.cuda.empty_cache()
        if step % (args.empty_cache_freq) == 0:
            torch.cuda.empty_cache()
            