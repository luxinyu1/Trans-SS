import os
import sys
import csv
import shutil
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from utils.paths import DATA_DIR, DATASETS_DIR
from utils.calc import sentence_fkf, sentence_fkgl
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--output-dir",
                    type=str,
                    required=True,
                    help="The output dir.")
parser.add_argument("--use-num",
                    required=False,
                    type=int,
                    help="The number of simple and complex file to be splited into train, test and valid data.")
parser.add_argument("--dataset",
                    required=True,
                    type=str,
                    help="The name of dataset to be splited.")
parser.add_argument("--using-full",
                    default=False,
                    action="store_true",
                    help="Whether to use the full dataset or not.")

args = parser.parse_args()

if args.dataset != 'wikilarge':
    source_data = pd.read_csv(DATA_DIR / ('complex_'+args.dataset+'.csv'), encoding='utf-8', delimiter='\n', header=None, names=['source'], quoting=csv.QUOTE_NONE)
    target_data = pd.read_csv(DATA_DIR / ('simple_'+args.dataset+'.csv'), encoding='utf=8', delimiter='\n', header=None, names=['target'], quoting=csv.QUOTE_NONE)

else:
    source_data = pd.read_csv(DATASETS_DIR / 'wikilarge' / 'wikilarge.train.src', encoding='utf-8', delimiter='\n', header=None, names=['source'], quoting=csv.QUOTE_NONE)
    target_data = pd.read_csv(DATASETS_DIR / 'wikilarge' / 'wikilarge.train.dst', encoding='utf-8', delimiter='\n', header=None, names=['target'], quoting=csv.QUOTE_NONE)

if args.using_full:
    logger.info("Using full({} samples).".format(len(source_data)))
    data = pd.concat([source_data, target_data], axis=1)        
elif args.use_num > len(source_data):
    logger.warning("Exceeding the maximum samples of the dataset, using full({} samples).".format(len(source_data)))
    data = pd.concat([source_data, target_data], axis=1)
else:
    data = pd.concat([source_data, target_data], axis=1).sample(n=args.use_num, replace=False, random_state=1)

src = data["source"].tolist()
dst = data["target"].tolist()

fkf_ss = []
fkf_ds = []
delta_fkfs = []
fkgl_ss = []
fkgl_ds = []
delta_fkgls = []

for s, d in zip(src, dst):
    s = s.strip()
    d = d.strip()
    try:
        fkf_s = sentence_fkf(s)
        fkf_d = sentence_fkf(d)
        fkgl_s = sentence_fkgl(s)
        fkgl_d = sentence_fkgl(d)
        delta_fkf = abs(fkf_s - fkf_d)
        delta_fkgl = abs(fkgl_s - fkgl_d)
        fkf_ss.append(fkf_s)
        fkf_ds.append(fkf_d)
        delta_fkfs.append(delta_fkf)

        fkgl_ss.append(fkgl_s)
        fkgl_ds.append(fkgl_d)
        delta_fkgls.append(delta_fkgl)

    except:
        continue

mean_delta_fkf = np.mean(delta_fkfs)
mean_delta_fkgl = np.mean(delta_fkgls)

logger.info("mean delta fkf:{} | mean delata fkgl:{}".format(mean_delta_fkf, mean_delta_fkgl))

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

data['source'].to_csv(Path(args.output_dir) / (args.dataset+".train.src"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")
data['target'].to_csv(Path(args.output_dir) / (args.dataset+".train.dst"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")

# dummy test and valid files

if args.dataset == 'trans_fr':
    
    shutil.copyfile(DATASETS_DIR / 'alector' / 'alector.test.complex', Path(args.output_dir) / (args.dataset+".test.src"))
    shutil.copyfile(DATASETS_DIR / 'alector' / 'alector.test.simple', Path(args.output_dir) / (args.dataset+".test.dst"))
    
    shutil.copyfile(DATASETS_DIR / 'alector' / 'alector.valid.complex', Path(args.output_dir) / (args.dataset+".valid.src"))
    shutil.copyfile(DATASETS_DIR / 'alector' / 'alector.valid.simple', Path(args.output_dir) / (args.dataset+".valid.dst"))
    
elif args.dataset == 'trans_es':
    
    shutil.copyfile(DATASETS_DIR / 'simplext' / 'simplext.test.complex', Path(args.output_dir) / (args.dataset+".test.src"))
    shutil.copyfile(DATASETS_DIR / 'simplext' / 'simplext.test.simple', Path(args.output_dir) / (args.dataset+".test.dst"))
    
    shutil.copyfile(DATASETS_DIR / 'simplext' / 'simplext.valid.complex', Path(args.output_dir) / (args.dataset+".valid.src"))
    shutil.copyfile(DATASETS_DIR / 'simplext' / 'simplext.valid.simple', Path(args.output_dir) / (args.dataset+".valid.dst"))
    
else:

    shutil.copyfile(DATASETS_DIR / 'wikilarge' / 'wikilarge.test.src', Path(args.output_dir) / (args.dataset+".test.src"))
    shutil.copyfile(DATASETS_DIR / 'wikilarge' / 'wikilarge.test.dst', Path(args.output_dir) / (args.dataset+".test.dst"))

    shutil.copyfile(DATASETS_DIR / 'wikilarge' / 'wikilarge.valid.src', Path(args.output_dir) / (args.dataset+".valid.src"))
    shutil.copyfile(DATASETS_DIR / 'wikilarge' / 'wikilarge.valid.dst', Path(args.output_dir) / (args.dataset+".valid.dst"))

# test['source'].to_csv(Path(args.output_dir) / (args.dataset+".test.src"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")
# test['target'].to_csv(Path(args.output_dir) / (args.dataset+".test.dst"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")

# val['source'].to_csv(Path(args.output_dir) / (args.dataset+".valid.src"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")
# val['target'].to_csv(Path(args.output_dir) / (args.dataset+".valid.dst"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")