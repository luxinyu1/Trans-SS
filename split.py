import os
import sys
import csv
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from access.utils.paths import DATA_DIR
from access.utils.calc import sentence_fkf, sentence_fkgl
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
                    required=True,
                    type=int,
                    help="The number of simple and complex file to be splited into train, test and valid data.")
parser.add_argument("--dataset",
                    required=True,
                    type=str,
                    help="The name of dataset to be splited.")

args = parser.parse_args()

source_data = pd.read_csv(DATA_DIR / ('complex_'+args.dataset+'.csv'), encoding='utf-8', delimiter='\n', header=None, names=['source'], quoting=csv.QUOTE_NONE)
target_data = pd.read_csv(DATA_DIR / ('simple_'+args.dataset+'.csv'), encoding='utf=8', delimiter='\n', header=None, names=['target'], quoting=csv.QUOTE_NONE)

if args.use_num > len(source_data):
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

if args.use_num > len(source_data):
    logger.info("{} samples used.".format(len(source_data)))
else:
    logger.info("{} samples used.".format(args.use_num))

train, test = train_test_split(data, test_size=0.1)

train, val = train_test_split(train, test_size=(1.0/9.0))

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

train['source'].to_csv(Path(args.output_dir) / (args.dataset+".train.src"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")
train['target'].to_csv(Path(args.output_dir) / (args.dataset+".train.dst"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")

test['source'].to_csv(Path(args.output_dir) / (args.dataset+".test.src"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")
test['target'].to_csv(Path(args.output_dir) / (args.dataset+".test.dst"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")

val['source'].to_csv(Path(args.output_dir) / (args.dataset+".valid.src"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")
val['target'].to_csv(Path(args.output_dir) / (args.dataset+".valid.dst"), index=False, header=0, quoting=csv.QUOTE_NONE, sep='\n', na_rep=" ")