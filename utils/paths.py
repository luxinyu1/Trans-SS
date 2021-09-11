from itertools import product
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = REPO_DIR / 'checkpoints'
DATASETS_DIR = REPO_DIR / 'datasets'
MODELS_DIR = REPO_DIR / 'models'
CACHES_DIR = REPO_DIR / 'caches'
DATA_DIR = REPO_DIR / 'data'
EXP_DIR = REPO_DIR / 'access' / 'experiments'

LANGUAGES = ['src', 'dst']
PHASES = ['train', 'valid', 'test']

WIKILARGE_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/data/wikilarge.zip'
ASSET_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/data/asset.zip'
TURKCORPUS_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/data/turkcorpus.zip'

BART_URL = 'https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz'
MBART_URL = 'https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz'
WMT_19_DE_EN_URL = 'https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'


DE_EN_TRANS_RESULT_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/datasets/de-en_trans_result.tar.gz'
EN_ES_TRANS_RESULT_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/datasets/en-es_trans_result.tar.gz'
EN_FR_TRANS_RESULT_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/datasets/en-fr_trans_result.tar.gz'

TRANS_EN_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/datasets/trans-1M.tar.gz'
TRANS_FR_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/datasets/trans-fr.tar.gz'
TRANS_ES_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/datasets/trans-es.tar.gz'

WMT_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/data/wmt_en_de.tgz'
EUROPARL_URL = 'https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/data/training-parallel-europarl-v7.tgz'

def get_dataset_dir(dataset):
    return DATASETS_DIR / dataset

def get_data_dir(data):
    return DATA_DIR / data

def get_data_filepath(dataset, phase, language, i=None):
    suffix = ''
    if i is not None:
        suffix = f'.{i}'
    filename = f'{dataset}.{phase}.{language}{suffix}'
    return get_dataset_dir(dataset) / filename

def get_filepaths_dict(dataset):
    return {(phase, language): get_data_filepath(dataset, phase, language)
            for phase, language in product(PHASES, LANGUAGES)}