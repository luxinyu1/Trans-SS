from itertools import product
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = REPO_DIR / 'checkpoints'
DATASETS_DIR = REPO_DIR / 'datasets'
MODELS_DIR = REPO_DIR / 'models'
CACHES_DIR = REPO_DIR / 'caches'
BEST_MODEL_DIR = MODELS_DIR / 'best_model'
DATA_DIR = REPO_DIR / 'data'
FASTTEXT_EMBEDDINGS_PATH = MODELS_DIR / 'fasttext-vectors' / 'wiki.en.vec'
SUBTLEX_WORD_FREQ_PATH = DATA_DIR / 'SUBTLEX_frequency.xlsx'
EXP_DIR = REPO_DIR / 'access' / 'experiments'

LANGUAGES = ['src', 'dst']
PHASES = ['train', 'valid', 'test']

def get_dataset_dir(dataset):
    return DATASETS_DIR / dataset


def get_data_filepath(dataset, phase, language, i=None):
    suffix = ''
    if i is not None:
        suffix = f'.{i}'
    filename = f'{dataset}.{phase}.{language}{suffix}'
    return get_dataset_dir(dataset) / filename


def get_filepaths_dict(dataset):
    return {(phase, language): get_data_filepath(dataset, phase, language)
            for phase, language in product(PHASES, LANGUAGES)}