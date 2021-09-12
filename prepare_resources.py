import logging
import shutil
import glob

from utils.paths import ASSET_URL, CACHES_DIR, DATA_DIR, DATASETS_DIR, EN_ES_TRANS_MODEL_GIT_URL, EUROPARL_URL, TRANS_EN_URL, TRANS_ES_URL, TRANS_ES_URL, TRANS_FR_URL, \
                                TURKCORPUS_URL, WIKILARGE_URL, BART_URL, MBART_URL, MODELS_DIR, \
                                DE_EN_TRANS_RESULT_URL, EN_ES_TRANS_RESULT_URL, EN_FR_TRANS_RESULT_URL, WMT_14_EN_FR_URL, WMT_19_DE_EN_URL, WMT_URL,  \
                                get_dataset_dir, get_data_dir
from utils.utils import download_and_extract, create_directory_or_skip, git_clone

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def prepare_asset():
    dataset = 'asset'
    with create_directory_or_skip(get_dataset_dir(dataset)):
        download_and_extract(ASSET_URL, get_dataset_dir(dataset))
    return dataset

def prepare_turkcorpus():
    dataset = 'turkcorpus'
    with create_directory_or_skip(get_dataset_dir(dataset)):
        download_and_extract(TURKCORPUS_URL, get_dataset_dir(dataset))
    return dataset
         
def prepare_wikilarge():
    dataset = 'wikilarge'
    with create_directory_or_skip(get_dataset_dir(dataset)):
        download_and_extract(WIKILARGE_URL, get_dataset_dir(dataset))
    return dataset

def prepare_trans_result(url_list):
    for trans_result in url_list:
        download_and_extract(trans_result, DATA_DIR)

def prepare_mined_datasets(url_list):
    for mined_dataset in url_list:
        download_and_extract(mined_dataset, DATASETS_DIR)

def prepare_models(url_list):
    for model_url in url_list:
        download_and_extract(model_url, MODELS_DIR)

def prepare_wmt():
    with create_directory_or_skip(get_data_dir('wmt_en_de')):
        download_and_extract(WMT_URL, get_data_dir('wmt_en_de'))

def prepare_trans_corpora():
    prepare_wmt()
    europarl_cache_dir = CACHES_DIR / 'europarl'
    with create_directory_or_skip(europarl_cache_dir):
        download_and_extract(EUROPARL_URL, CACHES_DIR / 'europarl')
    with create_directory_or_skip(get_data_dir('europarl_en_fr')), \
        create_directory_or_skip(get_data_dir('europarl_en_es')):

        en_fr = glob.glob(str(europarl_cache_dir / 'training' / r'*.fr-en.*'))
        en_es = glob.glob(str(europarl_cache_dir / 'training' / r'*.es-en.*'))
        for p in en_fr:
            shutil.copy(p, get_data_dir('europarl_en_fr'))
        for p in en_es:
            shutil.copy(p, get_data_dir('europarl_en_es'))

if __name__ == '__main__':
    
    model_urls = [BART_URL, MBART_URL, WMT_14_EN_FR_URL, WMT_19_DE_EN_URL]
    trans_result_urls = [DE_EN_TRANS_RESULT_URL, EN_ES_TRANS_RESULT_URL, EN_FR_TRANS_RESULT_URL]
    mined_datasets_urls = [TRANS_EN_URL, TRANS_ES_URL, TRANS_FR_URL]

    prepare_asset()
    prepare_turkcorpus()
    prepare_wikilarge()
    
    prepare_models(model_urls)
    git_clone(EN_ES_TRANS_MODEL_GIT_URL, MODELS_DIR)
    
    prepare_trans_result(trans_result_urls)
    prepare_trans_corpora()
    prepare_mined_datasets(mined_datasets_urls)