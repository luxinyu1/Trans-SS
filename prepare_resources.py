from glob import glob
import os
from pathlib import Path
import shutil
import logging
import numpy as np

from access.utils.paths import ASSET_URL, TURKCORPUS_URL, WIKILARGE_URL, BART_URL, MBART_URL, MODELS_DIR, get_dataset_dir
from access.utils.utils import download_and_extract, create_directory_or_skip

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

def prepare_models(url_list):
    for model_url in url_list:
        download_and_extract(model_url, MODELS_DIR)

if __name__ == '__main__':
    model_urls = [BART_URL, MBART_URL]

    prepare_asset()
    prepare_turkcorpus()
    prepare_wikilarge()
    prepare_models(model_urls)