import logging
import shutil
import argparse
from access.fairseq.base import fairseq_preprocess
from access.utils.paths import get_data_filepath, get_dataset_dir, EXP_DIR
from access.preprocessors import get_preprocessors, get_preprocessor_by_name
from access.datasets import create_preprocessed_dataset_one_preprocessor

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name",
                    default=None,
                    type=str,
                    required=True,
                    help="The name of the dataset checkpoints trained on.")

parser.add_argument("--length-ratio-target-ratio",
                    type=float,
                    default=0.8,
                    required=False)
parser.add_argument("--levenshtein-traget-ratio",
                    type=float,
                    default=0.8,
                    required=False)
parser.add_argument("--word-rank-ratio-traget-ratio",
                    type=float,
                    default=0.8,
                    required=False)
parser.add_argument("--dependency-tree-depth-ratio",
                    type=float,
                    default=0.8,
                    required=False)
parser.add_argument("--sentencepiece-vocab-size",
                    type=int,
                    default=10000,
                    required=False)

args = parser.parse_args()

def create_preprocessed_dataset(dataset, preprocessors, n_jobs=1):
    for preprocessor in preprocessors:
        # Fit preprocessor on input dataset
        preprocessor.fit(get_data_filepath(dataset, 'train', 'src'), get_data_filepath(dataset, 'train', 'dst'))
        dataset = create_preprocessed_dataset_one_preprocessor(dataset, preprocessor, n_jobs)
    return dataset

def get_preprocessors(preprocessor_kwargs):
    preprocessors = []
    for preprocessor_name, kwargs in preprocessor_kwargs.items():
        preprocessors.append(get_preprocessor_by_name(preprocessor_name)(**kwargs))
    return preprocessors


kwargs = {
    'preprocessors_kwargs': {
        'LengthRatioPreprocessor': {
            'target_ratio': args.length_ratio_target_ratio
        },
        'LevenshteinPreprocessor': {
            'target_ratio': args.levenshtein_traget_ratio
        },
        'WordRankRatioPreprocessor': {
            'target_ratio': args.word_rank_ratio_traget_ratio
        },
        'DependencyTreeDepthRatioPreprocessor': {
            'target_ratio': args.dependency_tree_depth_ratio
        },
        'SentencePiecePreprocessor': {
            'vocab_size': args.sentencepiece_vocab_size
        }
    }
}

preprocessor_kwargs = kwargs["preprocessors_kwargs"]

preprocessors = get_preprocessors(preprocessor_kwargs)
logger.info(preprocessors)

if len(preprocessors) > 0:
    dataset = create_preprocessed_dataset(args.dataset_name, preprocessors, n_jobs=1)
    
preprocessed_dir = fairseq_preprocess(dataset)