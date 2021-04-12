# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import hashlib
import logging
from pathlib import Path

from .utils.preprocess import get_parallel_file_pair_preprocessor
from .preprocessors import dump_preprocessors, load_preprocessors
from .utils.paths import PHASES, get_dataset_dir, get_data_filepath, get_filepaths_dict
from .utils.utils import count_lines, read_lines, create_directory_or_skip

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def yield_indexes_of_lines(filepath, lines):
    lines = set(lines)
    with Path(filepath).open('r') as f:
        for idx, line in enumerate(f):
            if line.strip('\n') in lines:
                yield idx


def sort_files_by_line_count(filepaths):
    return sorted(filepaths, key=lambda filepath: count_lines(filepath))


def has_lines_in_common(filepath1, filepath2):
    [smallest_filepath, largest_filepath] = sort_files_by_line_count([filepath1, filepath2])
    for idx in yield_indexes_of_lines(largest_filepath, read_lines(smallest_filepath)):
        return True
    return False


def get_preprocessed_dataset_name(dataset, preprocessor):
    return '_' + hashlib.md5((dataset + preprocessor.get_hash()).encode()).hexdigest()


def create_preprocessed_dataset_one_preprocessor(dataset, preprocessor, n_jobs):
    new_dataset = get_preprocessed_dataset_name(dataset, preprocessor)
    with create_directory_or_skip(get_dataset_dir(new_dataset)):
        logger.info(f'Creating preprocessed dataset with {preprocessor}: {dataset} -> {new_dataset}')
        new_dataset_dir = get_dataset_dir(new_dataset)
        filepaths_dict = get_filepaths_dict(dataset)
        new_filepaths_dict = get_filepaths_dict(new_dataset)
        for phase in PHASES:
            if not filepaths_dict[phase, 'src'].exists() or not filepaths_dict[phase, 'src'].exists():
                continue
            parallel_file_pair_preprocessor = get_parallel_file_pair_preprocessor(
                preprocessor.encode_file_pair,
                n_jobs=n_jobs,
            )
            parallel_file_pair_preprocessor(filepaths_dict[phase, 'src'], filepaths_dict[phase, 'dst'],
                                            new_filepaths_dict[phase, 'src'], new_filepaths_dict[phase, 'dst'])
            previous_preprocessors = load_preprocessors(get_dataset_dir(dataset))
        if previous_preprocessors is not None:
            preprocessors = previous_preprocessors + [preprocessor]
        else:
            preprocessors = [preprocessor]
        dump_preprocessors(preprocessors, new_dataset_dir)
        with open(new_dataset_dir / 'original_dataset', 'w') as f:
            f.write(dataset + '\n')

    return new_dataset


def create_preprocessed_dataset(dataset, preprocessors, n_jobs=1):
    for preprocessor in preprocessors:
        # Fit preprocessor on input dataset
        preprocessor.fit(get_data_filepath(dataset, 'train', 'src'), get_data_filepath(dataset, 'train', 'dst'))
        dataset = create_preprocessed_dataset_one_preprocessor(dataset, preprocessor, n_jobs)
    return dataset