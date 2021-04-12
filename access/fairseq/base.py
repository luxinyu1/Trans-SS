# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time

from fairseq import options
from fairseq_cli import preprocess, train, generate

from access.utils.paths import get_dataset_dir, EXP_DIR
from access.utils.utils import (lock_directory, create_directory_or_skip, yield_lines)


def get_fairseq_exp_dir(job_id=None):
    if job_id is not None:
        dir_name = f'slurmjob_{job_id}'
    else:
        dir_name = f'local_{int(time.time() * 1000)}'
    return Path(EXP_DIR) / f'fairseq' / dir_name


def fairseq_preprocess(dataset):
    dataset_dir = get_dataset_dir(dataset)
    with lock_directory(dataset_dir):
        # 在原dataset目录下新建'fairseq_preprocessed'存放二进制化的preprocess数据
        preprocessed_dir = dataset_dir / 'fairseq_preprocessed'
        with create_directory_or_skip(preprocessed_dir):
            preprocessing_parser = options.get_preprocessing_parser()
            preprocess_args = preprocessing_parser.parse_args([
                '--source-lang',
                'src',
                '--target-lang',
                'dst',
                '--trainpref',
                os.path.join(dataset_dir, f'{dataset}.train'),
                '--validpref',
                os.path.join(dataset_dir, f'{dataset}.valid'),
                '--testpref',
                os.path.join(dataset_dir, f'{dataset}.test'),
                '--destdir',
                str(preprocessed_dir),
            ])
            preprocess.main(preprocess_args)
        return preprocessed_dir