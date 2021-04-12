from glob import glob
import os
import sys
from pathlib import Path
import shutil
import tempfile
import logging
import numpy as np

from access.utils.paths import CACHES_DIR, FASTTEXT_EMBEDDINGS_PATH, PHASES, get_dataset_dir, get_data_filepath
from access.utils.utils import (download_and_extract, add_newline_at_end_of_file, git_clone, \
                   create_directory_or_skip, write_lines_in_parallel, yield_lines_in_parallel, lock_directory)
from access.utils.preprocess import replace_lrb_rrb, replace_lrb_rrb_file, normalize_quotes
from access.utils.text import word_tokenize

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def prepare_turkcorpus_lower():
    dataset = 'turkcorpus_lower'
    with create_directory_or_skip(get_dataset_dir(dataset)):
        url = 'https://github.com/cocoxu/simplification.git'
        cache_dir = Path(CACHES_DIR)
        cache_dir = cache_dir / 'turkcorpus'
        git_clone(url, cache_dir)
        logger.info('Processing turkcorpus_lower...')
        # Only rename files and put them in local directory architecture
        turkcorpus_lower_dir = cache_dir / 'data/turkcorpus'
        for (old_phase, new_phase) in [('test', 'test'), ('tune', 'valid')]:
            for (old_language_name, new_language_name) in [('norm', 'complex'), ('simp', 'simple')]:
                old_path = turkcorpus_lower_dir / f'{old_phase}.8turkers.tok.{old_language_name}'
                new_path = get_data_filepath('turkcorpus_lower', new_phase, new_language_name)
                shutil.copyfile(old_path, new_path)
                add_newline_at_end_of_file(new_path)
                shutil.move(replace_lrb_rrb_file(new_path), new_path)
            for i in range(8):
                old_path = turkcorpus_lower_dir / f'{old_phase}.8turkers.tok.turk.{i}'
                new_path = get_data_filepath('turkcorpus_lower', new_phase, 'simple.turk', i=i)
                shutil.copyfile(old_path, new_path)
                add_newline_at_end_of_file(new_path)
                shutil.move(replace_lrb_rrb_file(new_path), new_path)
        logger.info('Done.')
    return dataset

def prepare_turkcorpus():
    dataset = 'turkcorpus'
    with create_directory_or_skip(get_dataset_dir(dataset)):
        # Import here to avoid circular imports
        from utils.calc import get_levenshtein_similarity
        prepare_turkcorpus_lower()
        url = 'https://github.com/cocoxu/simplification.git'
        cache_dir = Path(CACHES_DIR)
        cache_dir = cache_dir / 'turkcorpus'
        git_clone(url, cache_dir)
        logger.info('Processing turkcorpus...')
        # Only rename files and put them in local directory architecture
        turkcorpus_truecased_dir = cache_dir / 'data/turkcorpus/truecased'
        for (old_phase, new_phase) in [('test', 'test'), ('tune', 'valid')]:
            # (1) read the .tsv for which each line is tab separated:
            #     `idx, complex_sentence, *turk_sentences = line.split('\t')`
            # (2) replace lrb and rrb, tokenize
            # (3) Turk sentences are shuffled for each sample so need to realign them with turkcorpus lower
            tsv_filepath = turkcorpus_truecased_dir / f'{old_phase}.8turkers.organized.tsv'
            output_complex_filepath = get_data_filepath(dataset, new_phase, 'complex')
            output_ref_filepaths = [get_data_filepath(dataset, new_phase, 'simple.turk', i) for i in range(8)]
            # These files will be used to reorder the shuffled ref sentences
            ordered_ref_filepaths = [
                get_data_filepath('turkcorpus_lower', new_phase, 'simple.turk', i) for i in range(8)
            ]
            with write_lines_in_parallel([output_complex_filepath] + output_ref_filepaths) as files:
                input_filepaths = [tsv_filepath] + ordered_ref_filepaths
                for tsv_line, *ordered_ref_sentences in yield_lines_in_parallel(input_filepaths):
                    sample_id, complex_sentence, *shuffled_ref_sentences = [
                        word_tokenize(normalize_quotes(replace_lrb_rrb(s))) for s in tsv_line.split('\t')
                    ]
                    reordered_sentences = []
                    for ordered_ref_sentence in ordered_ref_sentences:
                        # Find the position of the ref_sentence in the shuffled sentences
                        similarities = [
                            get_levenshtein_similarity(ordered_ref_sentence.replace(' ', ''),
                                                       shuffled_ref_sentence.lower().replace(' ', ''))
                            for shuffled_ref_sentence in shuffled_ref_sentences
                        ]
                        idx = np.argmax(similarities)
                        # A few sentences have differing punctuation marks
                        assert similarities[idx] > 0.98, \
                            f'{ordered_ref_sentence} != {shuffled_ref_sentences[idx].lower()} {similarities[idx]:.2f}'
                        reordered_sentences.append(shuffled_ref_sentences.pop(idx))
                    assert len(shuffled_ref_sentences) == 0
                    assert len(reordered_sentences) == 8
                    files.write([complex_sentence] + reordered_sentences)
    return dataset
         
def prepare_wikilarge():
    dataset = 'wikilarge'
    with create_directory_or_skip(get_dataset_dir(dataset)):
        cache_dir = Path(CACHES_DIR)
        # Only rename files and put them in local directory architecture
        for phase in PHASES:
            for (old_language_name, new_language_name) in [('src', 'src'), ('dst', 'dst')]:
                old_path_glob = os.path.join(cache_dir, dataset, f'*.ori.{phase}.{old_language_name}')
                globs = glob(old_path_glob)
                assert len(globs) == 1
                old_path = globs[0]
                new_path = get_data_filepath(dataset, phase, new_language_name)
                shutil.copyfile(old_path, new_path)
                shutil.move(replace_lrb_rrb_file(new_path), new_path)
                add_newline_at_end_of_file(new_path)
    return dataset

def prepare_fasttext_embeddings():
    FASTTEXT_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with lock_directory(FASTTEXT_EMBEDDINGS_PATH.parent):
        if FASTTEXT_EMBEDDINGS_PATH.exists():
            return
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz'
        extracted_path = download_and_extract(url)[0]
        shutil.move(extracted_path, FASTTEXT_EMBEDDINGS_PATH)

if __name__ == '__main__':
    prepare_turkcorpus()
    prepare_wikilarge()
    prepare_fasttext_embeddings()