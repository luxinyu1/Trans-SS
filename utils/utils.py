import os
import git
import sys
import time
import pandas as pd
from urllib.parse import urlparse
import logging
import shutil
import tempfile
from tqdm import tqdm
from pathlib import Path
from itertools import zip_longest
from fcntl import flock, LOCK_EX, LOCK_UN
from urllib.request import urlretrieve
from contextlib import contextmanager, AbstractContextManager

import bz2
import gzip
import tarfile
import zipfile

from .paths import DATASETS_DIR, CACHES_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

@contextmanager
def open_files(filepaths, mode='r'):
    files = []
    try:
        files = [Path(filepath).open(mode) for filepath in filepaths]
        yield files
    finally:
        [f.close() for f in files]


def yield_lines_in_parallel(filepaths, strip=True, strict=True, n_lines=float('inf')):
    assert type(filepaths) == list
    with open_files(filepaths) as files:
        for i, parallel_lines in enumerate(zip_longest(*files)):
            if i >= n_lines:
                break
            if None in parallel_lines:
                assert not strict, f'Files don\'t have the same number of lines: {filepaths}, use strict=False'
            if strip:
                parallel_lines = [l.rstrip('\n') if l is not None else None for l in parallel_lines]
            yield parallel_lines
            
class FilesWrapper:
    '''Write to multiple open files at the same time'''
    def __init__(self, files, strict=True):
        self.files = files
        self.strict = strict  # Whether to raise an exception when a line is None

    def write(self, lines):
        assert len(lines) == len(self.files)
        for line, f in zip(lines, self.files):
            if line is None:
                assert not self.strict
                continue
            f.write(line.rstrip('\n') + '\n')


@contextmanager
def write_lines_in_parallel(filepaths, strict=True):
    with open_files(filepaths, 'w') as files:
        yield FilesWrapper(files, strict=strict)


def write_lines(lines, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w') as f:
        for line in lines:
            f.write(line + '\n')


def yield_lines(filepath, n_lines=float('inf'), prop=1):
    if prop < 1:
        assert n_lines == float('inf')
        n_lines = int(prop * count_lines(filepath))
    with open(filepath, 'r') as f:
        for i, l in enumerate(f):
            if i >= n_lines:
                break
            yield l.rstrip('\n')


def read_lines(filepath, n_lines=float('inf'), prop=1):
    return list(yield_lines(filepath, n_lines, prop))


def count_lines(filepath):
    n_lines = 0
    with Path(filepath).open() as f:
        for l in f:
            n_lines += 1
    return n_lines

class create_directory_or_skip(AbstractContextManager):
    '''Context manager for creating a new directory (with rollback and skipping with block if exists)
    In order to skip the execution of the with block if the dataset already exists, this context manager uses deep
    magic from https://stackoverflow.com/questions/12594148/skipping-execution-of-with-block
    '''
    def __init__(self, dir_path, overwrite=False):
        self.dir_path = Path(dir_path)
        self.overwrite = overwrite

    def __enter__(self):
        if self.dir_path.exists():
            self.directory_lock = lock_directory(self.dir_path)
            self.directory_lock.__enter__()
            files_in_directory = list(self.dir_path.iterdir())
            if set(files_in_directory) in [set([]), set([self.dir_path / '.lockfile'])]:
                # TODO: Quick hack to remove empty directories
                self.directory_lock.__exit__(None, None, None)
                logger.info(f'Removing empty directory {self.dir_path}')
                shutil.rmtree(self.dir_path)
            else:
                # Deep magic hack to skip the execution of the code inside the with block
                # We set the trace to a dummy function
                sys.settrace(lambda *args, **keys: None)
                # Get the calling frame (sys._getframe(0) is the current frame)
                frame = sys._getframe(1)
                # Set the calling frame's trace to the one that raises the special exception
                frame.f_trace = self.trace
                return
        logger.info(f'Creating {self.dir_path}...')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.directory_lock = lock_directory(self.dir_path)
        self.directory_lock.__enter__()

    def trace(self, frame, event, arg):
        # This method is called when a new local scope is entered, i.e. right when the code in the with block begins
        # The exception will therefore be caught by the __exit__()
        raise SkipWithBlock()

    def __exit__(self, type, value, traceback):
        self.directory_lock.__exit__(type, value, traceback)
        if type is not None:
            if issubclass(type, SkipWithBlock):
                return True  # Suppress special SkipWithBlock exception
            if issubclass(type, BaseException):
                # Rollback
                logger.info(f'Error: Rolling back creation of directory {self.dir_path}')
                shutil.rmtree(self.dir_path)
                return False  # Reraise the exception

class SkipWithBlock(Exception):
    pass

def reporthook(count, block_size, total_size):
    # Download progress bar
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size_mb = count * block_size / (1024 * 1024)
    speed = progress_size_mb / duration
    percent = int(count * block_size * 100 / total_size)
    msg = f'\r... {percent}% - {int(progress_size_mb)} MB - {speed:.2f} MB/s - {int(duration)}s'
    sys.stdout.write(msg)
            
def download(url, destination_path):
    logger.info('Downloading...')
    try:
        urlretrieve(url, destination_path, reporthook)
        sys.stdout.write('\n')
    except (Exception, KeyboardInterrupt, SystemExit):
        logger.info('Rolling back: remove partially downloaded file')
        os.remove(destination_path)
        raise

def download_and_extract(url, target_dir):
    compressed_filename = url.split('/')[-1]
    compressed_filepath = CACHES_DIR / compressed_filename
    download(url, compressed_filepath)
    logger.info('Extracting to {}...'.format(target_dir))
    return extract(compressed_filepath, target_dir)

def extract(filepath, output_dir):
    # Infer extract method based on extension
    extensions_to_methods = {
        '.tar.gz': untar,
        '.tar.bz2': untar,
        '.tgz': untar,
        '.zip': unzip,
        '.gz': ungzip,
        '.bz2': unbz2,
    }

    def get_extension(filename, extensions):
        possible_extensions = [ext for ext in extensions if filename.endswith(ext)]
        if len(possible_extensions) == 0:
            raise Exception(f'File {filename} has an unknown extension')
        # Take the longest (.tar.gz should take precedence over .gz)
        return max(possible_extensions, key=lambda ext: len(ext))

    filename = os.path.basename(filepath)
    extension = get_extension(filename, list(extensions_to_methods))
    extract_method = extensions_to_methods[extension]

    # Extract files in a temporary dir then move the extracted item back to
    # the ouput dir in order to get the details of what was extracted
    tmp_extract_dir = tempfile.mkdtemp()
    # Extract
    extract_method(filepath, output_dir=tmp_extract_dir)
    extracted_items = os.listdir(tmp_extract_dir)
    output_paths = []
    for name in extracted_items:
        extracted_path = os.path.join(tmp_extract_dir, name)
        output_path = os.path.join(output_dir, name)
        move_with_overwrite(extracted_path, output_path)
        output_paths.append(output_path)
    return output_paths


def move_with_overwrite(source_path, target_path):
    if os.path.isfile(target_path):
        os.remove(target_path)
    if os.path.isdir(target_path) and os.path.isdir(source_path):
        shutil.rmtree(target_path)
    shutil.move(source_path, target_path)


def untar(compressed_path, output_dir):
    with tarfile.open(compressed_path) as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, output_dir)


def unzip(compressed_path, output_dir):
    with zipfile.ZipFile(compressed_path, 'r') as f:
        f.extractall(output_dir)


def ungzip(compressed_path, output_dir):
    filename = os.path.basename(compressed_path)
    assert filename.endswith('.gz')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename[:-3])
    with gzip.open(compressed_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def unbz2(compressed_path, output_dir):
    extract_filename = os.path.basename(compressed_path).replace('.bz2', '')
    extract_path = os.path.join(output_dir, extract_filename)
    with bz2.BZ2File(compressed_path, 'rb') as compressed_file, open(extract_path, 'wb') as extract_file:
        for data in tqdm(iter(lambda: compressed_file.read(1024 * 1024), b'')):
            extract_file.write(data)


def get_dataset_dir(dataset):
    return DATASETS_DIR / dataset

def get_temp_filepath(create=False):
    temp_filepath = Path(tempfile.mkstemp()[1])
    if not create:
        temp_filepath.unlink()
    return temp_filepath

def get_temp_filepaths(n_filepaths, create=False):
    return [get_temp_filepath(create=create) for _ in range(n_filepaths)]

def get_data_filepath(dataset, phase, language, i=None):
    suffix = ''  # Create suffix e.g. for multiple references
    if i is not None:
        suffix = f'.{i}'
    filename = f'{dataset}.{phase}.{language}{suffix}'
    return get_dataset_dir(dataset) / filename

def add_newline_at_end_of_file(file_path):
    with open(file_path, 'r') as f:
        last_character = f.readlines()[-1][-1]
    if last_character == '\n':
        return
    logger.info(f'Adding newline at the end of {file_path}')
    with open(file_path, 'a') as f:
        f.write('\n')

def git_clone(url, output_dir, overwrite=True):
    path_name = filter(lambda x: x!='', urlparse(url).path.split("/"))
    repo_name = list(path_name)[-1]
    if Path(output_dir / repo_name).exists():
        shutil.rmtree(output_dir / repo_name)
    git.Repo.clone_from(url, output_dir / repo_name)
    logger.info('Cloning from {}...'.format(url))
    
@contextmanager
def lock_directory(dir_path):
    # TODO: Locking a directory should lock all files in that directory
    # Right now if we lock foo/, someone else can lock foo/bar.txt
    # TODO: Nested with lock_directory() should not be blocking
    assert Path(dir_path).exists(), f'Directory does not exists: {dir_path}'
    lockfile_path = get_lockfile_path(dir_path)
    with open_with_lock(lockfile_path, 'w'):
        yield

def get_lockfile_path(path):
    path = Path(path)
    if path.is_dir():
        return path / '.lockfile'
    if path.is_file():
        return path.parent / f'.{path.name}.lockfile'

@contextmanager
def open_with_lock(filepath, mode):
    with open(filepath, mode) as f:
        flock(f, LOCK_EX)
        yield f
        flock(f, LOCK_UN)

def delete_files(filepaths):
    for filepath in filepaths:
        filepath = Path(filepath)
        assert filepath.is_file()
        filepath.unlink()
        
def read_xlsx(filepath):
    return pd.read_excel(filepath).values.tolist()