from utils.utils import get_data_filepath, get_dataset_dir, read_lines
from utils.paths import DATASETS_DIR, DATA_DIR

# check if the training set contains the test set samples

turk_complex_path = get_data_filepath('turkcorpus', 'test', 'complex')

test_set = read_lines(turk_complex_path)

with open(DATA_DIR / 'complex_trans.csv', 'r', encoding='utf-8') as f_train:

    for line in f_train:
        if line.strip() in test_set:
            print(line)