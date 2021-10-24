from utils.utils import get_data_filepath, read_lines
from utils.paths import DATA_DIR

# check if the training set contains the testing set samples

turk_complex_path = get_data_filepath('turkcorpus', 'test', 'complex')

test_set = read_lines(turk_complex_path)

with open(DATA_DIR / 'complex_trans.csv', 'r', encoding='utf-8') as f_train:

    for line in f_train:
        assert line.strip() not in test_set

print("Training set doesn't contain any of the testing set samples!")