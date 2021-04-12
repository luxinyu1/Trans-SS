import csv
import logging
import sys
import os
import random

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for Finetune."""

    def __init__(self, guid, text_a, text_b, masked_b):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            masked_b: string. Masked text_b.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.masked_b = masked_b

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, delimiter="\t"):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels

class ELSProcessor(DataProcessor):
    """"Processor for HanLS."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            list_b = list(text_b)
            mask_idx = line[0].split(',')

            # del the first mask to enhance the performance at top-k sampling
            # ! Maybe has negative effects
            del_flag = False
            if len(mask_idx)==2 and random.random()<0.3:
                mask_idx.pop(0)
                del_flag = True
            
            for idx in mask_idx:
                list_b[int(idx)] = '[MASK]'

            if random.random()<0.6 and not del_flag: # keep the origin complex word
                # fix the posssible missing or uncessary mask when keep the complex word
                while len(text_a) != len(list_b):
                    first_mask_idx = int(mask_idx[0])
                    if len(text_a) > len(list_b):
                        list_b.insert(first_mask_idx, '[MASK]')
                    else:
                        del list_b[first_mask_idx]
                masked_b = ''.join(list_b)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_a, masked_b=masked_b))
            else:
                masked_b = ''.join(list_b)
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, masked_b=masked_b))
        return examples

def convert_examples_to_features(examples , max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        masked_b = tokenizer.tokenize(example.masked_b)

        _truncate_seq_pair(tokens_a, tokens_b, masked_b, max_seq_length - 3)

        try:
            assert len(masked_b) == len(tokens_b)
        except:
            logger.warn("Skipping one example")
            continue

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        labels = tokens + tokens_b + ["[SEP]"]
        
        tokens += masked_b + ["[SEP]"]

        segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        labels = tokenizer.convert_tokens_to_ids(labels)

        # Only sum the loss at the masked positions
        for i in range(len(labels)):
            if input_ids[i] != tokenizer.mask_token_id:
                labels[i] = -100
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        labels += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(labels) == max_seq_length

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, labels=labels))
        
    return features

def _truncate_seq_pair(tokens_a, tokens_b, masked_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        logger.warn("Truncating one example")
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            masked_b.pop()

processors = {
    "els": ELSProcessor,
}
