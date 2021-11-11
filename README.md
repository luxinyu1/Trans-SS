# Trans-SS

This repo contains codes, mined corpora, and model checkpoints for the paper "An Unsupervised Method for Building Sentence Simplification corpora in Multiple Languages".

## Mined Corpora

| Language | # Samples |                             Link                             |
| :------: | :-------: | :----------------------------------------------------------: |
| English  |  816,058  | [Download](https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/datasets/trans-1M.tar.gz) |
|  French  |  621,937  | [Download](https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/datasets/trans-fr.tar.gz) |
| Spanish  |  487,862  | [Download](https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/datasets/trans-es.tar.gz) |

## Models

| Model Architecture | Language |                             Link                             |
| :----------------: | :------: | :----------------------------------------------------------: |
|    Transformer     | English  | [Download](https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/models/transformer-en.zip) |
|      ConvS2S       | English  | [Download](https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/models/fconv-en.zip) |
|        BART        | English  | [Download](https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/models/bart-en-1.zip)<br/>[Download](https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/models/bart-en-2.zip) |

The model output files could be found in ```./sys_outputs/``` directory. 

## Requirements and Installation

This project is built with standard sentence simplification suite [EASSE](https://github.com/feralvam/easse) and sequence modeling toolkit [fairseq](https://github.com/pytorch/fairseq). Owing to that these two repo are still in fast developing,  we strongly recommend you to use the same version of packages that we use for reproducing our work.

We provide two methods to install the required dependencies:

- python==3.7
- torch==1.7.1

### Method 1: pip install

```shell
pip install -r requirements.txt
```

### Method 2: build from source

1. Download the source code of dependencies from our OSS bin:

    ```shell
    wget -O "easse-master.zip" https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/dependencies/easse-master.zip
    wget -O "fairseq.tar.gz" https://lxylab.oss-cn-shanghai.aliyuncs.com/Trans-SS/dependencies/fairseq.tar.gz
    ```

2. Build from the source code:
    ```shell
    tar -xzvf fairseq.tar.gz
    cd fairseq/
    pip install -e ./
    ```

    ```shell
    unzip easse-master.zip
    cd easse-master/
    pip install -e ./
    ```


Additionally, the C++ API of [fastBPE](https://github.com/glample/fastBPE) is also needed for word segmentation.

```shell
cd fastBPE/
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
```

You can also watch the visualized training process using [tensorboard](https://github.com/tensorflow/tensorboard).

```shell
pip install tensorboard
```

## <span id="prepare">Prepare resources</span>

Before running the mining/training scripts, please download the models and corpora that this repo depends on:

```shell
python ./prepare_resources.py
```

## Usage

This repo may still contain bugs and we are working on improving the reproductivity. Welcome to open an issue or submit a Pull Request to report/fix the bugs.

### Back Translation

Run the following scripts to back translate three multilingual NMT corpora:

```shell
./translate_de-en.sh
./translate_en-es.sh
./translate_en-fr.sh
```

Note that this process is very time-consuming, it took us several days to do back translation on a single NIVIDIA RTX 3090. The intermediate results can be download via running the resource preparation [script](#prepare).

### Extraction

Run the following python scripts to get the Sentence Simplification corpora:

```shell
python ./extract.py
python ./extract_fr.py
python ./extract_es.py
```

### Training and Testing

The training scripts function as they are named. For example, you can train a transformer model on Wikilarge via running:

```
./train_transformer_wikilarge.sh
```

and train transformer on our mined corpora via:

```
./train_transformer_trans-1M.sh
```

The checkpoints are stored in the directory ```./checkpoints/``` and logs for tensorboard can be found in ```./logs/tensorboard```.

To evaluate the trained models, you can run ```test.py``` . The testing logs will be generated in the model checkpoints directory.

```
usage: test.py [-h] --model-name MODEL_NAME --dataset-name DATASET_NAME --task-name TASK_NAME --bpe BPE [--source-lang SOURCE_LANG] [--target-lang TARGET_LANG]
               [--test-dataset TEST_DATASET] [--do-lower-case] [--eval-batch-size EVAL_BATCH_SIZE] [--num-train-epochs NUM_TRAIN_EPOCHS] [--no-cuda] [--overwrite-output-dir]
               [--show-eval-detail] [--eval-all-ckpt] [--fp16] [--tokenizer TOKENIZER] [--gpt2-encoder-json GPT2_ENCODER_JSON] [--gpt2-vocab-bpe GPT2_VOCAB_BPE]
               [--fairseq-task FAIRSEQ_TASK] [--sentencepiece-model SENTENCEPIECE_MODEL] [--bpe-codes BPE_CODES]
```


## Citation

If you find our corpora or paper useful, please consider citing:

```
@inproceedings{lu-etal-2021-unsupervised-method,
    title = "An Unsupervised Method for Building Sentence Simplification Corpora in Multiple Languages",
    author = "Lu, Xinyu  and
      Qiang, Jipeng  and
      Li, Yun  and
      Yuan, Yunhao  and
      Zhu, Yi",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.22",
    pages = "227--237",
    abstract = "The availability of parallel sentence simplification (SS) is scarce for neural SS modelings. We propose an unsupervised method to build SS corpora from large-scale bilingual translation corpora, alleviating the need for SS supervised corpora. Our method is motivated by the following two findings: neural machine translation model usually tends to generate more high-frequency tokens and the difference of text complexity levels exists between the source and target language of a translation corpus. By taking the pair of the source sentences of translation corpus and the translations of their references in a bridge language, we can construct large-scale pseudo parallel SS data. Then, we keep these sentence pairs with a higher complexity difference as SS sentence pairs. The building SS corpora with an unsupervised approach can satisfy the expectations that the aligned sentences preserve the same meanings and have difference in text complexity levels. Experimental results show that SS methods trained by our corpora achieve the state-of-the-art results and significantly outperform the results on English benchmark WikiLarge.",
}
```

## Acknowledgements

Some code in this repo is based on [access](https://github.com/facebookresearch/access). Thank for its wonderful works.
