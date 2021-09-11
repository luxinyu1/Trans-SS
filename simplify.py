import os
import sys
import math
import tempfile
import torch
import random
import time
import argparse
import logging
from pathlib import Path
import numpy as np
from shutil import copyfile

from utils.paths import CHECKPOINT_DIR, REPO_DIR
from utils.utils import get_data_filepath, get_dataset_dir, read_lines, create_directory_or_skip
from easse.report import get_all_scores, get_html_report
from fairseq.data import encoders
from fairseq_cli.interactive import buffered_read, make_batches
from fairseq.dataclass.configs import CheckpointConfig, FairseqConfig
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_cli.generate import get_symbols_to_strip_from_output

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

def set_random_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model-name", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Model name")
    parser.add_argument("--path-to-ckpt", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="The path to the ckpt to be used.")
    parser.add_argument("--path-to-file",
                        default=None,
                        type=str,
                        required=True,
                        help="The path to the file to be simplified.")
    parser.add_argument("--path-to-output-file",
                        default=None,
                        type=str,
                        required=True,
                        help="The path to the output file.")
    parser.add_argument("--task-name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of task(not fairseq task).")
    parser.add_argument("--bpe",
                        default=None,
                        type=str,
                        required=True,
                        help="Possible choices: gpt2, bytes, sentencepiece, subword_nmt, byte_bpe, characters, bert, fastbpe, hf_byte_bpe")

    ## Other parameters
    parser.add_argument("--source-lang",
                        default="src",
                        type=str,
                        required=False)
    parser.add_argument("--target-lang",
                        default="dst",
                        type=str,
                        required=False)
    parser.add_argument("--fp16", 
                        default=False, 
                        action='store_true',
                        help="use FP16.")
    parser.add_argument("--tokenizer",
                        default="nltk",
                        type=str,
                        help="Possible choices: nltk, space, moses")
    parser.add_argument("--no-cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpt2-encoder-json",
                        default=None,
                        type=str)
    parser.add_argument("--gpt2-vocab-bpe",
                        default=None,
                        type=str)
    parser.add_argument("--fairseq-task",
                        default="translation",
                        type=str,
                        required=False,
                        help="The name of fairseq task.")
    parser.add_argument("--sentencepiece-model",
                        type=str,
                        required=False,
                        help="The path to mBART sentencepiece model.")
    parser.add_argument("--bpe-codes",
                        type=str,
                        required=False,
                        help="The path to fastBPE bpecodes.")

    args = parser.parse_args()
    
    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        device = torch.device("cpu")
    
    weight_path = args.path_to_ckpt
    
    logger.info("Using checkpoint: \"{}\"".format(args.path_to_ckpt))
    
    args.data = str(REPO_DIR) + "/ts-" + args.task_name + "-bin/"
    args.task = args.fairseq_task
    if args.model_name == "mBART":
        args.langs = "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN"
    
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    fp16 = args.fp16
    
    cfg = convert_namespace_to_omegaconf(args)

    pred_filepath = args.path_to_output_file

    """Loads an ensemble of models.

    Args:
        filenames (List[str]): checkpoint files to load
        arg_overrides (Dict[str,Any], optional): override model args that
            were used during model training
        task (fairseq.tasks.FairseqTask, optional): task to use for loading
    """

    task = tasks.setup_task(cfg.task)

    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(weight_path),
        task=task,
    )

     # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if fp16:
            model.half()
        if use_cuda:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.task)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(cfg.tokenizer)
    bpe = encoders.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    start_id = 0

    for inputs in buffered_read(args.path_to_file, buffer_size=10):
        results = []
        for batch in make_batches(inputs, cfg, task, max_positions, encode_fn):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()
            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths,
                },
            }
            translations = task.inference_step(
                generator, models, sample, constraints=constraints
            )
            list_constraints = [[] for _ in range(bsz)]
            if cfg.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                constraints = list_constraints[i]
                results.append(
                    (
                        start_id + id,
                        src_tokens_i,
                        hypos,
                        {
                            "constraints": constraints,
                        },
                    )
                )
        with open(pred_filepath, 'a+') as f_pred:
            # sort output to match input order
            for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
                    detok_src_str = decode_fn(src_str)

                # Process top predictions
                for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=cfg.common_eval.post_process,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                    )
                    detok_hypo_str = decode_fn(hypo_str)
                    f_pred.write(detok_hypo_str + '\n')

        # update running id_ counter
        start_id += len(inputs)
                                                   
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()