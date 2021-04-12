import os
import sys
import math
import tempfile
import torch
import time
import argparse
import logging
from pathlib import Path

from access.utils.paths import CHECKPOINT_DIR, REPO_DIR
from access.utils.utils import get_data_filepath, get_dataset_dir, read_lines, create_directory_or_skip
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

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model-name", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="Model name")
    parser.add_argument("--dataset-name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the dataset checkpoints trained on.")
    parser.add_argument("--task-name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of task.")
    parser.add_argument("--bpe",
                        default=None,
                        type=str,
                        required=True,
                        help="Possible choices: gpt2, bytes, sentencepiece, subword_nmt, byte_bpe, characters, bert, fastbpe, hf_byte_bpe")

    ## Other parameters
    parser.add_argument("--eval-dataset", 
                        default="turk", 
                        type=str,
                        required=False,
                        help="The name of eval dataset")
    parser.add_argument("--do-lower-case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval-batch-size",
                        default=32,
                        type=int,
                        help="Total batch size for evaluating.")
    parser.add_argument("--num-train-epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no-cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite-output-dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--show-eval-detail",
                        default=False,
                        action='store_true',
                        help="Show the intermediate results of evaluation")
    parser.add_argument("--eval-all-ckpt", 
                        default=False, 
                        action='store_true',
                        help="Eval all the ckpt in dir.")
    parser.add_argument("--fp16", 
                        default=False, 
                        action='store_true',
                        help="use FP16.")
    parser.add_argument("--tokenizer",
                        default="nltk",
                        type=str,
                        help="Possible choices: nltk, space, moses")
    parser.add_argument("--gpt2-encoder-json",
                        default=None,
                        type=str)
    parser.add_argument("--gpt2-vocab-bpe",
                        default=None,
                        type=str)
    

    args = parser.parse_args()
    
    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        device = torch.device("cpu")
    
    weight_dir = Path(CHECKPOINT_DIR) / args.dataset_name / args.model_name
    report_dir = weight_dir / 'html_reports'
    
    log_path = weight_dir / 'val.log'
    
    logger.info("Now searching: \"{}\"".format(weight_dir))
    
    if os.path.isdir(weight_dir):
        ckpt_list = os.listdir(weight_dir)
        if os.path.isfile(weight_dir / 'val.log'):
            ckpt_list.remove('val.log')
        if os.path.isdir(report_dir):
            ckpt_list.remove('html_reports')
        if args.eval_all_ckpt:
            ckpt_list.remove('checkpoint_best.pt')
            ckpt_list.remove('checkpoint_last.pt')
            logger.info("{} checkpoints to be evaluated.".format(len(ckpt_list)))
        else:
            ckpt_list = ['model_best.pt']
            logger.info("Only the best ckpt will be evaluated.")
    else:
        raise ValueError("Please check model_name and dataset_name!")
    
    args.data = str(REPO_DIR) + "/ts-" + args.task_name + "-bin/"
    args.task = "translation"
    args.source_lang = "src"
    args.target_lang = "dst"
    
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    fp16 = args.fp16
    
    cfg = convert_namespace_to_omegaconf(args)

    if args.eval_all_ckpt:    
        for e in range(1, len(ckpt_list)+1):
            weight_path = os.path.join(weight_dir, "checkpoint{}.pt".format(e))
            complex_filepath = get_data_filepath('turkcorpus', 'valid', 'complex')
            pred_filepath = tempfile.mkstemp()[1]
            
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

            for inputs in buffered_read(complex_filepath, buffer_size=10):
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
                
        
            ref_filepaths = [get_data_filepath('turkcorpus', 'valid', 'simple.turk', i) for i in range(8)]
            turk_scores = get_all_scores(orig_sents=read_lines(complex_filepath), sys_sents=read_lines(pred_filepath), refs_sents=[read_lines(ref_filepath) for ref_filepath in ref_filepaths])
            ref_filepaths = [get_data_filepath('asset', 'valid', 'simp', i) for i in range(10)]
            asset_scores = get_all_scores(orig_sents=read_lines(complex_filepath), sys_sents=read_lines(pred_filepath), refs_sents=[read_lines(ref_filepath) for ref_filepath in ref_filepaths])
            report = get_html_report(orig_sents=read_lines(complex_filepath), sys_sents=read_lines(pred_filepath), refs_sents=[read_lines(ref_filepath) for ref_filepath in ref_filepaths], test_set=args.eval_dataset)
            with create_directory_or_skip(report_dir):
                pass
            with open(report_dir / (str(e)+'.html'), 'w+', encoding='utf-8') as f_html:
                f_html.write(report + '\n')
            logger.info("[Epoch {}][turk] {}".format(e, turk_scores))
            logger.info("[Epoch {}][asset] {}".format(e, asset_scores))
            with open(log_path,'a+',encoding='utf-8') as f_log:
                f_log.write("[Epoch {}][turk] {}\n".format(e, turk_scores))
                f_log.write("[Epoch {}][asset] {}\n".format(e, asset_scores))
            
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()