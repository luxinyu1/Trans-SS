import argparse
import logging
import os
import sys
from shutil import copyfile
import random
from tqdm import tqdm, trange
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (AutoConfig, AutoModelForMaskedLM, AutoTokenizer, AdamW, 
                        get_linear_schedule_with_warmup, 
                        CONFIG_NAME, WEIGHTS_NAME)

from access.utils.data import processors, convert_examples_to_features
from access.utils.paths import MODELS_DIR

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data-dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert-model", default=None, type=str, required=True,
                        help="BERT pre-trained model name")
    parser.add_argument("--task-name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output-dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--vocab-file", 
                        default="", 
                        type=str)
    parser.add_argument("--cache-dir",
                        default=str(MODELS_DIR),
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max-seq-length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do-train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do-eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do-lower-case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train-batch-size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning-rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam-epsilon", 
                        default=1e-8,
                        type=float, 
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight-decay",
                        default=0.0,
                        type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num-train-epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup-proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--no-cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite-output-dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--local-rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max-grad-norm", 
                        default=1.0, 
                        type=float, 
                        help="Max gradient norm.")
    parser.add_argument("--use-proportion",
                        default=1.0,
                        type=float,
                        help="the use proportion of training data.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--fp16-opt-level",
                        type=str,
                        default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss-scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only usactivated when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--adaptive-loss',
                        default=False,
                        action='store_true',
                        help="Whether to use adaptive loss")
    parser.add_argument('--log-freq',
                        type=float, default=500,
                        help='frequency of logging loss')
    parser.add_argument("--eval-file", 
                        default=None, 
                        type=str, 
                        help="The path to eval dataset")
    parser.add_argument("--show-eval-detail",
                        default=False,
                        action='store_true',
                        help="Show the intermediate results of evaluation")
    parser.add_argument("--save-all",  
                        default=False,
                        action="store_true",
                        help="Wether to save all models")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank!=-1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    if args.vocab_file:
        tokenizer = AutoTokenizer(args.vocab_file, args.do_lower_case)
    elif args.cache_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.cache_dir, do_lower_case=args.do_lower_case)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.cache_dir:
        config = AutoConfig.from_pretrained(args.cache_dir)
        config.is_decoder = False
        model = AutoModelForMaskedLM.from_pretrained(args.cache_dir, config=config)
    else:
        config = AutoConfig.from_pretrained(args.bert_model)
        config.is_decoder = False
        model = AutoModelForMaskedLM.from_pretrained(args.bert_model, config=config)
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0

    if args.do_train:
        # Prepare data loader

        train_examples = processor.get_train_examples(args.data_dir)
        logger.info("%d examples in total"%(len(train_examples)))
        if args.use_proportion < 1.0:
            train_examples_num = int(args.use_proportion*len(train_examples))
            train_examples = train_examples[:train_examples_num:]
        elif args.use_proportion == 1.0:
            pass
        else:
            raise ValueError('use_proportion should be <= 1.0')
        logger.info("%d examples used"%(len(train_examples)))
        cached_train_features_file = os.path.join(args.data_dir, 'train_{0}_{1}_{2}_{3}'.format(
            list(filter(None, args.bert_model.split('/'))).pop(),
                        str(args.max_seq_length),
                        str(task_name),
                        str(args.use_proportion)))
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.long)
        
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                "weight_decay": 0.0
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_proportion*num_train_optimization_steps),
            num_training_steps=num_train_optimization_steps
        )

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

        os.makedirs(os.path.join(args.output_dir, "all_models"), exist_ok=True)
        model.train()
        model.zero_grad()

        average_loss = 0.0 # averaged loss every args.log_freq steps
        
        for e in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, labels = batch

                if args.adaptive_loss:
                    logits = model(input_ids, segment_ids, input_mask)[0]
                    loss = compute_loss(logits, labels)
                else:
                    outputs = model(input_ids, segment_ids, input_mask, labels=labels) # loss, logits
                    loss = outputs[0]
                
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                average_loss += loss.item()

                if (step + 1) % (args.log_freq * args.gradient_accumulation_steps) == 0:
                    logger.info("Global Step:{} Average Loss = {} Step Loss = {} LR {}".format(global_step, average_loss / args.log_freq, 
                                                                                loss.item(), optimizer.param_groups[0]['lr']))
                    average_loss = 0.0

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

            # save each epoch
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, "all_models",  "e{}_{}".format(e, WEIGHTS_NAME))
            torch.save(model_to_save.state_dict(), output_model_file)

            torch.cuda.empty_cache()

    ### Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        output_args_file = os.path.join(args.output_dir, 'training_args.bin')
        torch.save(args, output_args_file)
    elif args.cache_dir:
        model = AutoModelForMaskedLM.from_pretrained(args.cache_dir)
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.bert_model)

	### Evaluation
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if args.eval_file==None:
            raise ValueError("Argument eval_file is not defined.")

        import OpenHowNet
        import gensim
        from bert_generate import generate
        from substitute_selection import substitute_selection
        from substitute_ranking import read_dataset, read_dict, substitute_ranking
        from experiment import read_manual_data, evaluate_SS_scores, evaluate_pipeline_scores

        hownet = OpenHowNet.HowNetDict(use_sim=True)
        model_word2vector = gensim.models.KeyedVectors.load_word2vec_format('./model/merge_sgns_bigram_char300.txt', binary=False)

        best_F_score = 0
        best_accuracy = 0
        best_F_score_epoch = 0
        best_accuracy_epoch = 0
        val_res_file = os.path.join(args.output_dir, "valid_results.txt")
        val_f = open(val_res_file, "w")
        logger.info("***** Dev Eval results *****")
        for e in range(int(args.num_train_epochs)):
            weight_path = os.path.join(args.output_dir, "all_models", "e{}_{}".format(e, WEIGHTS_NAME))

            model.load_state_dict(torch.load(weight_path))
            model.to(device)

            if args.vocab_file:
                tokenizer = AutoTokenizer(args.vocab_file, args.do_lower_case)
            elif args.cache_dir:
                tokenizer = AutoTokenizer.from_pretrained(args.cache_dir, do_lower_case=args.do_lower_case)
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

            logger.info("***** Running evaluation *****")

            # Generate
            logger.info("Start Generating...")
            results = generate(model, tokenizer, args.eval_file, args.max_seq_length)
            logger.info("Generated")
            # Select
            logger.info("Start Selecting...")
            row_lines, source_sentences, source_words = read_dataset(args.eval_file)
            word_freq_dict = read_dict('./dict/modern_chinese_word_freq.txt')
            valid_results = substitute_selection(results, word_freq_dict, source_words)
            logger.info("Selected")
            # Rank
            logger.info("Start Ranking...")
            if args.cache_dir:
                config = AutoConfig.from_pretrained(args.cache_dir)
                config.is_decoder = False
                bert_base_chinese = AutoModelForMaskedLM.from_pretrained(args.cache_dir, config=config)
            else:
                config = AutoConfig.from_pretrained(args.bert_model)
                config.is_decoder = False
                bert_base_chinese = AutoModelForMaskedLM.from_pretrained(args.bert_model, config=config)
            bert_base_chinese.to(device)
            substitution_num = 10
            bert_pre_words = []
            ss = []
            for row_line, source_sentence, source_word, result in zip(row_lines, source_sentences, source_words, valid_results):
                bert_pre_word, bert_ss_sorted = substitute_ranking(row_line, model_word2vector, bert_base_chinese, tokenizer, hownet, source_sentence, source_word, result, word_freq_dict, substitution_num)
                bert_pre_words.append(bert_pre_word)
                ss.append(bert_ss_sorted)
            assert len(bert_pre_words) == len(ss)
            
            if args.show_eval_detail:
                logger.info('\n'.join(' '.join(x) for x in valid_results))
                logger.info('\n'.join(bert_pre_words))
            # Eval
            _, manual_labels = read_manual_data(args.eval_file)
            potential, precision_A, recall, F_score = evaluate_SS_scores(valid_results, manual_labels)
            logger.info("[Batch {}] potential: {:.4f}, precision: {:.4f}, recall: {:.4f}, F_score: {:.4f}".format(e, potential, precision_A, recall, F_score))
            val_f.write("[Batch {}] potential: {:.4f}, precision: {:.4f}, recall: {:.4f}, F_score: {:.4f}\n".format(e, potential, precision_A, recall, F_score))

            if F_score > best_F_score:
                best_F_score = F_score
                best_F_score_epoch = e

            precision_B, accuracy, changed_proportion = evaluate_pipeline_scores(bert_pre_words, source_words, manual_labels)
            logger.info("[Batch {}] precision: {:.4f}, accuracy: {:.4f}, changed_proportion: {:.4f}".format(e, precision_B, accuracy, changed_proportion))
            val_f.write("[Batch {}] precision: {:.4f}, accuracy: {:.4f}, changed_proportion: {:.4f}\n".format(e, precision_B, accuracy, changed_proportion))
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_epoch = e

            torch.cuda.empty_cache()

        logger.info("\nBest epoch: {:d}. Best val F1: {:.4f}".format(best_F_score_epoch, best_F_score))
        logger.info("\nBest epoch: {:d}. Best val accuracy: {:.4f}".format(best_accuracy_epoch, best_accuracy))
        val_f.write("Best F1 epoch: {:d}. Best val F1: {:.4f}\n".format(best_F_score_epoch, best_F_score))
        val_f.write("Best accuracy epoch: {:d}. Best val accuracy: {:.4f}\n".format(best_accuracy_epoch, best_accuracy))
        val_f.close()

        if best_accuracy_epoch==best_F_score_epoch:
            best_weight_path = os.path.join(args.output_dir, "all_models", "e{}_{}".format(best_accuracy_epoch, WEIGHTS_NAME))
            best_model_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            os.system("cp {} {}/{}".format(best_weight_path, best_model_dir, WEIGHTS_NAME))
            with open(os.path.join(best_model_dir, CONFIG_NAME), 'w') as f:
                f.write(model.config.to_json_string())
            tokenizer.save_vocabulary(os.path.join(best_model_dir, 'vocab.txt'))
        else:
            # best_accuracy
            best_accuracy_path = os.path.join(args.output_dir, "all_models", "e{}_{}".format(best_accuracy_epoch, WEIGHTS_NAME))
            best_accuracy_dir = os.path.join(args.output_dir, "best_model", "best_accuracy")
            os.makedirs(best_accuracy_dir)
            os.system("cp {} {}/{}".format(best_accuracy_path, best_accuracy_dir, WEIGHTS_NAME))
            with open(os.path.join(best_accuracy_dir, CONFIG_NAME), 'w') as f:
                f.write(model.config.to_json_string())
            tokenizer.save_vocabulary(os.path.join(best_accuracy_dir, 'vocab.txt'))
            # best_F1
            best_F_score_path = os.path.join(args.output_dir, "all_models", "e{}_{}".format(best_F_score_epoch, WEIGHTS_NAME))
            best_F_score_dir = os.path.join(args.output_dir, "best_model", "best_F1_score")
            os.makedirs(best_F_score_dir)
            os.system("cp {} {}/{}".format(best_F_score_path, best_F_score_dir, WEIGHTS_NAME))
            with open(os.path.join(best_F_score_dir, CONFIG_NAME), 'w') as f:
                f.write(model.config.to_json_string())
            tokenizer.save_vocabulary(os.path.join(best_F_score_dir, 'vocab.txt'))

        if not args.save_all:
            os.system("rm -r {}".format(os.path.join(args.output_dir, "all_models")))

if __name__ == '__main__':
    main()
