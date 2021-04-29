if [ ! -d "./datasets/trans-1M/" ]; then
    python ./split.py --use-num 1000000 \
        --output-dir './datasets/trans-1M/' \
        --dataset 'trans'
fi

# wget -P './bpe' -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
# wget -P './bpe' -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
# wget -P './bpe' -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

TASK=ts-trans-levenshtein

if [ "$1" != "no-preprocess" ]; then
    
    # BPE
    if [ ! -d "./${TASK}/" ]; then
        mkdir ${TASK}
    fi
#     for split in 'train' 'test' 'valid'; do
#         for type in 'src' 'dst'; do
#             python -m bpe.multiprocessing_bpe_encoder \
#             --encoder-json ./bpe/encoder.json \
#             --vocab-bpe ./bpe/vocab.bpe \
#             --inputs ./datasets/trans-1M/trans.${split}.${type} \
#             --outputs ./${TASK}/${split}.bpe.${type} \
#             --workers 60 \
#             --keep-empty;
#         done
#     done
    
    # preprocess

    fairseq-preprocess \
      --source-lang "src" \
      --target-lang "dst" \
      --trainpref "./datasets/trans-1M/trans.train" \
      --testpref "./datasets/trans-1M/trans.test" \
      --validpref "./datasets/trans-1M/trans.valid"  \
      --destdir "${TASK}-bin/" \
      --workers 60 \
      --joined-dictionary \
    
fi

CUDA_VISIBLE_DEVICES=0 python ./train.py \
    ${TASK}-bin/ \
    --save-dir ./checkpoints/trans-1M/levenshtein-transformer/ \
    --ddp-backend=no_c10d \
    --task translation_lev \
    --criterion nat_loss \
    --arch levenshtein_transformer \
    --noise random_delete \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --fixed-validation-seed 7 \
    --max-tokens 8000 \
    --save-interval-updates 10000 \
    --max-update 300000 \
    --cpu