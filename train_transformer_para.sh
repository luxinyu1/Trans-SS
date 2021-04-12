if [ ! -d "./datasets/para/" ]; then
    python ./split.py --use-num 1000000 \
        --output-dir './datasets/para/' \
        --dataset 'para'
fi

wget -P './bpe' -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -P './bpe' -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -P './bpe' -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

TASK=ts-para

if [ "$1" != "no-preprocess" ]; then
    
    # BPE
    if [ ! -d "./${TASK}/" ]; then
        mkdir ${TASK}
    fi
    for split in 'train' 'test' 'valid'; do
        for type in 'src' 'dst'; do
            python -m bpe.multiprocessing_bpe_encoder \
            --encoder-json ./bpe/encoder.json \
            --vocab-bpe ./bpe/vocab.bpe \
            --inputs ./datasets/para/para.${split}.${type} \
            --outputs ./${TASK}/${split}.bpe.${type} \
            --workers 60 \
            --keep-empty;
        done
    done
    
    # preprocess

    fairseq-preprocess \
      --source-lang "src" \
      --target-lang "dst" \
      --trainpref "${TASK}/train.bpe" \
      --testpref "${TASK}/test.bpe" \
      --validpref "${TASK}/valid.bpe"  \
      --destdir "${TASK}-bin/" \
      --workers 60 \
      --srcdict ./bpe/dict.txt \
      --tgtdict ./bpe/dict.txt;
    
fi

TOTAL_NUM_UPDATES=240000
WARMUP_UPDATES=1000
LR=3e-04
MAX_TOKENS=2048
UPDATE_FREQ=1

# Training
CUDA_VISIBLE_DEVICES=0 python ./train.py ${TASK}-bin/ \
    --lr $LR --clip-norm 0.1 --dropout 0.1 --max-tokens $MAX_TOKENS \
    --lr-scheduler polynomial_decay \
    --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch 15 \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --arch transformer --save-dir './checkpoints/para/transformer/' --optimizer adam \
    --tensorboard-logdir "./logs/tensorboard/para/transformer/" \
    --skip-invalid-size-inputs-valid-test \
    --bpe "gpt2" \
    --gpt2-encoder-json "./bpe/encoder.json" \
    --gpt2-vocab-bpe "./bpe/vocab.bpe"