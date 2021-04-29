if [ ! -d "./datasets/trans-es-1M/" ]; then
    python ./split.py --use-num 1000000 \
        --output-dir './datasets/trans-es-1M/' \
        --dataset 'trans_es'
fi

TASK=ts-trans-es

if [ "$1" != "no-preprocess" ]; then
    
    # BPE
    if [ ! -d "./${TASK}/" ]; then
        mkdir ${TASK}
    fi
    
    ./fastBPE/fast learnbpe 50000 ./datasets/trans-es-1M/trans_es.train.src ./datasets/trans-es-1M/trans_es.train.dst > ./bpe/bpecodes50000-es
    
    for split in 'train' 'test' 'valid'; do
        for type in 'src' 'dst'; do
            ./fastBPE/fast applybpe ./${TASK}/${split}.bpe.${type} ./datasets/trans-es-1M/trans_es.${split}.${type} ./bpe/bpecodes50000-es
        done
    done
    
    # preprocess

    fairseq-preprocess \
      --source-lang "src" \
      --target-lang "dst" \
      --trainpref "${TASK}/train.bpe" \
      --testpref "${TASK}/test.bpe" \
      --validpref "${TASK}/valid.bpe" \
      --destdir "${TASK}-bin/" \
      --workers 60 \
      --joined-dictionary \
    
fi

TOTAL_NUM_UPDATES=200000
WARMUP_UPDATES=1000
LR=3e-04
MAX_TOKENS=2048
UPDATE_FREQ=1

# Training
CUDA_VISIBLE_DEVICES=0 python ./train_es.py ${TASK}-bin/ \
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
    --arch transformer --save-dir './checkpoints/trans-es/transformer/' --optimizer adam \
    --tensorboard-logdir "./logs/tensorboard/trans-es/transformer/" \
    --skip-invalid-size-inputs-valid-test \
    --bpe "fastbpe" \
    --bpe-codes "./bpe/bpecodes50000-es"