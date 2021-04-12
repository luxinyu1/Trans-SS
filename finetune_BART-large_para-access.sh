HASH=_7deba6984c424c0857c3ff3dae961e13

TASK=ts-para-access

if [ "$1" != "no-preprocess" ]; then
    
    # BPE
    if [ ! -d "./${TASK}/" ]; then
        mkdir ${TASK}
    fi
    for split in 'train' 'test' 'valid'; do
        for type in 'src' 'dst'; do
            python -m access.fairseq.encoders.multiprocessing_bpe_encoder \
            --bpe-dir ./access/fairseq/encoders/resources/ \
            --inputs ./datasets/${HASH}/${HASH}.${split}.${type} \
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
      --destdir "./datasets/${HASH}/fairseq_preprocessed/" \
      --workers 60 \
      --srcdict ./access/fairseq/encoders/resources/dict.txt \
      --tgtdict ./access/fairseq/encoders/resources/dict.txt;
    
fi

# Fine-tuning

TOTAL_NUM_UPDATES=200000
WARMUP_UPDATES=500
LR=3e-06
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=./models/bart.large/model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./access_train.py "./datasets/${HASH}/fairseq_preprocessed/" \
    --restore-file $BART_PATH  \
    --max-tokens $MAX_TOKENS  \
    --task translation \
    --source-lang src --target-lang dst \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir "./logs/tensorboard/para-access/bart-large-pretrained/" \
    --save-dir "./checkpoints/para-access/bart-large-pretrained/" \
    --find-unused-parameters \
    --bpe "gpt2" \
    --gpt2-encoder-json "./access/fairseq/encoders/resources/encoder.json" \
    --gpt2-vocab-bpe "./access/fairseq/encoders/resources/vocab.bpe"