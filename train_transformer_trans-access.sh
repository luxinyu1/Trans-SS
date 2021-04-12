HASH=_2c3e6cd772e540fb1ad4b23f1bb94de6

# Train

WARMUP_UPDATES=4000
LR=0.00011
MAX_TOKENS=5000
MAX_EPOCH=100

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./access_train.py ./datasets/${HASH}/fairseq_preprocessed/ \
    --max-tokens $MAX_TOKENS  \
    --task translation \
    --source-lang src --target-lang dst \
    --truncate-source \
    --layernorm-embedding \
    --arch transformer \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.54 \
    --dropout 0.2 \
    --weight-decay 0.01 --optimizer adam  --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr ${LR} \
    --lr-scheduler fixed \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir "./logs/tensorboard/trans-access/transformer/" \
    --save-dir "./checkpoints/trans-access/transformer/"