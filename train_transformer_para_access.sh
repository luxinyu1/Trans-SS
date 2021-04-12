HASH=_fe1a35a68ee6d01893a64d79b661dfee

# Train

WARMUP_UPDATES=4000
LR=0.00011
MAX_TOKENS=5000
MAX_EPOCH=100

# Training
CUDA_VISIBLE_DEVICES=0 python ./access_train.py ./datasets/${HASH}/fairseq_preprocessed \
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
    --tensorboard-logdir "./logs/tensorboard/para-access/transformer/" \
    --save-dir "./checkpoints/para-access/transformer/"