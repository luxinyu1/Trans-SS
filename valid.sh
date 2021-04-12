python ./valid.py --model-name "transformer" \
    --dataset-name "trans-wofkgl" \
    --task-name "trans" \
    --eval-all-ckpt \
    --bpe "gpt2" \
    --gpt2-encoder-json "./bpe/encoder.json" \
    --gpt2-vocab-bpe "./bpe/vocab.bpe"