python ./test.py --model-name "bart-large-pretrained" \
    --dataset-name "newsela/newsela-finetune" \
    --task-name "trans" \
    --test-dataset 'newsela' \
    --eval-all-ckpt \
    --bpe "gpt2" \
    --gpt2-encoder-json "./bpe/encoder.json" \
    --gpt2-vocab-bpe "./bpe/vocab.bpe"