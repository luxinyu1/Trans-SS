python simplify.py --model-name 'bart-large-pretrained' \
                --path-to-ckpt './checkpoints/trans-800K/bart-large-pretrained/checkpoint15.pt' \
                --path-to-file './datasets/simplext-en/simplext-en.txt' \
                --path-to-output-file './baseline_sys_outputs/simplext_output.txt' \
                --task-name 'trans' \
                --bpe 'gpt2' \
                --gpt2-encoder-json "./bpe/encoder.json" \
                --gpt2-vocab-bpe "./bpe/vocab.bpe"