python ./test.py --model-name "mBART" \
    --source-lang src --target-lang dst \
    --dataset-name "trans-fr" \
    --task-name "trans-fr-mBART" \
    --fairseq-task "translation_from_pretrained_bart" \
    --test-dataset 'alector' \
    --eval-all-ckpt \
    --bpe 'sentencepiece' \
    --sentencepiece-model ./models/mbart.cc25.v2/sentence.bpe.model \