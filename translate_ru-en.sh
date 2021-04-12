model_dir=./models/wmt19.ru-en.ensemble

TASK=translate_ru-en

if [ "$1" != "no-preprocess" ]; then
  # BPE
  
    if [ ! -d "./${TASK}/" ]; then
        mkdir ${TASK}
    fi

    cd fastBPE
    ./fast applybpe ../${TASK}/test.bpe.ru ../data/en-ru/UNv1.0.en-ru.ru ../models/wmt19.ru-en.ensemble/bpecodes
    ./fast applybpe ../${TASK}/test.bpe.en ../data/en-ru/UNv1.0.en-ru.en ../models/wmt19.ru-en.ensemble/bpecodes
    cd ..

    # preprocess

    fairseq-preprocess \
        -s ru -t en \
        --destdir ./${TASK}-bin/ \
        --workers 60 \
        --testpref ./${TASK}/test.bpe \
        --srcdict ${model_dir}/dict.ru.txt \
        --tgtdict ${model_dir}/dict.en.txt;

fi

# generate

fairseq-generate ./${TASK}-bin/ \
    -s ru -t en \
    --path ${model_dir}/model1.pt:${model_dir}/model2.pt:${model_dir}/model3.pt:${model_dir}/model4.pt  \
    --results-path ./data/ \
    --bpe fastbpe \
    --bpe-codes ${model_dir}/bpecodes/ru-en_trans_result/ \
    --tokenizer moses \
    --skip-invalid-size-inputs-valid-test \
    --beam 5 --batch-size 64 --remove-bpe