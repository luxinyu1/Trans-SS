model_dir=./models/wmt19.de-en.joined-dict.ensemble

TASK=translate_de-en

if [ "$1" != "no-preprocess" ]; then
  # BPE
  
    if [ ! -d "./${TASK}/" ]; then
        mkdir ${TASK}
    fi

    cd fastBPE
    ./fast applybpe ../${TASK}/test.bpe.de ../data/wmt_en_de/train.de ../models/wmt19.de-en.joined-dict.ensemble/bpecodes
    ./fast applybpe ../${TASK}/test.bpe.en ../data/wmt_en_de/train.en ../models/wmt19.de-en.joined-dict.ensemble/bpecodes
    cd ..

    # preprocess

    fairseq-preprocess \
        -s de -t en \
        --destdir ./${TASK}-bin/ \
        --workers 60 \
        --testpref ./${TASK}/test.bpe \
        --srcdict ${model_dir}/dict.de.txt \
        --tgtdict ${model_dir}/dict.en.txt;

fi

# generate

fairseq-generate ./${TASK}-bin/ \
    -s de -t en \
    --path ${model_dir}/model1.pt:${model_dir}/model2.pt:${model_dir}/model3.pt:${model_dir}/model4.pt  \
    --results-path ./data/de-en_trans_result/ \
    --bpe fastbpe \
    --bpe-codes ${model_dir}/bpecodes \
    --tokenizer moses \
    --skip-invalid-size-inputs-valid-test \
    --beam 5 --batch-size 64 --remove-bpe