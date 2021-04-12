model_dir=./models/wmt14.en-fr.joined-dict.transformer
subword_nmt_dir=./subword-nmt
europarl_en_fr=./data/europarl_en_fr

TASK=translate_en-fr

if [ "$1" != "no-preprocess" ]; then
  
    if [ ! -d "./${TASK}/" ]; then
        mkdir ${TASK}
    fi
    
    if [ ! -d "./${TASK}-bin/" ]; then
        mkdir ${TASK}-bin
    fi

    # apply BPE
    
    subword-nmt apply-bpe -c ${model_dir}/bpecodes < ${europarl_en_fr}/europarl-v7.fr-en.en > ${TASK}/test.bpe.en
    subword-nmt apply-bpe -c ${model_dir}/bpecodes < ${europarl_en_fr}/europarl-v7.fr-en.fr > ${TASK}/test.bpe.fr
    
    # preprocess

    fairseq-preprocess \
        -s en -t fr \
        --destdir ./${TASK}-bin/ \
        --workers 60 \
        --testpref ./${TASK}/test.bpe \
        --srcdict ${model_dir}/dict.en.txt \
        --tgtdict ${model_dir}/dict.fr.txt;

fi

# generate

fairseq-generate ./${TASK}-bin/ \
    -s en -t fr \
    --path ${model_dir}/model.pt \
    --results-path ./data/en-fr_trans_result/ \
    --bpe subword_nmt \
    --bpe-codes ${model_dir}/bpecodes \
    --tokenizer moses \
    --skip-invalid-size-inputs-valid-test \
    --beam 5 --batch-size 64 --remove-bpe