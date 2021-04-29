if [ ! -d "./datasets/trans-es-1M/" ]; then
    python ./split.py --use-num 1000000 \
        --output-dir './datasets/trans-es-1M/' \
        --dataset 'trans_es'
fi

TASK=ts-trans-es-mBART
mBART_DIR=./models/mbart.cc25.v2
DATASET_DIR=./datasets/trans-es-1M

if [ "$1" != "no-preprocess" ]; then
    
    # BPE
    if [ ! -d "./${TASK}/" ]; then
        mkdir ${TASK}
    fi

    for split in 'train' 'test' 'valid'; do
        python ./access/utils/spm_encode.py \
            --model=${mBART_DIR}/sentence.bpe.model \
            --input=${DATASET_DIR}/trans_es.${split}.src \
            --output=${TASK}/${split}.spm.src
        python ./access/utils/spm_encode.py \
            --model=${mBART_DIR}/sentence.bpe.model \
            --input=${DATASET_DIR}/trans_es.${split}.dst \
            --output=${TASK}/${split}.spm.es_XX     
    done

    fairseq-preprocess \
        --source-lang src \
        --target-lang es_XX \
        --trainpref ${TASK}/train.spm \
        --validpref ${TASK}/valid.spm \
        --testpref ${TASK}/test.spm \
        --destdir ${TASK}-bin \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --srcdict ${mBART_DIR}/dict.txt \
        --tgtdict ${mBART_DIR}/dict.txt \
        --workers 70 \
        --bpe 'sentencepiece' \

fi

# Fine-tuning

TOTAL_NUM_UPDATES=1000000
WARMUP_UPDATES=2500
LR=3e-04
MAX_TOKENS=1024
UPDATE_FREQ=2
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./train_es.py ${TASK}-bin/ \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large --layernorm-embedding \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --task translation_from_pretrained_bart \
    --source-lang src --target-lang es_XX \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 1024 --update-freq 2 \
    --restore-file ${mBART_DIR}/model.pt  \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --bpe 'sentencepiece' --sentencepiece-model $mBART_DIR/sentence.bpe.model \
    --save-dir "./checkpoints/trans-es/mBART/" \
    --max-epoch 10 \
    --langs $langs \