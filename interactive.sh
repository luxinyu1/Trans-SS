 fairseq-interactive ts-para-bin/ \
    --results-path ./generate-result/ \
    --source-lang "src" \
    --target-lang "dst" \
    --bpe "gpt2" \
    --path checkpoints/wikilarge/bart-base-pretrained/checkpoint3.pt \
    --beam 5 \