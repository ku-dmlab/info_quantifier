#!/bin/bash

info=/workspace/fairseq/examples/info_quantizer
agent_train(){
    CUDA_VISIBLE_DEVICES=$1 fairseq-train \
        ${data_dir} \
        -s src -t trg \
        --clip-norm 5 \
        --user-dir $info \
        --max-epoch 40  \
        --lr 8e-4 \
        --dropout 0.4 \
        --lr-scheduler fixed \
        --optimizer adam \
        --arch info_linear \
        --task info_quantizer \
        --criterion info_loss \
        --agent-arch softplus2 \
        --agent-loss relu2 \
        --update-freq 4 \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 600 \
        --ddp-backend=no_c10d \
        --hf-model \
        --report-accuracy \
        --lr-shrink 0.95 \
        --force-anneal 4 \
        --save-dir ${save_dir} \
        --tensorboard-logdir ${save_dir}/logs \
        --no-epoch-checkpoints \
        --use-transformer-feature \
        --length-loss-coef 0.3 \
        --layer-size normal
}

data_dir=/workspace/data/generated_data/ko-en-marian/bin-data
save_dir=/workspace/model/iq/ko-en
agent_train 0,1,2,3

wait

agent_path=/workspace/fairseq/SimulEval/simuleval/user_agent
test_src=/workspace/data/opus.spm_tokenized.ko-en/test.ko
test_tgt=/workspace/data/opus.spm_tokenized.ko-en/test.en
dict=/workspace/data/iwslt17.spm_tokenized.ko-en/dict.en.txt.no_symbol
output_path=simulresult/ko-en-marian
simuleval(){
    CUDA_VISIBLE_DEVICES=$1 simuleval \
        --user-dir $info \
        --task info_quantizer \
        --source $test_src \
        --target $test_tgt \
        --eval-latency-unit word \
        --agent $agent_path/info_agent_trg_marian.py \
        --agent-arch softplus2 \
        --policy-model-path $save_dir/checkpoint_best.pt \
        --dict-path $dict \
        --beam-size 1 \
        --threshold $2 \
        --source-type text \
        --target-type text \
        --latency-metrics AL LAAL AP DAL \
        --quality-metrics BLEU \
        --sacrebleu-tokenizer 13a \
        --bpe sentencepiece \
        --device gpu \
        --output $output_path/threshold_$2
}

simuleval 0 0 &
simuleval 1 0.5 &
simuleval 2 1 &
simuleval 3 1.5 &
simuleval 1 2 &
simuleval 2 2.5 &
simuleval 3 3 &
simuleval 0 3.5 &
simuleval 0 4 &