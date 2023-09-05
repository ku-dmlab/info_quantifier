
run_nmt_train_en_de(){
    CUDA_VISIBLE_DEVICES=$1 fairseq-train \
        $data_path \
        -s $2 -t $3 \
        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --save-dir $save_path/$2-$3 \
        --tensorboard-logdir $save_path/$2-$3/logs \
        --max-epoch 5 \
        --no-epoch-checkpoints
}

data_path=/workspace/data/iwslt14.tokenized.de-en
save_path=/workspace/model/offline_nmt
run_nmt_train_en_de 0,1 en de &
run_nmt_train_en_de 2,3 de en

run_nmt_train_en_vi(){
    CUDA_VISIBLE_DEVICES=$1 fairseq-train \
        $data_path \
        -s $2 -t $3 \
        --arch transformer_wmt_en_de --share-decoder-input-output-embed \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --no-epoch-checkpoints \
        --save-dir $data_path/$2-$3 \
        --tensorboard-logdir $data_path/$2-$3/logs \
        --update-freq 4 --max-epoch 5
}

data_path=/workspace/data/iwslt15.tokenized.vi-en
run_nmt_train_en_vi 0,1 en vi &
run_nmt_train_en_vi 2,3 vi en


run_nmt_train_en_ko(){
    CUDA_VISIBLE_DEVICES=$1 fairseq-train \
        $data_path \
        -s $2 -t $3 \
        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --dropout 0.3 --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 4096 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok $4 \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --tensorboard-logdir $data_path/$2-$3/logs \
        --save-dir $data_path/$2-$3 \
        --max-epoch 5 \
        --no-epoch-checkpoints
}

# data_path=/workspace/data/opus.spm_tokenized.ko-en/data-bin
data_path=/workspace/data/iwslt17.spm_tokenized.ko-en
run_nmt_train_en_ko 0,1 en ko space &
run_nmt_train_en_ko 2,3 ko en moses