#!/bin/bash
info=/workspace/fairseq/examples/info_quantizer

# ########################## en-vi ############################
src=vi
tgt=en
nmt_path=/workspace/model/offline_nmt/$src-$tgt
data_path=/workspace/data/iwslt15.tokenized.vi-en
save_path=/workspace/data/generated_data/$src-$tgt
generate_data(){
    CUDA_VISIBLE_DEVICES=$1 python $info/generate_data.py \
        $data_path \
        -s $src -t $tgt \
        --user-dir $info \
        --task info_quantizer_task \
        --gen-subset $2 \
        --path $nmt_path/checkpoint_best.pt \
        --scoring sacrebleu \
        --max-tokens 12000 \
        --skip-invalid-size-inputs-valid-test \
        --beam $3 \
        --has-target \
        --post-process subword_nmt \
        --results-path $save_path
}

generate_data 0 train 3 &
generate_data 1 test 3 &
generate_data 2 valid 3 &

########################## SORT ############################
sort(){
    for f in ${save_path}/generate*; do
        python $info/data/sort_data.py $f
    done
}
sort

########################## PREP ############################
preprocess(){
    python $info/data/preprocess.py \
        -s src -t trg \
        --trainpref $save_path/sorted/generate-train-target-beam3 \
        --validpref $save_path/sorted/generate-valid-target-beam3 \
        --testpref $save_path/sorted/generate-test-target-beam3 \
        --tgtdict $save_path/dict.trg.txt \
        --srcdict $save_path/dict.src.txt \
        --destdir $save_path/bin-data
}
preprocess





########################## en-ko-hugginface ############################
src=ko
tgt=en
data_path=/workspace/data/opus.spm_tokenized.ko-en/data-bin
save_path=/workspace/data/generated_data/$src-$tgt-marian
generate_data(){
    CUDA_VISIBLE_DEVICES=$1 python python $info/generate_data.py \
        $data_path \
        -s $src -t $tgt \
        --user-dir $info \
        --task info_quantizer_task \
        --gen-subset $2 \
        --path marian \
        --max-tokens $4 \
        --skip-invalid-size-inputs-valid-test \
        --remove-bpe sentencepiece \
        --beam 3 \
        --has-target \
        --subsample $3 \
        --seed $5 \
        --max-len-b 50 \
        --hf-model \
        --results-path $save_path
}

result_path=$save_path/data/oracle_gen_data
generate_data 1 test 1 500 42 &
generate_data 2 valid 1 500 42
generate_data 3 train 0.2 2500 1

cp $data_path/dict.act.txt $result_path
cp $data_path/dict.en.txt.no_symbol $result_path
cp $data_path/dict.ko.txt.no_symbol $result_path
mv $result_path/dict.ko.txt.no_symbol $result_path/dict.src.txt.no_symbol
mv $result_path/dict.en.txt.no_symbol $result_path/dict.trg.txt.no_symbol

########################## SORT ############################
sort(){
    for f in $save_path/generate*; do
        python $info/data/sort_data.py $f
    done
}
sort 

########################## PREP ############################
preprocess(){
    python $info/data/preprocess.py \
        -s src -t trg \
        --trainpref ${result_path}/sorted/generate-train-target-beam3 \
        --validpref ${result_path}/sorted/generate-valid-target-beam3 \
        --testpref ${result_path}/sorted/generate-test-target-beam3 \
        --tgtdict ${result_path}/dict.trg.txt.no_symbol \
        --srcdict ${result_path}/dict.src.txt.no_symbol \
        --destdir ${result_path}/bin-data
}
preprocess

mv ${result_path}/bin-data/dict.src.txt ${result_path}/bin-data/dict.src.txt.no_symbol
mv ${result_path}/bin-data/dict.trg.txt ${result_path}/bin-data/dict.trg.txt.no_symbol