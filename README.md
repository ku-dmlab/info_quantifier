## SR-KU Final report repository

Trained NMT model, Training Dataset, Trained Policy model
[Download link](https://www.notion.so/Back-Data-29ce49f76aab483eb3e4890f220247fa)

### Dataset
- IWSLT14 En <-> De
- IWSLT15 En <-> Vi
- IWSLT17 En <-> Ko
- OPUS Ko -> En

### Install
```bash
# Fairseq
git clone https://github.com/ku-dmlab/info_quantizer
cd info-quantizer
pip install -e .

# Simuleval
cd Simuleval
pip install -e .
```

### Train Offline NMT Model 
or download pretrained model [Download link](https://www.notion.so/Back-Data-29ce49f76aab483eb3e4890f220247fa) or Huggingface Model

```bash
data_path=/workspace/data/iwslt14.tokenized.de-en
save_path=/workspace/model/offline_nmt

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

run_nmt_train_en_de 0 en de
```

### Generate oracle action sequences
or download generated dataset [Download link](https://www.notion.so/Back-Data-29ce49f76aab483eb3e4890f220247fa)
- Dataset generation code based on 
  - https://aclanthology.org/2021.emnlp-main.130/
  - https://github.com/sfu-natlang/Supervised-Simultaneous-MT

```bash
info=/workspace/fairseq/examples/info_quantizer

src=en
tgt=de
nmt_path=/workspace/model/offline_nmt/$src-$tgt
data_path=/workspace/data/iwslt14.tokenized.de-en
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

sort(){
    for f in ${save_path}/generate*; do
        python $info/data/sort_data.py $f
    done
}

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

generate_data 0 train 3 &
generate_data 1 test 3 &
generate_data 2 valid 3

sort

preprocess
```

### Train IQ (Information Quantizer)
[Download link](https://www.notion.so/Back-Data-29ce49f76aab483eb3e4890f220247fa)

```bash
info=/workspace/fairseq/examples/info_quantizer
data_dir=/workspace/data/generated_data/en-de/bin-data
save_dir=/workspace/model/iq/en-de

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

agent_train 0,1,2,3
```

### SimulEval
```bash
agent_path=/workspace/fairseq/SimulEval/simuleval/user_agent
test_src=/workspace/data/iwslt14.tokenized.de-en/simul.en
test_tgt=/workspace/data/iwslt14.tokenized.de-en/simul.de
output_path=simulresult/en-de
nmt_path=/workspace/model/offline_nmt/en-de

simuleval(){
    CUDA_VISIBLE_DEVICES=$1 simuleval \
        --user-dir $info \
        --task info_quantizer \
        --source $test_src \
        --target $test_tgt \
        --eval-latency-unit word \
        --agent $agent_path/info_agent_trg.py \
        --agent-arch softplus2 \
        --mt-model-path $nmt_path/checkpoint_best.pt \
        --policy-model-path $save_dir/checkpoint_best.pt \
        --beam-size 1 \
        --threshold $2 \
        --source-type text \
        --target-type text \
        --latency-metrics AL LAAL AP DAL \
        --quality-metrics BLEU \
        --device gpu \
        --output $output_path/threshold_$2
}

simuleval 0 0 &
```