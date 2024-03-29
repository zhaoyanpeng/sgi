#!/usr/bin/sh

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=$2

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

mode="dp"
num_proc=4
seed=1213

echo "GPUs: "$CUDA_VISIBLE_DEVICES "#"$ngpu "PORT: "$port

alias_root="$HOME/backup/model/sgi"
model_root=$alias_root

data_root="$HOME/backup/data/scene/clevr/"

enc_vocab_name="object_label.split.train.70k.topk.dict"
dec_vocab_name="captions_train.70k.topk.dict"
dec_vocab_name="captions_train.2.0.topk.dict"

data_name="CLEVR_train_captions.70k.one_hop.json"
eval_name="CLEVR_val_captions.one_hop.json"

data_name="CLEVR_train_captions.2.0.one_hop.json"
eval_name="CLEVR_val_captions.2.0.one_hop.json"

data_name="CLEVR_train_captions.2.0.thesis.toy.one_hop.json"
eval_name="CLEVR_train_captions.2.0.thesis.toy.one_hop.json"

#
# bash bash/run_clevr_eval.sh default 0

alias_name=sgi_test
model_file=00000643.pth
model_file=00000525.pth


# train: clevr 
model_name="sgi.clevr.base.eval"
mtask="
verbose=False alias_name=$alias_name model_root=$model_root model_name=$model_name model_file=$model_file 
data.enc_vocab_name=$enc_vocab_name data.dec_vocab_name=$dec_vocab_name

data.input_cap_type=full

data.relation_words=[left,right]
data.cate_type=\"\"
data.cate_max_len=1
data.mlm_prob=0.15

running.batch_size=50
data.eval_name=$eval_name data.eval_samples=1e3
"

# config
extra="$mtask "

#data.relation_words=[left,right,in_front_of,behind]
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup python -m torch.utils.bottleneck train.py \
#nohup 
python train.py port=$port num_gpus=$ngpu eval=True mode=$mode num_proc=$num_proc seed=$seed \
    alias_root=$alias_root data.data_name=$data_name data.data_root=$data_root \
    +model/encoder=mini_tf \
    +model/decoder=mini_tf \
    +model/relation=dummy \
    +model/loss=ce_lm \
    +optimizer=default \
    +data=clevr \
    +running=$run_type $extra 
#> ./log/$model_name 2>&1 &

