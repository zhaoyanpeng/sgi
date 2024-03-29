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

data_name="CLEVR_train_captions.70k.one_hop.json"
eval_name="CLEVR_val_captions.one_hop.json"

data_name="CLEVR_train_captions.toy.json"
eval_name="CLEVR_train_captions.toy.json"

#
# bash bash/run_clevr_base.sh default 0

# train: clevr 
model_name="sgi.clevr.rel.test"
mtask="alias_name=$model_name
verbose=True optimizer.warmup=False optimizer.weight_decay=1e-6
data.enc_vocab_name=$enc_vocab_name data.dec_vocab_name=$dec_vocab_name

data.relation_words=[left,front]
data.cate_type=atomic_object
data.cate_max_len=64

model.loss.optim_only_relation=True
model.decoder.num_p=512 
model.relation.num_relation=32
model.relation.num_p=4 model.relation.cat_p=True model.relation.p_dim=512 model.relation.use_only_pos=False

model.encoder.num_p=-1 model.encoder.w_dim=128 model.encoder.num_w=4
optimizer.lr=5e-5 optimizer.scheduler=[MultiStepLR,{milestones:[15,36,45,50],gamma:0.5}]

model.encoder.attn_cls_intra=RelationTFAttention
model.encoder.num_layer=2 model.encoder.t_dropout=0. model.encoder.p_dropout=0. model.encoder.num_head=8
model.decoder.num_layer=1 model.decoder.t_dropout=0 model.decoder.p_dropout=0
model.relation.num_layer=2 model.relation.t_dropout=0 model.relation.p_dropout=0 model.relation.num_head=8

running.epochs=100 running.batch_size=50 running.peep_rate=100 
running.save_rate=1e9 running.save_epoch=True running.skip_save=True running.save_last=True
data.eval_name=$eval_name data.eval_samples=1e6
"

#data.relation_words=[left,right]
#data.relation_words=[left,right,front,behind]
#running.epochs=1 running.batch_size=2 running.peep_rate=1 
#optimizer.lr=1e-4 optimizer.scheduler=[MultiStepLR,{milestones:[10,30,60,90],gamma:0.5}]
#running.save_rate=1e9 running.save_epoch=False
#model.decoder.require_inter_attn=True

# config
extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup python -m torch.utils.bottleneck train.py \
#nohup 
python train.py port=$port num_gpus=$ngpu eval=False mode=$mode num_proc=$num_proc seed=$seed \
    alias_root=$alias_root data.data_name=$data_name data.data_root=$data_root \
    +model/encoder=mini_tf \
    +model/decoder=mini_tf \
    +model/relation=mini_tf \
    +model/loss=ce_lm \
    +optimizer=default \
    +data=clevr \
    +running=$run_type $extra 
#> ./log/$model_name 2>&1 &
