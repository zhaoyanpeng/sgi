#!/usr/bin/sh

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=$2

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

mode="dp"
num_proc=4
seed=5657 #4546 #3435 #2324 #1213 #

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

data_name="CLEVR_train_captions.2.0.thesis.one_hop.json"
eval_name="CLEVR_val_captions.2.0.thesis.one_hop.json"

data_name="CLEVR_train_captions.2.0.thesis.toy.one_hop.json"
eval_name="CLEVR_train_captions.2.0.thesis.toy.one_hop.json"

# Train VC-MLM w/ an additional causal language modelling objective
# bash bash/run_clevr_wclm.sh default 0

# train: clevr 
model_name="sgi.thesis.sym.clm.test.$seed"

mtask="alias_name=$model_name
verbose=True optimizer.warmup=False optimizer.weight_decay=1e-8
data.enc_vocab_name=$enc_vocab_name data.dec_vocab_name=$dec_vocab_name

data.input_cap_type=full_caps

data.cate_type=\"\"
data.cate_max_len=1

data.mlm_prob=0.15
data.relation_words=[left,right,in_front_of,behind]
model.loss.alpha_l1=0.
model.loss.name=MLMLMLossHead
+model.loss.alpha_lm=0.5
model.decoder.name=SGIMiniTFMLMLMDecHead
model.loss.optim_only_relation=False
model.decoder.num_p=512 model.decoder.cat_p=True model.decoder.p_dim=256 model.decoder.w_dim=256 
model.encoder.num_p=4 model.encoder.cat_p=True model.encoder.p_dim=256 model.encoder.w_dim=64 model.encoder.p_type=learned
optimizer.lr=5e-5 optimizer.scheduler=[MultiStepLR,{milestones:[15,36,45,50],gamma:0.5}]

model.encoder.activation=gelu model.encoder.ln_input=False
model.decoder.activation=gelu model.decoder.ln_input=False
model.encoder.num_layer=1 model.encoder.num_head=4 model.encoder.t_dropout=0. model.encoder.p_dropout=0.0
model.encoder.attn_dropout=0.0 model.encoder.proj_dropout=0.0
model.decoder.num_layer=2 model.decoder.num_head=4 model.decoder.t_dropout=0. model.decoder.p_dropout=0.0
model.decoder.attn_dropout=0.0 model.decoder.proj_dropout=0.0

model.encoder.sign_q=True
model.encoder.sign_k=False
model.encoder.attn_cls_intra=MiniTFAttention

model.decoder.sign_q_intra=False
model.decoder.sign_k_intra=False
model.decoder.sign_q_inter=True
model.decoder.sign_k_inter=False
model.decoder.attn_cls_intra=MiniTFAttention

model.decoder.block=MiniTFBlock
model.decoder.attn_cls_inter=MiniTFAttention
model.decoder.route_self_dropout=.15 model.decoder.route_memo_dropout=.15
model.decoder.infer_self_dropout=.25 model.decoder.infer_memo_dropout=.25

model.decoder.q_activation=gelu
model.decoder.k_activation=gelu

model.decoder.num_head_intra=4
model.decoder.num_head_inter=2
model.decoder.inter_layers=[1,1]
+model.decoder.split_vl=False

running.epochs=100 running.batch_size=50 running.peep_rate=100
running.save_rate=1e9 running.save_epoch=True running.skip_save=False running.save_last=True


data.eval_name=$eval_name data.eval_samples=1e3
"

#echo "exit..."
#exit 0

#model.decoder.block=RouteTFBlock
#model.decoder.attn_cls_inter=SpecialTFAttention

#data.relation_words=[left,right]

#running.milestones=[1,.0,15,18,20,21,22,23,24,30,35,40,45,50,55,60,61,62,63,64,65,70,75,80,81,82,83,84,85,90,100]
#running.milestones=[1,.0,15,18,20,21,22,23,24,30]
#running.milestones=[1,.0,30,35,38,40,41,42,43,44,50]

#running.epochs=1 running.batch_size=3 running.peep_rate=1 running.save_rate=1e9 running.save_epoch=True running.save_last=False
#model.decoder.num_p=512 model.decoder.cat_p=True model.decoder.p_dim=256 model.decoder.w_dim=256 model.decoder.p_type=sinuous 
#model.decoder.num_p=512 model.decoder.cat_p=False model.decoder.p_dim=512 model.decoder.w_dim=512 model.decoder.p_type=sinuous 

#model.encoder.num_p=4 model.encoder.cat_p=True model.encoder.p_dim=128 model.encoder.w_dim=96 model.encoder.p_type=learned

# config
extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup python -m torch.utils.bottleneck train.py \
#nohup 
python train.py port=$port num_gpus=$ngpu eval=False mode=$mode num_proc=$num_proc seed=$seed \
    alias_root=$alias_root data.data_name=$data_name data.data_root=$data_root \
    +model/encoder=mini_tf \
    +model/decoder=route_tf \
    +model/relation=dummy \
    +model/loss=ce_lm \
    +optimizer=default \
    +data=clevr \
    +running=$run_type $extra 
#> ../log/$model_name 2>&1 &
