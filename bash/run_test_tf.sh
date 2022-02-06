#!/usr/bin/sh

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=$2

run_type=$1
gpu_list=(${CUDA_VISIBLE_DEVICES//,/ })

port=$(expr $RANDOM + 1000)
ngpu=${#gpu_list[@]}

mode="dp"
num_proc=1
seed=1213

echo "GPUs: "$CUDA_VISIBLE_DEVICES "#"$ngpu "PORT: "$port

alias_root="/net/nfs2.mosaic/yann/model/sgi"
model_root=$alias_root

data_root=null
data_name=null

enc_vocab_name=null
dec_vocab_name=null

# bash bash/run_test.sh default 0 

model_name="tf.test"
mtask="alias_name=$model_name
monitor=MiniTFMonitor worker=MiniTFLM
verbose=True optimizer.warmup=False optimizer.weight_decay=0
data.enc_vocab_name=$enc_vocab_name data.dec_vocab_name=$dec_vocab_name

+data.bptt=35
model.encoder.w_dim=512
optimizer.max_gnorm=0.5
optimizer.optimizer=[SGD,{lr:5.0}]
optimizer.scheduler=[StepLR,{step_size:1.0,gamma:0.95}]

model/encoder=mini_tf model/decoder=mini_tf
model.encoder.num_layer=2 model.decoder.num_layer=2
running.epochs=5 running.batch_size=32 running.peep_rate=200 running.save_rate=1e9 running.save_epoch=True data.eval_samples=100
"

#model.decoder.require_inter_attn=True

# config
extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup python -m torch.utils.bottleneck train.py \
#nohup 
python train.py port=$port num_gpus=$ngpu eval=False mode=$mode num_proc=$num_proc seed=$seed \
    alias_root=$alias_root data.data_name=$data_name data.data_root=$data_root \
    +model/encoder=default \
    +model/decoder=default \
    +model/loss=ce_lm \
    +optimizer=default \
    +data=clevr \
    +running=$run_type $extra 
#> ./log/$model_name 2>&1 &
