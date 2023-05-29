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

# config for CLEVR 
data_root="$HOME/backup/data/scene/clevr"
more_root="$HOME/backup/data/scene/clevr_bbox"

data_name="CLEVR_train_captions.2.0.mp.one_hop.json"
data_name="CLEVR_val_captions.2.0.mp.one_hop.json"

# config for AbstractScene
data_root="$HOME/backup/data/scene/AbstractScenes_v1.1"
more_root=$data_root

vgg_name=vgg19_bn

# Encode CLEVR / AbstractScene images into vectors
# bash bash/run_obj_enc.sh default 0

# train: clevr 
model_name="obj.enc.clevr.train.$vgg_name"
model_name="obj.enc.clevr.val.$vgg_name"
model_name="obj.enc.abscene.$vgg_name"
model_name="obj.enc.test"
mtask="alias_name=$model_name monitor=ObjEncMonitor worker=PretrainedVGG  
verbose=True 

running.batch_size=1 running.peep_rate=100

+model.name=$vgg_name
data.name=abscene data.more_root=$more_root data.dump_root='\${.data_root}/object_$vgg_name' 

data.eval_name=$eval_name data.eval_samples=1e6
"

#data.name=clevr data.more_root=$more_root data.dump_root='\${.data_root}/object_$vgg_name' 

# config
extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup python -m torch.utils.bottleneck train.py \
nohup python train.py port=$port num_gpus=$ngpu eval=True mode=$mode num_proc=$num_proc seed=$seed \
    alias_root=$alias_root data.data_name=$data_name data.data_root=$data_root \
    +data=clevr \
    +running=$run_type $extra > ./log/$model_name 2>&1 &
