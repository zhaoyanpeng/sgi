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

enc_vocab_name=notafile
dec_vocab_name="toy.256.dict"

data_root="/net/nfs2.mosaic/yann/data/copy/"
data_name="train.10.256.5.9.50.json"
data_name="train.10.256x.5.9.50.json"
data_name="train.100.256.5.9.50.json"
eval_name="test.10.256.5.9.25.json"
eval_name="test.10.256x.5.9.25.json"
eval_name="test.100.256.5.9.25.json"


model_name="toy.copy."$data_name
model_name="toy.copy"
mtask="alias_name=$model_name
verbose=True optimizer.warmup=False optimizer.weight_decay=1e-6
data.enc_vocab_name=$enc_vocab_name data.dec_vocab_name=$dec_vocab_name

data.name=copy
model.encoder.num_p=32 model.encoder.cat_p=True model.encoder.p_dim=512 
model.encoder.w_dim=512 model.encoder.num_w=1
optimizer.lr=1e-4 optimizer.scheduler=[MultiStepLR,{milestones:[5,10,12,15,20],gamma:0.5}]

model.encoder.num_layer=2 model.encoder.num_head=8 model.encoder.t_dropout=0.
model.decoder.num_layer=2 model.decoder.t_dropout=0.

running.epochs=20 running.batch_size=50 
running.peep_rate=100 running.save_rate=1e9 running.save_epoch=True
data.eval_name=$eval_name data.eval_samples=1e6
"

#optimizer.lr=1e-4 optimizer.scheduler=[]
#optimizer.lr=1e-4 optimizer.scheduler=[StepLR,{step_size:1.0,gamma:0.95}]
#optimizer.lr=5e-5 optimizer.scheduler=[StepLR,{step_size:1.0,gamma:0.95}]
#model.encoder.name=TorchTFEncHead model.decoder.name=TorchTFDecHead

# config
extra="$mtask "
 
#export CUDA_LAUNCH_BLOCKING=1
#nohup python -m torch.utils.bottleneck train.py \
#nohup 
python train.py port=$port num_gpus=$ngpu eval=False mode=$mode num_proc=$num_proc seed=$seed \
    alias_root=$alias_root data.data_name=$data_name data.data_root=$data_root \
    +model/encoder=mini_tf \
    +model/decoder=mini_tf \
    +model/relation=dummy \
    +model/loss=ce_lm \
    +optimizer=default \
    +data=clevr \
    +running=$run_type $extra 
#> ./log/$model_name 2>&1 &
