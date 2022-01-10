#!/bin/bash

data_dir=./data/processed
base_output_dir=./checkpoints
setup=clip_laion


# For `CLIPPretrained` it is just saved the model checkpoint for compatibility
# For others, the CLIP model is finetuned on LAION with an additional MLP on top
algorithm_array=(CLIPPretrained ContrastCLIPBottleneckBase ContrastCLIPBottleneckEnt)

#################### finetune clip ###################

for algorithm in ${algorithm_array[@]}; do
  if [[ "$algorithm" == "ContrastCLIPBottleneckEnt" ]]
  then
    lambda=1
    lr=1e-3
  else
    lambda=0
    lr=3e-3
  fi

  string=${algorithm}_${lambda}_${lr}
  output_dir_path=${base_output_dir}/${setup}_${string}
  mkdir -p ${output_dir_path}

  out_path="${output_dir_path}/train.out"
  err_path="${output_dir_path}/train.err"
  script_path="${output_dir_path}/train.sh"

  # train model
  cmd_train='python3 -m scripts.finetune_clip_laion
   --data_dir='${data_dir}'
   --output_dir='${output_dir_path}'
   --algorithm '${algorithm}'
   --dataset LAIONDomainsOriginal
   --hparams_seed 0
   --trial_seed 0
   --seed 0
   --test_envs -1
   --wandb_logger false
   --hparams '"'"'{"clip_model": "ViT-B/32","lr":'${lr}',"lmbda":'${lambda}',"mlp_blocks":2, "mlp_width": 2048, "mlp_depth": 3, "mlp_dropout": 0.1, "mlp_norm": true,"temperature":0.07, "is_symmetric": true, "max_epoch": 1, "batch_size": 16384, "clf_type": "LogisticPT"}'"'"

  echo "#!/bin/sh" > $script_path
  echo ${cmd_train} >> $script_path


  # if you are use slurm, launch jobs in parallel
  sbatch -o $out_path -e $err_path --gres=gpu:1 --mem=48G -c 12 -p p100 --qos normal $script_path
  # otherwise
#  bash $script_path
done