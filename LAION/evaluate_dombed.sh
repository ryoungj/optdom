#!/bin/bash

dombed_data_dir=./data/domainbed
base_output_dir=./checkpoints
launcher=slurm_launcher
setup=clip_laion

algorithm_array=(CLIPPretrained ContrastCLIPBottleneckBase ContrastCLIPBottleneckEnt)


################### evaluate finetuned model on DomainBed ###################
dataset_array=(PACS DomainNet VLCS OfficeHome TerraIncognita)

# add path for finding domainbed modules
export PYTHONPATH=$PYTHONPATH:../DomainBed

for dataset in ${dataset_array[@]}; do
  for algorithm in ${algorithm_array[@]}; do
    if [[ "$algorithm" == "ContrastCLIPBottleneckEnt" ]]
  then
    lambda=1
    lr=1e-3
  elif [[ "$algorithm" == "ContrastCLIPBottleneckBase" ]]
  then
    lambda=0
    lr=3e-3
  else
    lambda=0
    lr=3e-3
  fi

  string=${algorithm}_${lambda}_${lr}
  output_dir_path=${base_output_dir}/${setup}_${string}

  if [[ "$dataset" == "DomainNet" ]]
    then
      clf_type="LogisticPT"
    else
      clf_type="SVM"
    fi

  python -m domainbed.scripts.sweep_clip delete_and_launch\
       --data_dir=${dombed_data_dir}\
       --output_dir=${output_dir_path}/eval/${dataset}\
       --command_launcher ${launcher}\
       --algorithms ${algorithm}\
       --datasets ${dataset}\
       --n_hparams 1\
       --n_trials 5\
       --skip_confirmation\
       --train_script domainbed.scripts.train_clip\
       --single_test_envs\
       --task 'domain_generalization'\
       --only_eval true\
       --warmstart_model_ckpt ${output_dir_path}/model.pkl\
       --hparams '{"clip_model":"ViT-B/32","lmbda":0.0,"clf_type":"'${clf_type}'","mlp_blocks":2,"mlp_width":2048,"mlp_depth":3,"mlp_dropout":0.1,"mlp_norm":true,"temperature":0.07,"is_symmetric":true}'
  done
done