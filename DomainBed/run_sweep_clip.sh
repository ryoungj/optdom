#!/bin/bash

data_dir=./datasets/
base_output_dir=./checkpoints
launcher=slurm_launcher

setup=clip_resnet
clip_model=RN50   # RN50, ViT-B/32
algorithm_array=(CLIPPretrained SupCLIPBottleneckBase SupCLIPBottleneckCondCAD)

########### datasets except for DomainNet #########
dataset_array=(PACS VLCS OfficeHome TerraIncognita)

for dataset in ${dataset_array[@]}; do
  for algorithm in ${algorithm_array[@]}; do
    if [[ "$algorithm" == "CLIPPretrained" ]] || [[ "$algorithm" == "SupCLIPBottleneckBase" ]]
    then
      if [[ "$algorithm" == "CLIPPretrained" ]]
      then
        n_params=1
      else
        n_params=10
      fi

      subdir_name=base
      group_name=${setup}_${dataset}_${algorithm}_${subdir_name}
      subdir=${dataset}/${setup}/${algorithm}/${subdir_name}

      python -m domainbed.scripts.sweep_clip delete_and_launch\
       --data_dir=${data_dir}\
       --output_dir=${base_output_dir}/${subdir}\
       --command_launcher ${launcher}\
       --algorithms ${algorithm}\
       --datasets ${dataset}\
       --n_hparams ${n_params}\
       --n_trials 5\
       --skip_confirmation\
       --train_script domainbed.scripts.train_clip\
       --single_test_envs\
       --wandb_group ${group_name}\
       --task 'domain_generalization'\
       --hparams '{"clip_model":"'"${clip_model}"'","mlp_depth":2}'
    elif [[ "$algorithm" == "SupCLIPBottleneckCondCAD" ]]
    then
      lambda=1e-2

      subdir_name=lambda_${lambda}
      group_name=${setup}_${dataset}_${algorithm}_${subdir_name}
      subdir=${dataset}/${setup}/${algorithm}/${subdir_name}

      python -m domainbed.scripts.sweep_clip delete_and_launch\
             --data_dir=${data_dir}\
             --output_dir=${base_output_dir}/${subdir}\
             --command_launcher ${launcher}\
             --algorithms ${algorithm}\
             --datasets ${dataset}\
             --n_hparams 10\
             --n_trials 5\
             --skip_confirmation\
             --train_script domainbed.scripts.train_clip\
             --single_test_envs\
             --wandb_group ${group_name}\
             --task 'domain_generalization'\
             --hparams '{"lmbda":'"${lambda}"',"clip_model":"'"${clip_model}"'","mlp_depth":2}'
    else
        echo "Unknown algorithms: ${algorithm}"
        exit
    fi
  done
done


########## special: DomainNet (refit with pytorch implemented classifier with minibatch training) #########
dataset_array=(DomainNet)

for dataset in ${dataset_array[@]}; do
  for algorithm in ${algorithm_array[@]}; do
    if [[ "$algorithm" == "CLIPPretrained" ]] || [[ "$algorithm" == "SupCLIPBottleneckBase" ]]
    then
      if [[ "$algorithm" == "CLIPPretrained" ]]
      then
        n_params=1
      else
        n_params=10
      fi

      subdir_name=base
      group_name=${setup}_${dataset}_${algorithm}_${subdir_name}
      subdir=${dataset}/${setup}/${algorithm}/${subdir_name}

      python -m domainbed.scripts.sweep_clip delete_and_launch\
       --data_dir=${data_dir}\
       --output_dir=${base_output_dir}/${subdir}\
       --command_launcher ${launcher}\
       --algorithms ${algorithm}\
       --datasets ${dataset}\
       --n_hparams ${n_params}\
       --n_trials 5\
       --skip_confirmation\
       --train_script domainbed.scripts.train_clip\
       --single_test_envs\
       --wandb_group ${group_name}\
       --task 'domain_generalization'\
       --hparams '{"clip_model":"'"${clip_model}"'","mlp_depth":2,"max_epoch": 200, "clf_type": "LogisticPT"}'
    elif [[ "$algorithm" == "SupCLIPBottleneckCondCAD" ]]
    then
      if [[ "$clip_model" == "RN50" ]]
      then
        lambda=1
      elif [[ "$clip_model" == "ViT-B/32" ]]
      then
        lambda=1e-1
      else
        echo "Unknown model: ${clip_model}"
        exit
      fi

      subdir_name=lambda_${lambda}
      group_name=${setup}_${dataset}_${algorithm}_${subdir_name}
      subdir=${dataset}/${setup}/${algorithm}/${subdir_name}

      python -m domainbed.scripts.sweep_clip delete_and_launch\
             --data_dir=${data_dir}\
             --output_dir=${base_output_dir}/${subdir}\
             --command_launcher ${launcher}\
             --algorithms ${algorithm}\
             --datasets ${dataset}\
             --n_hparams 10\
             --n_trials 5\
             --skip_confirmation\
             --train_script domainbed.scripts.train_clip\
             --single_test_envs\
             --wandb_group ${group_name}\
             --task 'domain_generalization'\
             --hparams '{"lmbda":'"${lambda}"',"clip_model":"'"${clip_model}"'","mlp_depth":2,"max_epoch": 200, "clf_type": "LogisticPT"}'
    else
        echo "Unknown algorithms: ${algorithm}"
        exit
    fi
  done
done








