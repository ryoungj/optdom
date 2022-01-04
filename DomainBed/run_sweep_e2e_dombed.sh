#!/bin/bash

data_dir=./datasets/
base_output_dir=./checkpoints
launcher=slurm_launcher

setup=e2e_dombed

dataset_array=(PACS VLCS OfficeHome TerraIncognita DomainNet)
algorithm_array=(CondCAD)
lambda_array=(0 1e-5)


for dataset in ${dataset_array[@]}; do
  for algorithm in ${algorithm_array[@]}; do
    for lambda in ${lambda_array[@]}; do
      subdir_name=lambda_${lambda}
      group_name=${setup}_${dataset}_${algorithm}_${subdir_name}
      subdir=${dataset}/${setup}/${algorithm}/${subdir_name}

      python -m domainbed.scripts.sweep delete_and_launch\
             --data_dir=${data_dir}\
             --output_dir=${base_output_dir}/${subdir}\
             --command_launcher ${launcher}\
             --algorithms ${algorithm}\
             --datasets ${dataset}\
             --n_hparams 20\
             --n_trials 3\
             --skip_confirmation\
             --single_test_envs\
             --task 'domain_generalization'\
             --hparams '{"lmbda":'"${lambda}"'}'
    done
  done
done

