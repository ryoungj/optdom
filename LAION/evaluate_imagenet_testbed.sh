#!/bin/bash

imagenet_data_dir=./data/imagenet
base_output_dir=./checkpoints
setup=clip_laion

algorithm_array=(CLIPPretrained ContrastCLIPBottleneckBase ContrastCLIPBottleneckEnt)

############ train linear classifier on imagenet and evaluate on netural distribution shift datasets ############
for algorithm in ${algorithm_array[@]}; do
  if [[ "$algorithm" == "ContrastCLIPBottleneckEnt" ]]
  then
    lambda=1
    lr=1e-3
    name="clip_vit_finetuned_ent"
  elif [[ "$algorithm" == "ContrastCLIPBottleneckBase" ]]
  then
    lambda=0
    lr=3e-3
    name="clip_vit_finetuned_base"
  else
    lambda=0
    lr=3e-3
    name="clip_vit_pretrained"
  fi

  string=${algorithm}_${lambda}_${lr}
  output_dir_path=${base_output_dir}/${setup}_${string}

  out_path="${output_dir_path}/eval_imagenet.out"
  err_path="${output_dir_path}/eval_imagenet.err"
  script_path="${output_dir_path}/eval_imagenet.sh"

  # fit classifier
  cmd_fit="python -m scripts.train_imagenet_classifier
        --dataset_dir ${imagenet_data_dir}
        --model_ckpt_path ${output_dir_path}/model.pkl
        --save_ckpt_path ${output_dir_path}/model_with_logistic.pkl
        --clf_type LogisticPT
        --load_or_save_features
        --clf_train_lr 3e-4
        --clf_train_l2_reg 1e-5"


  # eval classifier
  cmd_eval="python eval.py --gpus 0 --models ${name} --eval-settings val val_subsampled_class_1_8 imagenetv2-matched-frequency imagenet-sketch ytbb-robust imagenet-vid-robust objectnet-1.0-beta imagenet-a imagenet-r"

  echo "#!/bin/sh" > $script_path
  echo $cmd_fit >> $script_path
  echo "cd ./imagenet-testbed/" >> $script_path
  echo $cmd_eval >> $script_path

  # if you are use slurm, launch jobs in parallel
  sbatch -o $out_path -e $err_path --gres=gpu:1 --mem=48G -c 12 -p p100 --qos normal $script_path
  # otherwise
#  bash $script_path
done