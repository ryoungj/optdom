#!/bin/bash


python3 -m domainbed.scripts.train_clip\
       --data_dir=./datasets/\
       --output_dir=./checkpoints/test\
       --algorithm SupCLIPBottleneckCondCAD\
       --dataset PACS\
       --hparams_seed 0\
       --trial_seed 0\
       --seed 1010\
       --task 'domain_generalization'\
       --test_envs 3\
       --hparams '{"lmbda":1e-2,"mlp_depth":2}'\
       --load_or_save_clip_features true\
       --wandb_logger false\
       --always_rerun true\
       --debug