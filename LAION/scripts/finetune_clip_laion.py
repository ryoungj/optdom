import argparse
import collections
import json
import os
import random
import sys
import time
from tqdm import tqdm
import wandb

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, Dataset

import clip

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune CLIP on LAION-400M with bottlenecks')
    parser.add_argument('--data_dir', type=str,
                        help='dataset directory')
    parser.add_argument('--dataset', type=str, default="PACS",
                        help='dataset for finetuning and evaluation')
    parser.add_argument('--algorithm', type=str, default="ERM",
                        help='algorithm implemented with CLIP')
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and random hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0],
                        help='list of left-out environments for testing, others used for training/finetuning')
    parser.add_argument('--output_dir', type=str, default="train output",
                        help='output directory')
    parser.add_argument('--always_rerun', type=misc.str2bool, default=False,
                        help='whether rerun jobs even already done')
    parser.add_argument('--wandb_logger', type=misc.str2bool, default=False,
                        help='whether use wandb logger')
    parser.add_argument('--wandb_proj', type=str, default='optdom',
                        help='wandb project name')
    parser.add_argument('--wandb_group', type=str, default='test',
                        help='wandb group name')
    args = parser.parse_args()

    num_workers = 4
    start_step = 0
    os.makedirs(args.output_dir, exist_ok=True)
    # sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    # sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if os.path.exists(os.path.join(args.output_dir, 'done')) and not args.always_rerun:
        print("Job already done!")
        exit(0)

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hseed = misc.seed_hash(args.hparams_seed, args.trial_seed)
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, hseed)

    if args.hparams:
        hparams.update(json.loads(args.hparams))

    if args.wandb_logger:
        config = hparams.copy()
        config.update(vars(args))
        wandb.init(project=args.wandb_proj, config=config, dir=args.output_dir, group=args.wandb_group)

    # set seed
    run_seed = args.seed
    hparams['run_seed'] = run_seed
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    assert hparams['clip_model'] == "ViT-B/32", "The LAION-400M features are obtained by CLIP-ViT-B/32"
    pretrained, preprocess = clip.load(hparams['clip_model'], device, jit=False)
    pretrained.float()

    print("Loaded pretrained CLIP model: {}, # of params: {}.".format(
        hparams['clip_model'], sum(p.numel() for p in pretrained.visual.parameters())))
    hparams['img_transform'] = preprocess
    hparams['debug'] = False
    feature_dim = pretrained.visual.output_dim

    assert args.dataset == "LAIONDomainsOriginal", "Please use LAION-400M dataset!"
    assert args.test_envs == [-1], "No test envs in this setup, please set `test_envs` to -1!"
    args.test_envs = []  # no test envs in for this setup, data from all envs should be used for training
    dataset = vars(datasets)[args.dataset](args.data_dir,
                                           args.test_envs, hparams)

    if 'img_transform' in hparams:
        del hparams['img_transform']

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))


    def save_checkpoint(filename):
        save_dict = {
            "args": vars(args),
            "model_feature_dim": feature_dim,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        save_path = os.path.join(args.output_dir, filename)
        torch.save(save_dict, save_path)
        print("Saved checkpoint to {}...".format(save_path))


    def infinite_wrapper(dl):
        while True:
            for x in dl:
                yield x


    assert len(dataset) == 1
    train_data = dataset[0]
    train_loader = infinite_wrapper(DataLoader(
        dataset=train_data,
        pin_memory=True,
        shuffle=False,  # for data loading speed
        batch_size=hparams['batch_size'],
        num_workers=num_workers))

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(feature_dim, dataset.num_classes,  # th inout shame becomes the output dim
                                len(dataset) - len(args.test_envs), hparams, pretrained, None)

    algorithm.to(device)
    algorithm.train()

    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = len(train_data) // hparams['batch_size'] + 1
    if algorithm.trainable:
        if not hparams["use_fix_step"]:
            n_steps = int(hparams['max_epoch'] * steps_per_epoch + 1)
        else:
            n_steps = hparams['max_step']
        checkpoint_freq = n_steps // 10
    else:
        n_steps = 1
        checkpoint_freq = 1

    last_results_keys = None
    for step in tqdm(range(start_step, n_steps)):
        if algorithm.trainable:
            step_start_time = time.time()
            image_feature, text_feature = next(train_loader)
            minibatches_device = [(image_feature.to(device).float(), text_feature.to(device).float())]

            algorithm.adjust_lr(step, n_steps, steps_per_epoch)
            step_vals = algorithm.update(minibatches_device, None)
            checkpoint_vals['step_time'].append(time.time() - step_start_time)

            if args.wandb_logger:
                wandb.log({f'train/{k}': v for k, v in step_vals.items()}, step=step)
                wandb.log({'train/lr': algorithm.optimizer.param_groups[0]['lr']}, step=step)

            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            results_keys = [k for k in sorted(results.keys())]
            colwidth = max(12, max([len(k) for k in results_keys]))
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=colwidth)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=colwidth)

            if args.wandb_logger:
                wandb.log({f'valid/{k}': v for k, v in results.items()}, step=step)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            # save_checkpoint('model_step{}.pkl'.format(step))

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
