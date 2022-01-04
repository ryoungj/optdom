# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from tqdm import tqdm
import wandb
import copy

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

import clip

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune CLIP for domain generalization')
    parser.add_argument('--data_dir', type=str,
                        help='dataset directory')
    parser.add_argument('--dataset', type=str, default="PACS",
                        help='dataset for finetuning and evaluation')
    parser.add_argument('--algorithm', type=str, default="ERM",
                        help='algorithm implemented with CLIP')
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"],
                        help='only support domain generalization')
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and random hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0],
                        help='list of left-out environments for testing, others used for training/finetuning')
    parser.add_argument('--output_dir', type=str, default="train output",
                        help='output directory')
    parser.add_argument('--holdout_fraction', type=float, default=0.2,
                        help='dataset holdout fraction for evaluation')
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--always_rerun', type=misc.str2bool, default=False,
                        help='whether rerun jobs even already done')
    parser.add_argument('--only_eval', type=misc.str2bool, default=False,
                        help='only evaluate a trained model')
    parser.add_argument('--warmstart_model_ckpt', type=str, default=None,
                        help='the trained model checkpoint for warmstarting or evaluation')
    parser.add_argument('--load_or_save_clip_features', type=misc.str2bool, default=True,
                        help='whether load precomputed CLIP features if saved before, or save features after precompute')
    parser.add_argument('--feature_save_dir', type=str, default=None,
                        help='directory for saving precomputed features')
    parser.add_argument('--debug', action='store_true',
                        help='whether in the debug mode')
    parser.add_argument('--wandb_logger', type=misc.str2bool, default=False,
                        help='whether use wandb logger')
    parser.add_argument('--wandb_proj', type=str, default='optdom',
                        help='wandb project name')
    parser.add_argument('--wandb_group', type=str, default='test',
                        help='wandb group name')

    args = parser.parse_args()

    num_workers = 2 if args.dataset != 'DomainNet' else 0
    start_step = 0
    larger_batch = args.dataset in ['OfficeHome', 'DomainNet']

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

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

    if args.hparams_seed == 0:  # load default hparams
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset,
                                                   larger_batch=larger_batch)
    else:  # randomize hparams
        hseed = misc.seed_hash(args.hparams_seed, args.trial_seed)
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset, hseed,
                                                  larger_batch=larger_batch)

    if args.hparams:
        hparams.update(json.loads(args.hparams))
    hparams["debug"] = args.debug
    assert not (hparams['data_augmentation']), "No data augmentation with feature pre-compute for freezed models!"

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

    available_models = clip.available_models()
    assert hparams['clip_model'] in available_models

    pretrained, preprocess = clip.load(hparams['clip_model'], device)
    print("Loaded pretrained CLIP model: {}, # of params: {}.".format(
        hparams['clip_model'], sum(p.numel() for p in pretrained.visual.parameters())))
    hparams['img_transform'] = preprocess

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    if 'img_transform' in hparams:
        del hparams['img_transform']

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    num_envs = len(dataset)
    class2idx = dataset.datasets[0].class_to_idx
    class2idx = {' '.join(k.lower().split('_')): v for k, v in class2idx.items()}
    idx2class = {v: k for k, v in class2idx.items()}
    num_classes = len(class2idx.items())


    def save_checkpoint(filename):
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        save_path = os.path.join(args.output_dir, filename)
        torch.save(save_dict, save_path)
        print(f"Saved checkpoint to {save_path}...")


    def save_processed_dataset(dirname, filename, _in_splits, _out_splits):
        save_dict = {}
        for i, ds in enumerate(_in_splits):
            save_dict.update({'env{}_in'.format(i): ds[0]})

        for i, ds in enumerate(_out_splits):
            save_dict.update({'env{}_out'.format(i): ds[0]})

        torch.save(save_dict, os.path.join(dirname, filename))


    def load_processed_dataset(path):
        processed_dataset = torch.load(path)

        in_splits = []
        out_splits = []
        for i in range(num_envs):
            in_splits.append((processed_dataset[f'env{i}_in'], None))
            out_splits.append((processed_dataset[f'env{i}_out'], None))

        return in_splits, out_splits


    # Split dataset
    # split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    assert args.task == 'domain_generalization'
    split_seed = args.trial_seed

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
                                      int(len(env) * args.holdout_fraction),
                                      misc.seed_hash(split_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                                          int(len(in_) * args.uda_holdout_fraction),
                                          misc.seed_hash(split_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))


    def get_clip_feature(clip_model, x):
        with torch.no_grad():
            z = clip_model.encode_image(x).float()
        return z


    # Load model
    feature_dim = get_clip_feature(pretrained, torch.zeros(1, 3, 224, 224).to(device)).size(1)

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(feature_dim, dataset.num_classes,  # th input shape becomes the output dim
                                len(dataset) - len(args.test_envs), hparams, pretrained, idx2class)

    if args.only_eval:
        assert args.warmstart_model_ckpt is not None

    if args.warmstart_model_ckpt is not None:
        print("Warm starting from {}..".format(args.warmstart_model_ckpt))
        assert os.path.exists(args.warmstart_model_ckpt)
        algorithm_dict = torch.load(args.warmstart_model_ckpt)["model_dict"]
        missing_keys, unexpected_keys = algorithm.load_state_dict(algorithm_dict, strict=False)
        print("Missing: {}. Unexpected: {}".format(missing_keys, unexpected_keys))

    algorithm.to(device)

    # Precompute CLIP features
    if args.load_or_save_clip_features:
        if args.feature_save_dir is None:
            clip_features_save_dir = os.path.join(args.data_dir, 'clip_features')
        else:
            clip_features_save_dir = args.feature_save_dir
        os.makedirs(clip_features_save_dir, exist_ok=True)

        clip_model_name = 'clip_' + hparams["clip_model"] if hparams["clip_model"] != "ViT-B/32" else "ViT"
        clip_features_save_name = f'{args.dataset}_{clip_model_name}_split_seed_{split_seed}.pkl'
        clip_features_save_path = os.path.join(clip_features_save_dir, clip_features_save_name)


    def clip_featurize_data(clip_model, dataset, device):
        """Compute CLIP features"""

        Z, Y = [], []
        for x, y in tqdm(DataLoader(dataset, batch_size=512, num_workers=4)):
            z = get_clip_feature(clip_model, x.to(device))
            Y += [y.cpu()]
            Z += [z.cpu()]

        return TensorDataset(torch.cat(Z), torch.cat(Y))


    def clip_precompute_splits(clip_model, splits, device):
        _splits = []
        for sp in splits:
            dataset, weights = sp
            dataset_new = clip_featurize_data(clip_model, dataset, device)
            _splits.append((dataset_new, weights))

        return _splits


    if args.load_or_save_clip_features and os.path.exists(clip_features_save_path):
        del in_splits, out_splits
        in_splits, out_splits = load_processed_dataset(clip_features_save_path)
        print("Loaded CLIP featurized dataset from {}.".format(clip_features_save_path))
    else:
        in_splits = clip_precompute_splits(algorithm.clip_model, in_splits, device)
        out_splits = clip_precompute_splits(algorithm.clip_model, out_splits, device)
        print("Pre-computed CLIP features.")

    uda_splits = clip_precompute_splits(algorithm.clip_model, uda_splits, device)  # not used
    in_splits_train = in_splits
    out_splits_val = out_splits

    if args.load_or_save_clip_features and not os.path.exists(clip_features_save_path):
        save_processed_dataset(clip_features_save_dir, clip_features_save_name, in_splits, out_splits)
        print("Saved CLIP featurized dataset to {}.".format(clip_features_save_path))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=num_workers)
        for i, (env, env_weights) in enumerate(in_splits_train)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=num_workers)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    if larger_batch:
        eval_batch_size = 256
    else:
        eval_batch_size = 64
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=eval_batch_size,
        num_workers=num_workers)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
                         for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])
    steps_per_epoch = int(min([len(env) / hparams['batch_size'] for env, _ in in_splits]))

    if algorithm.trainable and not args.only_eval:
        if not hparams["use_fix_step"]:
            n_steps = int(hparams['max_epoch'] * steps_per_epoch + 1)
            checkpoint_freq = int(5 * steps_per_epoch)
        else:
            n_steps = hparams['max_step']
            checkpoint_freq = 200
    else:
        n_steps = 1
        checkpoint_freq = 1


    def eval_val_loss(_algorithm, _test_envs=None):
        """Compute validation loss"""

        _algorithm.eval()
        if _test_envs is None:
            _test_envs = []
        _train_envs = [i for i in range(len(out_splits_val)) if i not in _test_envs]

        valid_features, valid_labels = zip(*[out_splits_val[i][0].tensors for i in _train_envs])
        valid_dom_labels = [torch.full((x.shape[0],), i, dtype=torch.int64)
                            for i, x in enumerate(valid_features)]
        valid_features, valid_labels, valid_dom_labels = torch.cat(valid_features), torch.cat(
            valid_labels), torch.cat(valid_dom_labels)
        # use a fixed shuffle of the data across epochs and also trials
        indices = torch.LongTensor(
            np.random.RandomState(seed=split_seed).permutation(valid_features.size()[0]))
        valid_features, valid_labels, valid_dom_labels = valid_features[indices], valid_labels[indices], \
                                                         valid_dom_labels[indices]

        valid_dataloader = DataLoader(
            dataset=TensorDataset(valid_features, valid_labels, valid_dom_labels),
            batch_size=eval_batch_size * 2,
            num_workers=num_workers,
            drop_last=True)

        val_loss = misc.loss(_algorithm, valid_dataloader, device)
        _algorithm.train()
        return val_loss


    def average_by_filtering(result_dict, cond_func):
        values = [v for k, v in result_dict.items() if cond_func(k)]
        return np.mean(values)


    def eval_acc(_algorithm, _test_envs):
        """Evaluate accuracy on all splits"""

        _algorithm.eval()
        assert len(_test_envs) > 0
        _train_envs = [i for i in range(num_envs) if i not in _test_envs]

        clf_train_features, clf_train_labels = zip(*[in_splits[i][0].tensors for i in _train_envs])
        clf_train_features, clf_train_labels = torch.cat(clf_train_features), torch.cat(clf_train_labels)
        clf_train_dataloader = DataLoader(
            dataset=TensorDataset(clf_train_features, clf_train_labels),
            batch_size=512,
            num_workers=num_workers)

        # validation selection for training classifier
        clf_valid_features, clf_valid_labels = zip(*[out_splits[i][0].tensors for i in _train_envs])
        clf_valid_features, clf_valid_labels = torch.cat(clf_valid_features), torch.cat(clf_valid_labels)
        # use a fixed shuffle of the data across epochs and also trials
        indices = torch.LongTensor(
            np.random.RandomState(seed=split_seed).permutation(clf_valid_features.size()[0]))
        clf_valid_features, clf_valid_labels = clf_valid_features[indices], clf_valid_labels[indices]

        clf_valid_dataloader = DataLoader(
            dataset=TensorDataset(clf_valid_features, clf_valid_labels),
            batch_size=512,
            num_workers=num_workers)
        clf_train_data = algorithm.preprocess_features(clf_train_dataloader)
        clf_valid_data = algorithm.preprocess_features(clf_valid_dataloader)

        _algorithm.fit_classifier(clf_train_data, clf_valid_data)

        _results = {}
        evals = zip(eval_loader_names, eval_loaders, eval_weights)
        for name, loader, weights in evals:
            acc = misc.accuracy(_algorithm, loader, weights, device)
            _results[name + '_acc'] = acc

        summary_results = {}
        summary_results.update(
            {
                'in_acc_all': average_by_filtering(_results, lambda k: 'in_acc' in k),
                'in_acc_src': average_by_filtering(_results,
                                                   lambda k: 'in_acc' in k and k not in [f'env{env_i}_in_acc' for env_i
                                                                                         in _test_envs]),
                'in_acc_tgt': average_by_filtering(_results,
                                                   lambda k: k in [f'env{env_i}_in_acc' for env_i in _test_envs]),
                'out_acc_all': average_by_filtering(_results, lambda k: 'out_acc' in k),
                'out_acc_src': average_by_filtering(_results,
                                                    lambda k: 'out_acc' in k and k not in [f'env{env_i}_out_acc' for
                                                                                           env_i in _test_envs]),
                'out_acc_tgt': average_by_filtering(_results,
                                                    lambda k: k in [f'env{env_i}_out_acc' for env_i in _test_envs]),
            }
        )
        _results.update(summary_results)

        _algorithm.train()
        return summary_results, _results


    last_results_keys = None
    best_valid_loss = np.inf
    best_valid_results = None
    best_valid_loss_all_doms = np.inf
    best_valid_results_all_doms = None
    best_valid_model_all_doms = None
    only_eval_last = args.dataset in ['OfficeHome', 'DomainNet']

    for step in range(start_step, n_steps):
        if algorithm.trainable and not args.only_eval:
            step_start_time = time.time()

            minibatches_device = [(x.to(device), y.to(device)) for x, y in next(train_minibatches_iterator)]
            update_args = (minibatches_device, None)

            algorithm.adjust_lr(step, n_steps, steps_per_epoch)
            step_vals = algorithm.update(*update_args)
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

            if algorithm.trainable:
                valid_loss = eval_val_loss(algorithm, args.test_envs)
                results.update({'val_loss': valid_loss})

                valid_loss_all_doms = eval_val_loss(algorithm, None)
                results.update({'val_loss_all_doms': valid_loss_all_doms})

            if (not only_eval_last) or (step == n_steps - 1):
                eval_results, full_eval_results = eval_acc(algorithm, args.test_envs)
                results.update(eval_results)

            exclude_keys = ['env']
            results_keys = [k for k in sorted(results.keys()) if np.all([ex_key not in k for ex_key in exclude_keys])]
            colwidth = max(12, max([len(k) for k in results_keys]))
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=colwidth)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=colwidth)

            if algorithm.trainable:
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_valid_results = results.copy()

                if valid_loss_all_doms < best_valid_loss_all_doms:
                    best_valid_loss_all_doms = valid_loss_all_doms
                    best_valid_results_all_doms = results.copy()
                    best_valid_model_all_doms = copy.deepcopy(algorithm.state_dict())  # need deep copy

            if args.wandb_logger:
                wandb.log({f'valid/{k}': v for k, v in results.items()}, step=step)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            if (not only_eval_last and step > 0) or (step == n_steps - 1):
                results.update({"full_eval_results": full_eval_results})

                epochs_path = os.path.join(args.output_dir, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            # save checkpoint if needed, but the checkpoint size may be large
            # save_checkpoint(f'model_step{step}.pkl')

    if args.wandb_logger and algorithm.trainable:
        wandb.log({f'best/val_loss': best_valid_loss})
        wandb.log({f'best/{k}': v for k, v in best_valid_results.items()})
        wandb.log({f'best_all_doms/val_loss': best_valid_loss_all_doms})
        wandb.log({f'best_all_doms/{k}': v for k, v in best_valid_results_all_doms.items()})

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
