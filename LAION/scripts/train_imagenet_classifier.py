import argparse
import os
import random
import sys

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset, Dataset

from scripts.clip_imagenet_utils import build_clip_imagenet_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a linear classifier with pretrained encoder on ImageNet')
    parser.add_argument('--seed', type=int, default=0, help='Seed for everything else')
    parser.add_argument('--model_ckpt_path', type=str, help='Pretrained encoder checkpoint path')
    parser.add_argument('--save_ckpt_path', type=str, default=None, help='Path to save model')
    parser.add_argument('--load_or_save_features', action='store_true', help='Load or save preprocessed features')
    parser.add_argument('--always_rerun', action='store_true')
    parser.add_argument('--clf_type', type=str, default='linear', help='The linear classifier type',
                        choices=['LogisticPT', 'ZeroShot'])
    parser.add_argument('--clf_train_lr', type=float, default=3e-4, help='Learning rate for training linear classifier')
    parser.add_argument('--clf_train_l2_reg', type=float, default=1e-5,
                        help='L2 regularization for training linear classifier')
    parser.add_argument('--dataset_dir', type=str, help="Dataset directory")
    args = parser.parse_args()

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if os.path.exists(args.save_ckpt_path) and not args.always_rerun:
        print("Job already done!")
        exit(0)


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device, precision = "cuda", 32
    else:
        device, precision = "cpu", 32

    # Load pretrained encoder
    model, preprocess, save_dict = build_clip_imagenet_model(args.model_ckpt_path)
    model.to(device)

    # Load ImageNet dataset
    train_dataset = torchvision.datasets.ImageNet(args.dataset_dir, split="train", transform=preprocess)
    val_dataset = torchvision.datasets.ImageNet(args.dataset_dir, split="val", transform=preprocess)

    # Prepare model
    model.hparams["clf_type"] = args.clf_type
    save_dict["model_hparams"]["clf_type"] = args.clf_type
    if args.clf_type == 'ZeroShot':
        model.fit_classifier((torch.zeros((1, save_dict['model_feature_dim'])), None), (None, None),
                             prompt_engineer=True)
    else:
        features_save_path = os.path.join(os.path.dirname(args.save_ckpt_path), 'processed_features.pt')
        if args.load_or_save_features and os.path.exists(features_save_path):
            processed_features = torch.load(features_save_path)
            clf_train_data = processed_features['train']
            clf_valid_data = processed_features['val']
            print("Loaded processed features from {}".format(features_save_path))
        else:
            torch.multiprocessing.set_sharing_strategy('file_system')
            num_workers = 4
            clf_train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, num_workers=num_workers)
            clf_valid_dataloader = DataLoader(dataset=val_dataset, batch_size=128, num_workers=num_workers)
            clf_train_data = model.preprocess_features(clf_train_dataloader, return_tensor=True, use_tqdm=True)
            clf_valid_data = model.preprocess_features(clf_valid_dataloader, return_tensor=True, use_tqdm=True)

        if args.load_or_save_features and not os.path.exists(features_save_path):
            processed_features = {'train': clf_train_data, 'val': clf_valid_data}
            torch.save(processed_features, features_save_path)
            print("Saved processed features from {}".format(features_save_path))

        clf_train_data = map(lambda t: t.cpu().numpy(), clf_train_data)
        clf_valid_data = map(lambda t: t.cpu().numpy(), clf_valid_data)
        train_clf_hparams = {
            'precision': precision,
            'lr': args.clf_train_lr,
            'batch_size': 512,
            'max_epochs': 500,
            'l2_reg': args.clf_train_l2_reg,
        }
        model.fit_classifier(clf_train_data, clf_valid_data, clf_valid_data, train_clf_hparams=train_clf_hparams)

    # Save model
    # for adaptability, very tricky here
    assert isinstance(model.transform, torch.nn.Sequential)
    model.transform = model.transform[1]
    save_dict.update({"model_dict": model.state_dict()})
    torch.save(save_dict, args.save_ckpt_path)
    print('Saved model checkpoint to {}!'.format(args.save_ckpt_path))
