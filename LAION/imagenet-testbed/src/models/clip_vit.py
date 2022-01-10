from registry import registry
from models.model_base import Model
from models.model_base import Model, StandardTransform, StandardNormalization
import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, '../../..')
sys.path.append(base_dir)
from scripts.clip_imagenet_utils import build_clip_imagenet_model, clip_transform


def gen_classifier_loader(path):
    def classifier_loader():
        model = build_clip_imagenet_model(path)[0]
        return model

    return classifier_loader

clip_vit_base_path = os.path.join(base_dir, 'checkpoints/clip_laion_{}/model_with_logistic.pkl')
clip_vit_finetuned_models = {
    # CLIP pretrained
    'clip_vit_pretrained': clip_vit_base_path.format('CLIPPretrained_0_3e-3'),
    # CLIP finetuned without bottlenecks
    'clip_vit_finetuned_base': clip_vit_base_path.format('ContrastCLIPBottleneckBase_0_3e-3'),
    # CLIP finetuned with entropy bottleneck
    'clip_vit_finetuned_ent': clip_vit_base_path.format('ContrastCLIPBottleneckEnt_1_1e-3'),
}

for name, path in clip_vit_finetuned_models.items():
    if os.path.exists(path):
        registry.add_model(
            Model(
                name=name,
                transform=clip_transform(224),
                normalization=None,
                classifier_loader=gen_classifier_loader(path),
                eval_batch_size=256,

                # OPTIONAL
                arch='clip_vit',
            )
        )
    else:
        print("Warning! `{}` ({}) not exists!".format(name, path))
