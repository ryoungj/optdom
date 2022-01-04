# OptDom: Learning Optimal Representations for Domain Generalization

This repository contains the official implementation for [Optimal Representations for Covariate Shift](https://arxiv.org/abs/2201.00057)️. 
Our paper theoretically characterizes the minimal sufficient representations for optimal domain generalization (DG) under covariate shift and derives practical self-supervised learning (SSL) objectives for learning such representations.

We provide code for reproducing our main results with contribution highlights:
* Finetuning pretrained SSL models ([CLIP](https://github.com/openai/CLIP)) to be superior robust DG models ️[[minimal example]](#minimal)
* A novel contrastive adversarial domain bottleneck for learning domain-invariant representations ️[[implementation]](DomainBed/domainbed/bottlenecks.py)


## Setup
1. Install PyTorch 1.7.1 and CLIP following the [instructions](https://github.com/openai/CLIP#usage).
2. Install other packages: ``pip install -r requirements.txt``.


## Finetune & Evaluate CLIP on DomainBed
Our paper derives SSL objectives for learning optimally robust representations and gives insights into the superior robustness of CLIP (Sec 4). 
Here we include the code for finetuning CLIP with our proposed objectives and evaluating on the [DomainBed](https://github.com/facebookresearch/DomainBed) benchmark, which reproduces our experiments in Sec 6.2. 

The implementation is included in [DomainBed](DomainBed/) directory which is highly based on the [DomainBed](https://github.com/facebookresearch/DomainBed) repo. 
The CLIP based models are implemented in [domainbed/clip_algorithms.py](DomainBed/domainbed/clip_algorithms.py), and the domain bottlenecks are in [domainbed/bottlenecks.py](DomainBed/domainbed/bottlenecks.py).
The training script for finetuning CLIP with bottlenecks is [domainbed/scripts/train_clip.py](DomainBed/domainbed/scripts/train_clip.py).

### Preparation
Move to the [DomainBed](DomainBed/) directory and download the datasets:
```
python -m domainbed.scripts.download --data_dir ./datasets/
```
By default, we download the datasets: PACS, VLCS, OfficeHome, TerraIncognita, DomainNet.

### Launch a single run
If you want to launch a single run for debugging, run with command:
```
bash run_debug.sh
```
The key arguments include:
* `--dataset`: dataset for finetuning and evaluation.
* `--algorithm`: algorithms implemented with CLIP, see [domainbed/clip_algorithms.py](DomainBed/domainbed/clip_algorithms.py).
* `--test_envs`: list of left-out environments for testing, others used for training/finetuning.
* `--hparams`: JSON-serialized hyperprameter dict, see [domainbed/hparams_registry.py](DomainBed/domainbed/hparams_registry.py) for list of all hyperprameters.

Note that the result of a single run could be very sensitive to hyperparameters and random seed, we recommend to launch a sweep over hyperparameters and random seeds as in DomainBed.

### Launch a sweep with tuning
To launch a sweep, run with command:
```
bash run_sweep_clip.sh
```
A sweep over 10 hyperparameters and 5 random seeds is launched for each dataset and algorithm. 
By default, the CLIP-RN50 model is used, and you can also run with other models by changing the `clip_model` argument, e.g., `ViT-B/32` for CLIP-ViT-B/32.

After the sweep is finished, you can collect result with the notebook [collect_clip_results.ipynb](DomainBed/collect_clip_results.ipynb). Note that the results may be slightly different from the paper due to code cleaning. 


### (Optional) Run CAD in DomainBed setup
You can also evaluate our proposed (conditional) CAD bottleneck in the DomainBed setup where a ResNet-50 is *end-to-end* trained on source domains and evaluated on a left-out target domain. 
We include the implementation in [domainbed/algorithms.py](DomainBed/domainbed/algorithms.py), which you can run with command:
```
bash run_sweep_e2e_dombed.sh
```
Also you can collect result with the notebook [collect_e2e_results.ipynb](DomainBed/collect_e2e_results.ipynb). Note that as the claim of our paper, the algorithms in this setup lack access to the information of the target domain, so we don't expect our bottlenecks and other algorithms to necessarily outperform ERM.
However, our CAD bottleneck does lead to consistent improvement surprisingly.

## Finetune CLIP on LAION-400M
Coming soon!

## Minimal Code for Custom Finetuning <a name="minimal"></a>
If you want to finetune CLIP on your dataset with our bottlenecks, we provide the minimal code example:
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
import clip
from tqdm import tqdm

from domainbed import hparams_registry
from domainbed import algorithms


# 1. Determine whether you do supervised or contrastive finetuning:
#       - True: use a cross-entropy loss with a supervised dataset
#       - False: use a contrastive loss with a text-image-pair dataset
supervised_funetuning = True

if supervised_funetuning:
    loss_name = "Sup"
    dataset_name = "my suervised dataset"
else:
    loss_name = "Contrast"
    dataset_name = "my text-image pair dataset"


# 2. Determine the bottleneck you want to use with different properties
bottleneck_name = "CondCAD"  # Ent, CAD, CondCAD
algorithm_name = loss_name + "CLIPBottleneck" + bottleneck_name


# 3. Set hyperparameters, you can also change the hyperparameter dict and default values
hparams = hparams_registry.default_hparams(algorithm_name, dataset_name)


# 4. Load pretrained CLIP models
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

pretrained, preprocess = clip.load(hparams['clip_model'], device, jit=False)


# 5. Load your dataset, you  dataset should have the form:
#       - (image, label) for supervised finetuning
#       - (image, text) for contrastive finetuning
#    Remember to use the CLIP preprocessing function for image transformation,
#       and your dataset should be a list of sub-datasets from different domains (singleton for a single domain)
dataset = load_your_dataset(dataset_name, preprocess)
num_envs = len(dataset)
num_classes = dataset.num_classes  # dummy for text-image-pair dataset


# 6. Featurize your dataset with CLIP models

def get_clip_feature(clip_model, x, y):
    """Compute CLIP features"""
    with torch.no_grad():
        z = clip_model.encode_image(x).float()
        if not supervised_funetuning:  # `y` is a batch of texts that should be tokenized
            y = clip_model.encode_text(clip.tokenize(y)).float()
    return z, y

def clip_featurize_data(clip_model, dataset, device):
    """Featurize a dataset"""
    Z, Y = [], []
    for x, y in tqdm(DataLoader(dataset, batch_size=512, num_workers=4)):
        z, y = get_clip_feature(clip_model, x.to(device), y.to(device))
        Z += [z.cpu()]
        Y += [y.cpu()]
    return TensorDataset(torch.cat(Z), torch.cat(Y))

def clip_precompute_splits(clip_model, splits, device):
    _splits = []
    for ds in splits:
        _splits.append(clip_featurize_data(clip_model, ds, device))
    return _splits


dataset = clip_precompute_splits(pretrained, dataset, device)
train_loaders = [DataLoader(
    dataset=env,
    batch_size=hparams['batch_size'],
    num_workers=4)
    for i, env in enumerate(dataset)]
train_minibatches_iterator = zip(*train_loaders)
steps_per_epoch = int(min([len(env) / hparams['batch_size'] for env in dataset]))
n_steps = hparams['max_step']


# 7. Initialize the model:
algorithm_class = algorithms.get_algorithm_class(algorithm_name)
algorithm = algorithm_class(pretrained.visual.output_dim, num_classes, num_envs, hparams, pretrained, None)
algorithm.to(device)
algorithm.train()


# 8. Finetune the model:
for step in range(n_steps):
    minibatches_device = [(x.to(device), y.to(device)) for x, y in next(train_minibatches_iterator)]
    algorithm.adjust_lr(step, n_steps, steps_per_epoch)
    step_vals = algorithm.update(minibatches_device, None)
```

## Cite
If you find this work relevant to your work, please cite our paper:
```
@article{ruan2021optdom,
  title={Optimal Representations for Covariate Shift},
  author={Ruan, Yangjun and  Dubois, Yann and Maddison, Chris J},
  journal={arXiv preprint arXiv:2201.00057},
  year={2022},
}
```


## Acknowledgement
Our code is based on:
 * [DomainBed](https://github.com/facebookresearch/DomainBed) (commit `5c1c190`)
 * [Imagenet Testbed](https://github.com/modestyachts/imagenet-testbed) (commit `92914bd`)
