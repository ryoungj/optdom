import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.nn import functional as F
from torch.nn.functional import softmax
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics.functional import accuracy


class CLIPConLoss(nn.Module):
    """CLIP text-image contrastive loss"""

    def __init__(
            self,
            feature_dim,
            temperature=0.07,
            learnable_temperature=True,
            is_project=False,  # whether apply projection head
            is_symmetric=True,  # whether use a symmetric text-image contrastive loss
    ):
        super(CLIPConLoss, self).__init__()
        self.temperature = temperature
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        else:
            self.logit_scale = torch.ones([]) * np.log(1 / self.temperature)

        self.is_project = is_project
        self.is_symmetric = is_symmetric

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim),
            )

    def forward(self, z, text_features):
        device = z.device

        if self.is_project:
            z = self.project(z)

        image_features = z / z.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale = torch.clamp(logit_scale, max=100)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        y = torch.arange(z.shape[0]).to(z.device)  # diagonal

        if not self.is_symmetric:
            loss = F.cross_entropy(logits_per_image, y)
        else:
            loss_i = F.cross_entropy(logits_per_image, y)
            loss_t = F.cross_entropy(logits_per_text, y)

            loss = loss_t + loss_i

        return loss


def finite_mean(x):
    # only 1D for now
    num_finite = (torch.isfinite(x).float()).sum()
    # fixme: this is still problematic and will give NaN grad
    mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
    if num_finite != 0:
        mean = mean / num_finite
    else:
        return torch.tensor(0.0).to(x)
    return mean


class PLLogisticRegression(pl.LightningModule):
    """
    Logistic regression model
    """

    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            bias: bool = True,
            learning_rate: float = 1e-4,
            optimizer: Optimizer = Adam,
            l1_strength: float = 0.0,
            l2_strength: float = 0.0,
            is_nonlinear: bool = False,
            **kwargs
    ):
        """
        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default='Adam')
            l1_strength: L1 regularization strength (default=None)
            l2_strength: L2 regularization strength (default=None)
            is_nonlinear: whether use a nonlinear classifier
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer

        if not self.hparams.is_nonlinear:
            self.linear = nn.Linear(in_features=self.hparams.input_dim, out_features=self.hparams.num_classes,
                                    bias=bias)
        else:
            hidden_dim = self.hparams.input_dim * 2
            self.linear = nn.Sequential(
                nn.Linear(self.hparams.input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.hparams.num_classes),
            )

    def forward(self, x):
        logits = self.linear(x)
        # y_hat = softmax(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch

        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self(x)

        # PyTorch cross_entropy function combines log_softmax and nll_loss in single function
        loss = F.cross_entropy(y_hat, y, reduction='sum')

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.hparams.l2_strength * l2_reg

        loss /= x.size(0)
        acc = accuracy(y_hat, y)

        tensorboard_logs = {'train_ce_loss': loss, 'train_acc': acc}
        progress_bar_metrics = tensorboard_logs
        self.log_dict(tensorboard_logs, prog_bar=True)
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        acc = accuracy(y_hat, y)
        tensorboard_logs = {'val_loss': F.cross_entropy(y_hat, y), 'acc': acc}
        self.log_dict(tensorboard_logs, prog_bar=True)
        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_ce_loss': val_loss, 'val_acc': acc}
        progress_bar_metrics = tensorboard_logs
        self.log_dict(tensorboard_logs)
        return {'val_loss': val_loss, 'log': tensorboard_logs, 'progress_bar': progress_bar_metrics}

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--input_dim', type=int, default=None)
        parser.add_argument('--num_classes', type=int, default=None)
        parser.add_argument('--bias', default='store_true')
        parser.add_argument('--batch_size', type=int, default=16)
        return parser


clip_prompt_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
