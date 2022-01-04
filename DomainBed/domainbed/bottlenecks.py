import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.independent import Independent
from torch.distributions.categorical import Categorical

import math
import einops
import numpy as np

from compressai.entropy_models import EntropyBottleneck
from domainbed import networks
from domainbed.utils import finite_mean


class AbstractBottleneck(torch.nn.Module):
    """Domain Bottleneck (abstract class)"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams):
        super(AbstractBottleneck, self).__init__()
        self.hparams = hparams

    def forward(self, z):
        return z

    def loss(self, z, y, dom_labels):
        raise NotImplementedError

    def update(self, z, y, dom_labels):
        pass

    @property
    def trainable(self):
        """Whether the bottleneck has trainable parameters"""
        return False

    @property
    def is_conditional(self):
        """Whether the bottleneck is conditioned on labels"""
        return False


class DummyBottleneck(AbstractBottleneck):
    """Dummy Bottleneck (without bottleneck)"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams):
        super(DummyBottleneck, self).__init__(feature_dim, num_classes, num_domains, hparams)

    def loss(self, z, y, dom_labels):
        dummy_loss = torch.Tensor([0.]).to(z.device)
        return dummy_loss, z

    def update(self, z, y, dom_labels):
        pass

    @property
    def trainable(self):
        return False


class DiscreteEntropyBottleneck(AbstractBottleneck):
    """Entropy Bottleneck (with discretization)
    Introduced by J. Ballé, et al., in “Variational image compression with a scale hyperprior”.

    Properties:
    - Minimize H(Z)
    - Require no access to domain labels and task labels
    """

    def __init__(self, feature_dim, num_classes, num_domains, hparams):
        super(DiscreteEntropyBottleneck, self).__init__(feature_dim, num_classes, num_domains, hparams)
        self.bottleneck = EntropyBottleneck(feature_dim)
        self.scaling = torch.nn.Parameter(torch.ones(feature_dim) * math.log(10))

    def forward(self, z):
        z = z * self.scaling.exp()
        z_hat, _ = self.bottleneck(z.unsqueeze(-1).unsqueeze(-1))
        z_hat = z_hat.squeeze(-1).squeeze(-1) / self.scaling.exp()

        return z_hat

    def loss(self, z, y, dom_labels):
        z = z * self.scaling.exp()
        z_hat, q_z = self.bottleneck(z.unsqueeze(-1).unsqueeze(-1))
        z_hat = z_hat.squeeze() / self.scaling.exp()

        bn_loss = -torch.log(q_z).sum(-1).mean()
        return bn_loss, z_hat

    @property
    def trainable(self):
        return True


class AbstractContrastBottleneck(AbstractBottleneck):
    """Contrastive based bottlenecks (abstract class)
     The implementation is based on the supervised contrastive loss (SupCon) introduced by
     P. Khosla, et al., in “Supervised Contrastive Learning“.
     """

    def __init__(self, feature_dim, num_classes, num_domains, hparams):
        super(AbstractContrastBottleneck, self).__init__(feature_dim, num_classes, num_domains, hparams)
        self.bn_supcon = SupConLoss(feature_dim, num_domains, temperature=hparams['temperature'],
                                    is_normalized=hparams['is_normalized'], is_project=hparams['is_project'])

        # whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        # the two versions have the same optima, but we find the latter is more stable
        self.is_flipped = hparams["is_flipped"]

    def loss(self, z, y, dom_labels):
        return self.bn_supcon(z, y, dom_labels, bn_conditional=self.is_conditional, bn_flipped=self.is_flipped)[1], z

    @property
    def trainable(self):
        return self.bn_supcon.is_project


class CADBottleneck(AbstractContrastBottleneck):
    """Contrastive Adversarial Domain (CAD) bottleneck
    Introduced in Sec 4.2.1 in our paper.

    Properties:
    - Minimize I(D;Z)
    - Require access to domain labels but not task labels
    """

    def __init__(self, feature_dim, num_classes, num_domains, hparams):
        super(CADBottleneck, self).__init__(feature_dim, num_classes, num_domains, hparams)


class CondCADBottleneck(AbstractContrastBottleneck):
    """Conditional Contrastive Adversarial Domain (CAD) bottleneck
    Introduced in Appx C.4 in our paper.

    Properties:
    - Minimize I(D;Z|Y)
    - Require access to both domain labels and task labels
    """

    def __init__(self, feature_dim, num_classes, num_domains, hparams):
        super(CondCADBottleneck, self).__init__(feature_dim, num_classes, num_domains, hparams)

    @property
    def is_conditional(self):
        return True


class SupConLoss(nn.Module):
    """Supervised Contrastive (SupCon) loss
    Introduced by P. Khosla, et al., in “Supervised Contrastive Learning“.
    Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11
    """

    def __init__(
            self,
            feature_dim,
            num_domains,
            temperature=0.07,
            base_temperature=0.07,
            is_logsumexp=True,  # whether take sum in prob space (True) or log prob space (False)
            is_normalized=False,  # whether apply normalization to representation when computing loss
            is_project=False,  # whether apply projection head
    ):
        super(SupConLoss, self).__init__()
        self.num_domains = num_domains
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.is_logsumexp = is_logsumexp
        self.is_normalized = is_normalized
        self.is_project = is_project

        if self.is_project:
            self.project = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, 128),
            )

    def forward(self, z, y, dom_labels, bn_conditional=True, bn_flipped=True):
        """
        Args:
            z: hidden vector of shape [batch_size, z_dim].
            y: ground truth of shape [batch_size].
            dom_labels: ground truth domains of shape [batch_size].
            bn_conditional: if the bottleneck loss conditioned on the label
            bn_flipped: if flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss
        Returns:
            SupCon loss and the bottleneck loss.
        """

        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        dom_labels = dom_labels.contiguous().view(-1, 1)
        mask_y = torch.eq(y, y.T).to(device)
        mask_d = (torch.eq(dom_labels, dom_labels.T)).to(device)
        mask_drop = ~torch.eye(batch_size).bool().to(device)  # drop the "current"/"self" example
        mask_y &= mask_drop
        mask_y_n_d = mask_y & (~mask_d)  # contain the same label but from different domains
        mask_y_d = mask_y & mask_d  # contain the same label and the same domain
        mask_y, mask_drop, mask_y_n_d, mask_y_d = mask_y.float(), mask_drop.float(), mask_y_n_d.float(), mask_y_d.float()

        # compute logits
        if self.is_project:
            z = self.project(z)
        if self.is_normalized:
            z = F.normalize(z, dim=1)
        outer = z @ z.T
        logits = outer / self.temperature
        logits = logits * mask_drop
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log prob
        denominator = torch.logsumexp(logits + mask_drop.log(), dim=1, keepdim=True)
        log_prob = logits - denominator

        # just for numerical stability
        mask_valid = (mask_y.sum(1) > 0)
        log_prob = log_prob[mask_valid]
        mask_y = mask_y[mask_valid]
        mask_y_n_d = mask_y_n_d[mask_valid]
        mask_y_d = mask_y_d[mask_valid]
        mask_d = mask_d[mask_valid]
        logits = logits[mask_valid]
        outer = outer[mask_valid]
        batch_size = log_prob.shape[0]

        if self.is_logsumexp:
            # SupCon In
            agg_log_prob_pos = torch.logsumexp(log_prob + mask_y.log(), dim=1) / mask_y.sum(1) * batch_size
        else:
            # SupCon Out
            agg_log_prob_pos = (mask_y * log_prob).sum(1) / mask_y.sum(1)

        # SupCon loss, not finally used in our paper, but one can use it to replace cross-entropy loss
        loss = -(self.temperature / self.base_temperature) * agg_log_prob_pos

        if not bn_conditional:
            # unconditional CAD loss
            if bn_flipped:  # maximize log prob of samples from different domains
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (~mask_d).float().log(), dim=1)
            else:  # minimize log prob of samples from same domain
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob + (mask_d).float().log(), dim=1)
        else:
            # conditional CAD loss
            if bn_flipped:
                mask_valid = (mask_y_n_d.sum(1) > 0)
            else:
                mask_valid = (mask_y_d.sum(1) > 0)

            log_prob = log_prob[mask_valid]
            mask_y = mask_y[mask_valid]
            mask_y_d = mask_y_d[mask_valid]
            mask_y_n_d = mask_y_n_d[mask_valid]
            outer = outer[mask_valid]
            logits = logits[mask_valid]
            batch_size = log_prob.shape[0]

            # compute log_prob_y with the same label
            denominator = torch.logsumexp(logits + mask_y.log(), dim=1, keepdim=True)
            log_prob_y = logits - denominator

            if bn_flipped:  # maximize log prob of samples from different domains and with same label
                bn_loss = - (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_n_d.log(), dim=1)
            else:  # minimize log prob of samples from same domains and with same label
                bn_loss = (self.temperature / self.base_temperature) * torch.logsumexp(
                    log_prob_y + mask_y_d.log(), dim=1)

        return finite_mean(loss), finite_mean(bn_loss)
