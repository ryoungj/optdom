import clip
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, PredefinedSplit
import language_tool_python
from tqdm import tqdm

from domainbed.algorithms import Algorithm
from domainbed.utils import CLIPConLoss, clip_prompt_templates, PLLogisticRegression
from domainbed.bottlenecks import *

from pl_bolts.datamodules import SklearnDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

CLIP_ALGORITHMS = [
    'CLIPPretrained',
    'SupCLIPBottleneckBase',
    'SupCLIPBottleneckEnt',
    'SupCLIPBottleneckCondCAD',
    'SupCLIPBottleneckCAD',
    'ContrastCLIPBottleneckBase',
    'ContrastCLIPBottleneckEnt',
    'ContrastCLIPBottleneckCAD',
]


class AbstractCLIPAlgorithm(Algorithm):
    """CLIP based algorithms (abstract class)"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams, pretrained, idx2class):
        super(AbstractCLIPAlgorithm, self).__init__(feature_dim, num_classes, num_domains, hparams)
        self.clip_model = pretrained
        self.num_classes = num_classes
        self.idx2class = idx2class
        if self.clip_model is not None:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # to be overrided by subclasses
        self.transform = None  # transform CLIP features
        self.bottleneck = None  # domain bottleneck
        self.classifier = None  # task classifier
        self.is_debug = hparams["debug"]  # debug mode

    def get_clip_label_text_features(self, normalize=True, multiple_prompts=False):
        """Get CLIP features of label text prompts

        Args:
            normalize: whether normalize the output text features
            multiple_prompts: whether apply prompt engineering with multiple prompts
        """
        device = next(self.clip_model.parameters()).device
        class_names = [self.idx2class[idx] for idx in range(len(self.idx2class.items()))]

        if not multiple_prompts:
            tool = language_tool_python.LanguageTool('en-US')
            text_inputs = torch.cat([clip.tokenize(tool.correct(f"a picture of a {c}")) for c in class_names]).to(
                device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_inputs)
                if normalize:
                    text_features /= text_features.norm(dim=-1, keepdim=True)
        else:
            is_training = self.clip_model.training
            self.clip_model.eval()
            with torch.no_grad():
                text_features = []
                for classname in tqdm(class_names):
                    texts = [template.format(classname) for template in clip_prompt_templates]  # format with class
                    texts = clip.tokenize(texts).to(device)  # tokenize
                    class_embeddings = self.clip_model.encode_text(texts)  # embed with text encoder
                    if normalize:
                        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    if normalize:
                        class_embedding /= class_embedding.norm()
                    text_features.append(class_embedding)
                text_features = torch.stack(text_features, dim=0).to(device)

            if is_training:
                self.clip_model.train()

        text_features = text_features.float()
        return text_features

    def get_device(self):
        """Get model device"""
        if self.clip_model is not None:
            device = next(self.clip_model.parameters()).device
        else:
            device = next(self.transform.parameters()).device

        return device

    def get_transformed_feature(self, all_x):
        """Get the transformed feature of a batch samples"""
        all_z = self.transform(all_x)
        return all_z

    def preprocess_features(self, loader, return_tensor=False, use_tqdm=False):
        """Get the finetuned features (that can be directly used for training classifier) for a whole dataset
        Args:
            loader: the dataset loader
            return_tensor: whether return features as tensors (True) or numpy aarrays (False)
            use_tqdm: use tqdm to visualize progress
        """
        assert not self.training, "Should be in the evaluation mode!!!"
        device = self.get_device()

        if use_tqdm:
            loader = tqdm(loader)

        with torch.no_grad():
            Z, Y = [], []
            for x, y in loader:
                Z += [self.bottleneck(self.get_transformed_feature(x.to(device))).cpu().numpy()]
                Y += [y.cpu().numpy()]

        if return_tensor:
            return torch.tensor(np.concatenate(Z)), torch.tensor(np.concatenate(Y))
        else:
            return np.concatenate(Z), np.concatenate(Y)

    def loss(self, all_x, all_y, all_d):
        """Compute the loss"""
        raise NotImplementedError

    def update(self, minibatches, unlabeled=None):
        """Update the model with a batch"""
        raise NotImplementedError

    def fit_classifier(self, clf_train_data, clf_valid_data, prompt_engineer=False, train_clf_hparams=None):
        """Fit classifier
        The classifier types include:
        - 'SVM' or 'Logistic' for sklearn classifiers
        - 'LogisticPT' for pytorch implemented logistic regression, used with large dataset like DomainNet for
          minibatch training
        - 'ZeroShot' for CLIP zero-shot classifier with label prompts, note that it works with pretrained CLIP or CLIP
          finetuned with image-text contrastive loss but not supervised cross-entropy loss

        Args:
            clf_train_data: training data
            clf_valid_data: validation data
            prompt_engineer: whether use multiple prompts with prompt engineering for 'ZeroShot' classifier
            train_clf_hparams: hyperparameter dict for training the pytorch implemented logistic regression
        """
        device = self.get_device()
        clf_type = self.hparams['clf_type']
        assert clf_type in ['SVM', 'Logistic', "LogisticPT", "ZeroShot"]
        use_sklearn = clf_type in ['SVM', 'Logistic']
        print("Fitting classifier: {}...".format(clf_type))

        # (clf_train_data, clf_valid_data) should be processed by self.preprocess_features
        clf_train_features, clf_train_labels = clf_train_data
        clf_val_features, clf_val_labels = clf_valid_data

        if use_sklearn:
            clf_all_features = np.concatenate([clf_train_features, clf_val_features])
            clf_all_labels = np.concatenate([clf_train_labels, clf_val_labels])
            cv_fold = np.concatenate([
                np.full(clf_train_features.shape[0], -1, dtype=np.int8),  # setting training data to -1
                np.zeros(clf_val_features.shape[0], dtype=np.int8),  # setting validation data to 0
            ])
            cv = PredefinedSplit(cv_fold)

            # Perform linear classification
            # print("Tuning hyper parameters for linear classifer...")
            if clf_type == 'SVM':
                base_params = {'penalty': 'l2', 'max_iter': 1000, 'verbose': 0}
                base_estimator_class = LinearSVC
            elif clf_type == 'Logistic':
                base_params = {'penalty': 'l2', 'max_iter': 1000, 'multi_class': 'multinomial', 'solver': 'lbfgs',
                               'verbose': 0, 'n_jobs': -1, 'warm_start': False}
                base_estimator_class = LogisticRegression
            else:
                raise NotImplementedError

            base_estimator = base_estimator_class(**base_params)

            if self.is_debug:
                best_param = {'C': 1.}
            else:
                c_range = [1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3]
                param_grid = {'C': c_range}

                clf_cv = GridSearchCV(base_estimator, param_grid, cv=cv, refit=False,
                                      # no refit because we can't use target data
                                      scoring='accuracy', n_jobs=-1, error_score='raise', verbose=0)
                clf_cv.fit(clf_all_features, clf_all_labels)
                best_param = clf_cv.best_params_
                # print("Best Params:", best_param)
                if best_param['C'] in [c_range[0], c_range[-1]]:
                    print(f'The best param {best_param} hits the boundary! Please use a larger range!')

            clf = base_estimator_class(**best_param, **base_params)
            clf.fit(clf_train_features, clf_train_labels)

            if clf_type == 'Logistic':
                self.classifier = lambda z: torch.Tensor(clf.predict_proba(z.cpu().numpy())).to(device)
            else:
                self.classifier = lambda z: torch.Tensor(clf.decision_function(z.cpu().numpy())).to(device)
        elif clf_type == "LogisticPT":
            precision = 32
            lr = 5e-4
            batch_size = 512
            max_epochs = 500
            l2_reg = 0.0
            if train_clf_hparams is not None:
                assert isinstance(train_clf_hparams, dict)

                if 'precision' in train_clf_hparams:
                    precision = train_clf_hparams['precision']

                if 'lr' in train_clf_hparams:
                    lr = train_clf_hparams['lr']

                if 'batch_size' in train_clf_hparams:
                    batch_size = train_clf_hparams['batch_size']

                if 'max_epochs' in train_clf_hparams:
                    max_epochs = train_clf_hparams['max_epochs']

                if 'l2_reg' in train_clf_hparams:
                    l2_reg = train_clf_hparams['l2_reg']

            print("Training PyTorch logistic regression hyperparamters:\n"
                  "\tprecision: {}\n"
                  "\tlearning rate: {}\n"
                  "\tl2 regularization: {}\n"
                  "\tbatch size: {}\n"
                  "\tmax epochs: {}\n".format(precision, lr, l2_reg, batch_size, max_epochs))

            dm = SklearnDataModule(clf_train_features, clf_train_labels, x_val=clf_val_features, y_val=clf_val_labels,
                                   x_test=None, y_test=None, val_split=0, test_split=0, num_workers=4,
                                   shuffle=True, batch_size=batch_size, pin_memory=True, drop_last=False)
            self.classifier = PLLogisticRegression(input_dim=clf_train_features.shape[-1], num_classes=self.num_classes,
                                                   learning_rate=lr, l2_strength=l2_reg)

            # fit
            early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.0005, patience=3, verbose=True,
                                                mode="max")
            trainer = pl.Trainer(gpus=1, precision=precision, auto_lr_find=False, max_epochs=max_epochs,
                                 logger=False, checkpoint_callback=False,
                                 flush_logs_every_n_steps=50, progress_bar_refresh_rate=50,
                                 callbacks=[early_stop_callback])
            trainer.fit(self.classifier, train_dataloader=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

            trainer.validate(self.classifier, val_dataloaders=dm.val_dataloader())

            self.classifier.to(device)
        elif clf_type == "ZeroShot":
            if self.classifier is None:
                # a dummy linear layer
                self.classifier = PLLogisticRegression(input_dim=clf_train_features.shape[-1],
                                                       num_classes=self.num_classes)
                self.classifier.to(device)

                text_features = self.get_clip_label_text_features(multiple_prompts=prompt_engineer)
                self.classifier.linear.weight.data.copy_(text_features)
                self.classifier.linear.bias.data.copy_(torch.zeros_like(self.classifier.linear.bias))
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        assert self.classifier is not None, "Please fit the classifier by calling `fit_classifier` first!"
        z = self.bottleneck(self.get_transformed_feature(x))
        if self.hparams['clf_type'] == "ZeroShot":
            z /= z.norm(dim=-1, keepdim=True)
        return self.classifier(z)

    @property
    def trainable(self):
        return True

    def adjust_lr(self, step, max_steps, steps_per_epoch):
        learning_rate = self.hparams["lr"]
        warmup_from = self.hparams["lr"] / 5
        warm_epochs = 10
        lr_decay_rate = 0.1
        lr_decay_epochs = [25, 40]
        eta_min = self.hparams["lr"] * (lr_decay_rate ** 3)

        if self.hparams['warmup'] and (step <= warm_epochs * steps_per_epoch):
            if self.hparams['cosine_anneal']:
                warmup_to = eta_min + (learning_rate - eta_min) * (
                        1 + math.cos(math.pi * warm_epochs * steps_per_epoch / max_steps)) / 2
            else:
                warmup_to = learning_rate

            p = step / (warm_epochs * steps_per_epoch)
            lr = warmup_from + p * (warmup_to - warmup_from)
        elif self.hparams['cosine_anneal']:
            p = step / max_steps
            lr = eta_min + (learning_rate - eta_min) * (
                    1 + math.cos(math.pi * p)) / 2
        else:
            decay_steps = np.sum(step > (np.asarray(lr_decay_epochs) * steps_per_epoch))
            if decay_steps > 0:
                lr = learning_rate * (lr_decay_rate ** decay_steps)
            else:
                lr = learning_rate

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class CLIPPretrained(AbstractCLIPAlgorithm):
    """Pretrained CLIP model"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams, pretrained, idx2class):
        super(CLIPPretrained, self).__init__(feature_dim, num_classes, num_domains, hparams, pretrained, idx2class)
        self.transform = lambda x: x  # features are precomputed
        self.bottleneck = lambda x: x
        self.featurizer = self.clip_model.visual

    def update(self, minibatches, unlabeled=None):
        return {}

    def loss(self, all_x, all_y, all_d):
        return {}

    @property
    def trainable(self):
        return False


class AbstractCLIPBottleneck(AbstractCLIPAlgorithm):
    """CLIP based algorithms with an additional bottleneck (abstract class)"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams, pretrained, idx2class,
                 bottleneck_class, use_clip_contrast=False):
        """
        Args:
            feature_dim: dimension of CLIP output features
            num_classes: number of classes
            num_domains: number of domains
            hparams: hyperparameter dict
            pretrained: pretrained CLIP model
            idx2class: the dict mapping from indices to class names, used to get label prompts
            bottleneck_class: bottleneck class
            use_clip_contrast: whether use CLIP text-image contrastive loss
        """
        super(AbstractCLIPBottleneck, self).__init__(feature_dim, num_classes, num_domains, hparams, pretrained,
                                                     idx2class)
        assert isinstance(feature_dim, int)
        self.use_clip_contrast = use_clip_contrast
        self.bottleneck = bottleneck_class(feature_dim, num_classes, num_domains, hparams)

        self.transform = torch.nn.Sequential(
            *[networks.CLIPMLP(feature_dim, feature_dim, mlp_width=hparams['mlp_width'],
                               mlp_depth=hparams['mlp_depth'],
                               mlp_dropout=hparams['mlp_dropout'],
                               add_residual=True, add_norm=hparams['mlp_norm']) for _ in
              range(hparams['mlp_blocks'])])

        if not self.use_clip_contrast:
            self.classifier_head = nn.Linear(feature_dim, num_classes, bias=True)
            self.refit_classifier = hparams['refit_classifier']  # whether refit the classifier
            params = list(self.transform.parameters()) + list(self.classifier_head.parameters())
        else:
            assert not self.bottleneck.is_conditional
            self.clipcon = CLIPConLoss(feature_dim, temperature=hparams['temperature'],
                                       learnable_temperature=hparams['learnable_temperature'],
                                       is_project=hparams['is_project'], is_symmetric=hparams['is_symmetric'])
            params = list(self.transform.parameters()) + list(self.clipcon.parameters())

        if self.bottleneck.trainable:
            params += list(self.bottleneck.parameters())

        num_trainable_params = sum([sum(p.numel() for p in param_group.parameters())
                                    if not isinstance(param_group, nn.Parameter)
                                    else param_group.numel()
                                    for param_group in params])
        print("Trainable parameters # : ", num_trainable_params)

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def loss(self, all_x, all_y, all_d):
        all_z = self.get_transformed_feature(all_x)

        if not self.use_clip_contrast:
            # supervised cross-entropy loss
            # `all_y` should be integer-valued labels
            bn_loss, all_z_hat = self.bottleneck.loss(all_z, all_y, all_d)
            clf_out = self.classifier_head(all_z_hat)
            clf_loss = F.cross_entropy(clf_out, all_y)
            total_loss = clf_loss + self.hparams['lmbda'] * bn_loss

            losses = {"clf_loss": clf_loss, "bn_loss": bn_loss, "total_loss": total_loss}
        else:
            # text-image contrastive loss
            # `all_y` should be preprocessed text features
            text_features = all_y  # text are features
            all_y = torch.ones(all_z.shape[0]).to(all_z)  # dummy

            bn_loss, all_z_hat = self.bottleneck.loss(all_z, all_y, all_d)
            clipcon_loss = self.clipcon(all_z_hat, text_features)

            total_loss = clipcon_loss + self.hparams['lmbda'] * bn_loss

            losses = {"clipcon_loss": clipcon_loss, "bn_loss": bn_loss, "total_loss": total_loss}

        return losses

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][1].is_cuda else "cpu"

        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_d = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device=device)
            for i, (x, y) in enumerate(minibatches)
        ])

        _losses = self.loss(all_x, all_y, all_d)
        self.optimizer.zero_grad()
        _losses["total_loss"].backward()
        self.optimizer.step()

        losses = {k: v.item() for k, v in _losses.items()}

        return losses

    def fit_classifier(self, clf_train_data, clf_valid_data, prompt_engineer=False, train_clf_hparams=None):
        if (not self.use_clip_contrast) and (not self.refit_classifier):
            self.classifier = self.classifier_head
        else:
            super().fit_classifier(clf_train_data, clf_valid_data,
                                   prompt_engineer=prompt_engineer, train_clf_hparams=train_clf_hparams)


class SupCLIPBottleneckBase(AbstractCLIPBottleneck):
    """CLIP finetuned with supervised cross-entropy loss but no bottleneck"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams, pretrained, idx2class):
        super(SupCLIPBottleneckBase, self).__init__(feature_dim, num_classes, num_domains, hparams, pretrained,
                                                    idx2class,
                                                    DummyBottleneck)


class SupCLIPBottleneckEnt(AbstractCLIPBottleneck):
    """CLIP finetuned with supervised cross-entropy loss and entropy bottleneck"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams, pretrained, idx2class):
        super(SupCLIPBottleneckEnt, self).__init__(feature_dim, num_classes, num_domains, hparams, pretrained,
                                                   idx2class,
                                                   DiscreteEntropyBottleneck)


class SupCLIPBottleneckCAD(AbstractCLIPBottleneck):
    """CLIP finetuned with supervised cross-entropy loss and CAD bottleneck"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams, pretrained, idx2class):
        super(SupCLIPBottleneckCAD, self).__init__(feature_dim, num_classes, num_domains, hparams, pretrained,
                                                   idx2class,
                                                   CADBottleneck)


class SupCLIPBottleneckCondCAD(AbstractCLIPBottleneck):
    """CLIP finetuned with supervised cross-entropy loss and conditional CAD bottleneck"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams, pretrained, idx2class):
        super(SupCLIPBottleneckCondCAD, self).__init__(feature_dim, num_classes, num_domains, hparams, pretrained,
                                                       idx2class,
                                                       CondCADBottleneck)


class ContrastCLIPBottleneckBase(AbstractCLIPBottleneck):
    """CLIP finetuned with text-image contrastive loss but no bottleneck"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams, pretrained, idx2class):
        super(ContrastCLIPBottleneckBase, self).__init__(feature_dim, num_classes, num_domains, hparams, pretrained,
                                                         idx2class,
                                                         DummyBottleneck,
                                                         use_clip_contrast=True)


class ContrastCLIPBottleneckEnt(AbstractCLIPBottleneck):
    """CLIP finetuned with text-image contrastive loss and entropy bottleneck (no need to access to domain labels)"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams, pretrained, idx2class):
        super(ContrastCLIPBottleneckEnt, self).__init__(feature_dim, num_classes, num_domains, hparams, pretrained,
                                                        idx2class,
                                                        DiscreteEntropyBottleneck,
                                                        use_clip_contrast=True)


class ContrastCLIPBottleneckCAD(AbstractCLIPBottleneck):
    """CLIP finetuned with text-image contrastive loss and CAD bottleneck (require access to domain labels)"""

    def __init__(self, feature_dim, num_classes, num_domains, hparams, pretrained, idx2class):
        super(ContrastCLIPBottleneckCAD, self).__init__(feature_dim, num_classes, num_domains, hparams, pretrained,
                                                        idx2class,
                                                        CADBottleneck,
                                                        use_clip_contrast=True)
