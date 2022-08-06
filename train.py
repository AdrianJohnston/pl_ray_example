import argparse
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from path import Path
from pytorch_lightning.utilities.cli import LightningCLI
# from ray_lightning import RayStrategy
import ray

from network import LeNet

import torchmetrics
from jsonargparse import lazy_instance


#TODO: Add ray_lightning plugin to train the models!
available_models = {
    "LeNet": LeNet
}

MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
     'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
     'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
     '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
     'ec29112dd5afa0611ce80d1b7f02629c')
]


class ImageClassifier(LightningModule):
    def __init__(self, rootpath: str, dataset_folder: str, dataset_name: str,
                 dataset_version: str, network_backbone: str,
                 batch_size: int = 32, num_workers: int = 8, lr: float = 0.001,
                 weight_decay: float = 0.0005) -> None:
        super().__init__()

        self.rootpath = Path(rootpath)
        self.dataroot = self.rootpath / Path(dataset_folder)
        self.dataset = dataset_name
        self.version = str(dataset_version)
        self.network_backbone = network_backbone
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.__build_model()
        print(self.network)

        self.validation_metrics = nn.ModuleDict(
            {'accuracy': torchmetrics.classification.Accuracy()})

    def __build_model(self):
        self.network = available_models[self.network_backbone]()

    def train_dataloader(self):
        # transforms
        # prepare transforms standard to MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        # data
        train_dataset = MNIST(root=self.dataroot,
                              train=True,
                              download=True,
                              transform=transform)
        return DataLoader(train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        # transforms
        # prepare transforms standard to MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        # data
        val_dataset = MNIST(root=self.dataroot,
                            train=False,
                            transform=transform,
                            download=True)
        return DataLoader(val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.network(inputs)

        val_loss = F.nll_loss(outputs, targets)
        preds = outputs.argmax(
            dim=1)  # get the index of the max log-probability

        results = {'val/loss': val_loss}
        for metric_name, metric in self.validation_metrics.items():
            metric_result = metric(preds, targets)

        self.log('val/loss', val_loss, prog_bar=True)
        return {'val_loss': val_loss}

    def forward(self, images):
        return self.network(images)

    def validation_epoch_end(self, val_outputs):
        acc = self.validation_metrics['accuracy'].compute().cpu().item() * 100
        self.log('val/accuracy', acc, prog_bar=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.lr,
            # momentum=self.cfg.optimisation.momentum,
            weight_decay=self.weight_decay)

        return optimizer

from argparse import ArgumentParser
parser = ArgumentParser()

@ray.remote(num_gpus=1)
def train() -> None:

    cli = LightningCLI(ImageClassifier,
                       seed_everything_default=1337,
                       save_config_overwrite=True,
                       run=False,
                       trainer_defaults={"logger": lazy_instance(TensorBoardLogger, save_dir="logs")})

    # cli.parser.add_argument("--node-ip-address")
    # cli.parser.add_argument(
    #     "--node-manager-port", required=True, type=int, help="the port of the worker's node"
    # )
    # cli.parser.add_argument(
    #     "--raylet-ip-address",
    #     required=False,
    #     type=str,
    #     default=None,
    #     help="the ip address of the worker's raylet",
    # )
    # cli.parser.add_argument(
    #     "--redis-address", required=True, type=str, help="the address to use for Redis"
    # )
    # cli.parser.add_argument(
    #     "--gcs-address", required=True, type=str, help="the address to use for GCS"
    # )
    # cli.parser.add_argument(
    #     "--redis-password",
    #     required=False,
    #     type=str,
    #     default=None,
    #     help="the password to use for Redis",
    # )
    # cli.parser.add_argument(
    #     "--object-store-name", required=True, type=str, help="the object store's name"
    # )

    # model = cli.model
    # print(model)

    #tb_logger = None  # TensorBoardLogger(save_dir='tb-logs')
    # lr_logger = pl.callbacks.LearningRateLogger()

    # early_stopping_cb = None
    # checkpoint_cb = pl.callbacks.ModelCheckpoint('weights/{epoch}',
    #                                              monitor='val/loss',
    #                                              mode='max',
    #                                              save_top_k=3)

    # trainer = pl.Trainer(logger=[tb_logger],
    #                      checkpoint_callback=checkpoint_cb,
    #                      callbacks=[],
    #                      **cfg.trainer)
    # trainer.fit(model)

    #cli.instantiate_classes()
    #cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    print(f"SUCCESS: {torch.cuda.is_available()}")
    return "SUCCESS"

if __name__ == '__main__':
    # main()
    obj_ref = train.remote()
    result = ray.get(obj_ref)
    print(result)
