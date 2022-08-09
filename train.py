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
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser
# from ray_lightning import RayStrategy
import ray

from network import LeNet

import torchmetrics
from jsonargparse import lazy_instance

# TODO: Add ray_lightning plugin to train the models!
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

# import ray_cli.ray_constants as ray_constants
# class XaminCLI(LightningCLI):

#     def add_arguments_to_parser(self, parser):
#         parser.add_argument(
#             "--node-ip-address",
#             required=True,
#             type=str,
#             help="the ip address of the worker's node",
#         )
#         parser.add_argument(
#             "--node-manager-port", required=True, type=int, help="the port of the worker's node"
#         )
#         parser.add_argument(
#             "--raylet-ip-address",
#             required=False,
#             type=str,
#             default=None,
#             help="the ip address of the worker's raylet",
#         )
#         parser.add_argument(
#             "--redis-address", required=True, type=str, help="the address to use for Redis"
#         )
#         parser.add_argument(
#             "--gcs-address", required=True, type=str, help="the address to use for GCS"
#         )
#         parser.add_argument(
#             "--redis-password",
#             required=False,
#             type=str,
#             default=None,
#             help="the password to use for Redis",
#         )
#         parser.add_argument(
#             "--object-store-name", required=True, type=str, help="the object store's name"
#         )
#         parser.add_argument("--raylet-name", required=False,
#                             type=str, help="the raylet's name")
#         parser.add_argument(
#             "--logging-level",
#             required=False,
#             type=str,
#             default=ray_constants.LOGGER_LEVEL,
#             choices=ray_constants.LOGGER_LEVEL_CHOICES,
#             help=ray_constants.LOGGER_LEVEL_HELP,
#         )
#         parser.add_argument(
#             "--logging-format",
#             required=False,
#             type=str,
#             default=ray_constants.LOGGER_FORMAT,
#             help=ray_constants.LOGGER_FORMAT_HELP,
#         )
#         parser.add_argument(
#             "--temp-dir",
#             required=False,
#             type=str,
#             default=None,
#             help="Specify the path of the temporary directory use by Ray process.",
#         )
#         parser.add_argument(
#             "--storage",
#             required=False,
#             type=str,
#             default=None,
#             help="Specify the persistent storage path.",
#         )
#         parser.add_argument(
#             "--load-code-from-local",
#             default=False,
#             action="store_true",
#             help="True if code is loaded from local files, as opposed to the GCS.",
#         )
#         parser.add_argument(
#             "--use-pickle",
#             default=False,
#             action="store_true",
#             help="True if cloudpickle should be used for serialization.",
#         )
#         parser.add_argument(
#             "--worker-type",
#             required=False,
#             type=str,
#             default="WORKER",
#             help="Specify the type of the worker process",
#         )
#         parser.add_argument(
#             "--metrics-agent-port",
#             required=True,
#             type=int,
#             help="the port of the node's metric agent.",
#         )
#         parser.add_argument(
#             "--object-spilling-config",
#             required=False,
#             type=str,
#             default="",
#             help="The configuration of object spilling. Only used by I/O workers.",
#         )
#         parser.add_argument(
#             "--logging-rotate-bytes",
#             required=False,
#             type=int,
#             default=ray_constants.LOGGING_ROTATE_BYTES,
#             help="Specify the max bytes for rotating "
#             "log file, default is "
#             f"{ray_constants.LOGGING_ROTATE_BYTES} bytes.",
#         )
#         parser.add_argument(
#             "--logging-rotate-backup-count",
#             required=False,
#             type=int,
#             default=ray_constants.LOGGING_ROTATE_BACKUP_COUNT,
#             help="Specify the backup count of rotated log file, default is "
#             f"{ray_constants.LOGGING_ROTATE_BACKUP_COUNT}.",
#         )
#         parser.add_argument(
#             "--runtime-env-hash",
#             required=False,
#             type=int,
#             default=0,
#             help="The computed hash of the runtime env for this worker.",
#         )
#         parser.add_argument(
#             "--startup-token",
#             required=True,
#             type=int,
#             help="The startup token assigned to this worker process by the raylet.",
#         )
#         parser.add_argument(
#             "--ray-debugger-external",
#             default=False,
#             action="store_true",
#             help="True if Ray debugger is made available externally.",
#         )


# from jsonargparse import ArgumentParser
# parser = ArgumentParser(
#     description=("Parse addresses for the worker to connect to.")
# )
# parser.add_argument("--test-argument", required=True, type=int, help="Required Test Argument for testing with ray")
# args = parser.parse_args()
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Type, Union
from pytorch_lightning import Callback, LightningDataModule, LightningModule, seed_everything, Trainer
from pytorch_lightning.utilities.cli import SaveConfigCallback, ReduceLROnPlateau
from pytorch_lightning.utilities.meta import get_all_subclasses
import inspect
from torch.optim import Optimizer

class _Registry(dict):
    def __call__(self, cls: Type, key: Optional[str] = None, override: bool = False) -> Type:
        """Registers a class mapped to a name.

        Args:
            cls: the class to be mapped.
            key: the name that identifies the provided class.
            override: Whether to override an existing key.
        """
        if key is None:
            key = cls.__name__
        elif not isinstance(key, str):
            raise TypeError(f"`key` must be a str, found {key}")

        if key not in self or override:
            self[key] = cls
        return cls

    def register_classes(self, module: ModuleType, base_cls: Type, override: bool = False) -> None:
        """This function is an utility to register all classes from a module."""
        for cls in self.get_members(module, base_cls):
            self(cls=cls, override=override)

    @staticmethod
    def get_members(module: ModuleType, base_cls: Type) -> Generator[Type, None, None]:
        return (
            cls
            for _, cls in inspect.getmembers(module, predicate=inspect.isclass)
            if issubclass(cls, base_cls) and cls != base_cls
        )

    @property
    def names(self) -> List[str]:
        """Returns the registered names."""
        return list(self.keys())

    @property
    def classes(self) -> Tuple[Type, ...]:
        """Returns the registered classes."""
        return tuple(self.values())

    def __str__(self) -> str:
        return f"Registered objects: {self.names}"


OPTIMIZER_REGISTRY = _Registry()
LR_SCHEDULER_REGISTRY = _Registry()
CALLBACK_REGISTRY = _Registry()
MODEL_REGISTRY = _Registry()
DATAMODULE_REGISTRY = _Registry()
LOGGER_REGISTRY = _Registry()

class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer: Optimizer, monitor: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(optimizer, *args, **kwargs)
        self.monitor = monitor


def _populate_registries(subclasses: bool) -> None:
    if subclasses:
        # this will register any subclasses from all loaded modules including userland
        for cls in get_all_subclasses(torch.optim.Optimizer):
            OPTIMIZER_REGISTRY(cls)
        for cls in get_all_subclasses(torch.optim.lr_scheduler._LRScheduler):
            LR_SCHEDULER_REGISTRY(cls)
        for cls in get_all_subclasses(pl.Callback):
            CALLBACK_REGISTRY(cls)
        for cls in get_all_subclasses(pl.LightningModule):
            MODEL_REGISTRY(cls)
        for cls in get_all_subclasses(pl.LightningDataModule):
            DATAMODULE_REGISTRY(cls)
        for cls in get_all_subclasses(pl.loggers.LightningLoggerBase):
            LOGGER_REGISTRY(cls)
    else:
        # manually register torch's subclasses and our subclasses
        OPTIMIZER_REGISTRY.register_classes(torch.optim, Optimizer)
        LR_SCHEDULER_REGISTRY.register_classes(torch.optim.lr_scheduler, torch.optim.lr_scheduler._LRScheduler)
        CALLBACK_REGISTRY.register_classes(pl.callbacks, pl.Callback)
        LOGGER_REGISTRY.register_classes(pl.loggers, pl.loggers.LightningLoggerBase)
    # `ReduceLROnPlateau` does not subclass `_LRScheduler`
    LR_SCHEDULER_REGISTRY(cls=ReduceLROnPlateau)



class XaminCLI(LightningCLI):

    """Implementation of a configurable command line tool for pytorch-lightning."""

    def __init__(
        self,
        model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_filename: str = "config.yaml",
        save_config_overwrite: bool = False,
        save_config_multifile: bool = False,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Optional[int] = None,
        description: str = "pytorch-lightning trainer command line tool",
        env_prefix: str = "PL",
        env_parse: bool = False,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        run: bool = True,
        auto_registry: bool = False,
    ) -> None:
        """Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which
        are called / instantiated using a parsed configuration file and / or command line args.

        Parsing of configuration from environment variables can be enabled by setting ``env_parse=True``.
        A full configuration yaml would be parsed from ``PL_CONFIG`` if set.
        Individual settings are so parsed from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        For more info, read :ref:`the CLI docs <common/lightning_cli:LightningCLI>`.

        .. warning:: ``LightningCLI`` is in beta and subject to change.

        Args:
            model_class: An optional :class:`~pytorch_lightning.core.lightning.LightningModule` class to train on or a
                callable which returns a :class:`~pytorch_lightning.core.lightning.LightningModule` instance when
                called. If ``None``, you can pass a registered model with ``--model=MyModel``.
            datamodule_class: An optional :class:`~pytorch_lightning.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` instance when
                called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
            save_config_callback: A callback class to save the training config.
            save_config_filename: Filename for the config file.
            save_config_overwrite: Whether to overwrite an existing config file.
            save_config_multifile: When input is multiple config files, saved config preserves this structure.
            trainer_class: An optional subclass of the :class:`~pytorch_lightning.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~pytorch_lightning.trainer.trainer.Trainer` instance when called.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks. The callbacks added through
                this argument will not be configurable from a configuration file and will always be present for
                this particular CLI. Alternatively, configurable callbacks can be added as explained in
                :ref:`the CLI docs <common/lightning_cli:Configurable callbacks>`.
            seed_everything_default: Default value for the :func:`~pytorch_lightning.utilities.seed.seed_everything`
                seed argument.
            description: Description of the tool shown when running ``--help``.
            env_prefix: Prefix for environment variables.
            env_parse: Whether environment variable parsing is enabled.
            parser_kwargs: Additional arguments to instantiate each ``LightningArgumentParser``.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            run: Whether subcommands should be added to run a :class:`~pytorch_lightning.trainer.trainer.Trainer`
                method. If set to ``False``, the trainer and model classes will be instantiated only.
            auto_registry: Whether to automatically fill up the registries with all defined subclasses.
        """
        self.save_config_callback = save_config_callback
        self.save_config_filename = save_config_filename
        self.save_config_overwrite = save_config_overwrite
        self.save_config_multifile = save_config_multifile
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        self.seed_everything_default = seed_everything_default

        self.model_class = model_class
        # used to differentiate between the original value and the processed value
        self._model_class = model_class or LightningModule
        self.subclass_mode_model = (model_class is None) or subclass_mode_model

        self.datamodule_class = datamodule_class
        # used to differentiate between the original value and the processed value
        self._datamodule_class = datamodule_class or LightningDataModule
        self.subclass_mode_data = (datamodule_class is None) or subclass_mode_data

        _populate_registries(auto_registry)

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(
            parser_kwargs or {},  # type: ignore  # github.com/python/mypy/issues/6463
            {"description": description, "env_prefix": env_prefix, "default_env": env_parse},
        )
        self.setup_parser(run, main_kwargs, subparser_kwargs)
        self.parse_arguments(self.parser)

        self.subcommand = self.config["subcommand"] if run else None

        seed = self._get(self.config, "seed_everything")
        if seed is not None:
            seed_everything(seed, workers=True)

    def delayed_init(self):
        self.before_instantiate_classes()
        self.instantiate_classes()

        if self.subcommand is not None:
            self._run_subcommand(self.subcommand)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--test-argument", required=True, type=int, help="Required Test Argument for testing with ray")

 
@ray.remote(num_gpus=1)
def train(cli) -> None:

    print(cli.model)

    # TODO: GPU not availble in head node

    # model = cli.model
    # print(model)

    # tb_logger = None  # TensorBoardLogger(save_dir='tb-logs')
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

    # cli.instantiate_classes()
    # cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    print(f"SUCCESS: {torch.cuda.is_available()}")
    return "SUCCESS"


if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser(
    # description=__doc__)

    # parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    # parser.add_argument('--dataset', default='coco', help='dataset')
    # parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    # parser.add_argument('--device', default='cuda', help='device')
    # parser.add_argument('-b', '--batch-size', default=2, type=int,
    #                     help='images per gpu, the total batch size is $NGPU x batch_size')
    # parser.add_argument('--epochs', default=26, type=int, metavar='N',
    #                     help='number of total epochs to run')
    # parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
    #                     help='number of data loading workers (default: 4)')
    # parser.add_argument('--lr', default=0.02, type=float,
    #                     help='initial learning rate, 0.02 is the default value for training '
    #                     'on 8 gpus and 2 images_per_gpu')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum')
    # parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)',
    #                     dest='weight_decay')
    # parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    # parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    # parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    # parser.add_argument('--output-dir', default='.', help='path where to save')
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # parser.add_argument(
    #     "--test-only",
    #     dest="test_only",
    #     help="Only test the model",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--pretrained",
    #     dest="pretrained",
    #     help="Use pre-trained models from the modelzoo",
    #     action="store_true",
    # )

    # # distributed training parameters
    # parser.add_argument('--world-size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # args = parser.parse_args()


    cli = XaminCLI(ImageClassifier,
                       seed_everything_default=1337,
                       save_config_overwrite=True,
                       run=False,
                       trainer_defaults={"logger": lazy_instance(TensorBoardLogger, save_dir="logs")})
    
    
    # result = train()
    
    obj_ref = train.remote(args)
    result = ray.get(obj_ref)
    print(result)
