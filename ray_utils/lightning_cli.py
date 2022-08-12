import torch
import pytorch_lightning as pl
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Type, Union
from pytorch_lightning import Callback, LightningDataModule, LightningModule, seed_everything, Trainer
from pytorch_lightning.utilities.cli import SaveConfigCallback
from pytorch_lightning.utilities.meta import get_all_subclasses
import inspect
from torch.optim import Optimizer
from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser

'''
This file serves as a replacement/mokeypatch to the LightningCLI.
The main change, is that the LightningCLI does not call self.instantiate_classes()
during the constructor phase. 

The XaminCLI adds a init_on_worker() method to only instantiate the 
trainer, model and datamodules and optimser after the worker node has recieved 
the serialised LightningCLI. 

This is caused by Ray jobs needed to have the argparser in globals
before the job is started on the worker node. E.g. The head node may not have GPU,
or other resources available when creating the LightningCLI object.

The alternatve might be to set the gpus to 0 and then reinit the lightning classes
after.
'''

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

    def init_on_worker(self):
        self.before_instantiate_classes()
        # print('Before Classes init')
        self.instantiate_classes()
        # print('In init')
        if self.subcommand is not None:
            self._run_subcommand(self.subcommand)

    # def add_arguments_to_parser(self, parser):
    #     parser.add_argument("--test-argument", required=True, type=int, help="Required Test Argument for testing with ray")

 