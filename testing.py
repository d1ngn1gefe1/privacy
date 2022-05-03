"""
Optuna example that optimizes multi-layer perceptrons using PyTorch Lightning's
distributed data-parallel training.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch Lightning, and FashionMNIST. We optimize the neural network architecture. As it is too time
consuming to use the whole FashionMNIST dataset, we here use a small subset of it.

You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python pytorch/pytorch_lightning_ddp.py [--pruning]

"""
import argparse
import os
from typing import List
from typing import Optional

import optuna
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from pytorch_lightning.strategies import DDPStrategy
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
from pytorch_lightning import Trainer


PERCENT_VALID_EXAMPLES = 0.2
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10
DIR = os.getcwd()


class Net(nn.Module):
    def __init__(self, dropout: float, output_dims: List[int]):
        super().__init__()
        layers: List[nn.Module] = []

        input_dim: int = 28 * 28
        for output_dim in output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, CLASSES))

        self.layers: nn.Module = nn.Sequential(*layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.layers(data)
        return F.log_softmax(logits, dim=1)


class LightningNet(pl.LightningModule):
    def __init__(self, dropout: float, output_dims: List[int]):
        super().__init__()
        self.model = Net(dropout, output_dims)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data.view(-1, 28 * 28))

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        data, target = batch
        output = self(data)
        return F.nll_loss(output, target)

    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy, sync_dist=True)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.model.parameters())


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.mnist_test = datasets.FashionMNIST(
            self.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        mnist_full = datasets.FashionMNIST(
            self.data_dir, train=True, download=True, transform=transforms.ToTensor()
        )
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, shuffle=True, pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )


def objective(trial: optuna.trial.Trial) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
    ]

    model = LightningNet(dropout, output_dims)
    datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)

    trainer = pl.Trainer(
        logger=True,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=EPOCHS,
        # accelerator='gpu',
        # devices=[1, 2],
        # strategy=DDPStrategy(find_unused_parameters=False),
        gpus=-1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )
    hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_acc"].item()


def on_init_start(self, trainer: Trainer) -> None:
    self.is_ddp_backend = trainer._accelerator_connector.is_distributed
    if self.is_ddp_backend:
        if version.parse(pl.__version__) < version.parse('1.5.0'):
            raise ValueError('PyTorch Lightning>=1.5.0 is required in DDP.')
        if not (
            isinstance(self._trial.study._storage, _CachedStorage)
            and isinstance(self._trial.study._storage._backend, RDBStorage)
        ):
            breakpoint()
            raise ValueError(
                'optuna.integration.PyTorchLightningPruningCallback'
                ' supports only optuna.storages.RDBStorage in DDP.'
            )


if __name__ == "__main__":
    PyTorchLightningPruningCallback.on_init_start = on_init_start

    parser = argparse.ArgumentParser(
        description="PyTorch Lightning distributed data-parallel training example."
    )
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    storage = "sqlite:///example.db"
    study = optuna.create_study(
        study_name="pl_ddp",
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))