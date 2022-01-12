import math
import os

import pytorch_lightning as pl
import torch
from filelock import FileLock
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import PopulationBasedTraining
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class LightningMNISTClassifier(pl.LightningModule):
    """
    This has been adapted from
    https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
    """

    def __init__(self, config, data_dir=None):
        super().__init__()

        self.data_dir = data_dir or os.getcwd()

        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    @staticmethod
    def download_data(data_dir):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        with FileLock(os.path.expanduser("~/.data.lock")):
            return MNIST(data_dir, train=True, download=True, transform=transform)

    def prepare_data(self):
        mnist_train = self.download_data(self.data_dir)

        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=int(self.batch_size),
            num_workers=32,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=int(self.batch_size),
            num_workers=32,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train_mnist(config):
    model = LightningMNISTClassifier(config)
    trainer = pl.Trainer(max_epochs=10, gpus=1)

    trainer.fit(model)


def train_mnist_no_tune():
    config = {"layer_1_size": 128, "layer_2_size": 256, "lr": 1e-3, "batch_size": 64}
    train_mnist(config)


# train_mnist_no_tune()


def train_mnist_tune_checkpoint(
    config, checkpoint_dir=None, num_epochs=10, num_gpus=0, data_dir="~/data"
):
    data_dir = os.path.expanduser(data_dir)
    kwargs = {
        "max_epochs": num_epochs,
        # If fractional GPUs passed in, convert to int.
        "gpus": math.ceil(num_gpus),
        "logger": TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        "enable_progress_bar": False,
        "callbacks": [
            TuneReportCheckpointCallback(
                metrics={"loss": "ptl/val_loss", "mean_accuracy": "ptl/val_accuracy"},
                filename="checkpoint",
                on="validation_end",
            )
        ],
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(checkpoint_dir, "checkpoint")

    model = LightningMNISTClassifier(config=config, data_dir=data_dir)
    trainer = pl.Trainer(**kwargs)

    trainer.fit(model)


def tune_mnist_asha(num_samples=10, num_epochs=10, gpus_per_trial=1, data_dir="~/data"):
    config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"],
    )

    train_fn_with_parameters = tune.with_parameters(
        train_mnist_tune_checkpoint,
        num_epochs=num_epochs,
        num_gpus=gpus_per_trial,
        data_dir=data_dir,
    )
    resources_per_trial = {"cpu": 16, "gpu": gpus_per_trial}

    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha",
    )

    print("Best hyperparameters found were: ", analysis.best_config)


tune_mnist_asha()


def tune_mnist_pbt(num_samples=10, num_epochs=10, gpus_per_trial=1, data_dir="~/data"):
    config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": 1e-3,
        "batch_size": 64,
    }

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={"lr": tune.loguniform(1e-4, 1e-1), "batch_size": [32, 64, 128]},
    )

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"],
    )

    analysis = tune.run(
        tune.with_parameters(
            train_mnist_tune_checkpoint,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
            data_dir=data_dir,
        ),
        resources_per_trial={"cpu": 16, "gpu": gpus_per_trial},
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_pbt",
    )

    print("Best hyperparameters found were: ", analysis.best_config)


# tune_mnist_pbt()
