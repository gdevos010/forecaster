import argparse

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim
import torchmetrics
from ray import tune

loss_choices = ["mse", "cross_entropy", "poisson", "l1", "smape"]
lr_scheduler_choices = ["exponential", "cosine", "plateau", "cyclic", "cosine_warm", "two_step_exp"]


class InformerForecastTask(pl.LightningModule):
    def __init__(
        self,
        model,
        seq_len,
        label_len,
        pred_len,
        variate,
        padding=0,
        loss="mse",
        learning_rate=0.0001,
        lr_scheduler="exponential",
        inverse_scaling=False,
        scaler=None,
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters()
        metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.MeanSquaredError(),
                torchmetrics.MeanAbsoluteError(),
                torchmetrics.MeanAbsolutePercentageError(),
                torchmetrics.SymmetricMeanAbsolutePercentageError(),
            ]
        )

        self.val_metrics = metrics.clone(prefix="Val_")
        self.test_metrics = metrics.clone(prefix="Test_")
        self.scaler = scaler

    def forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        if self.hparams.padding == 0:
            decoder_input = torch.zeros(
                (batch_y.size(0), self.hparams.pred_len, batch_y.size(-1))
            ).type_as(batch_y)
        else:  # self.hparams.padding == 1
            decoder_input = torch.ones(
                (batch_y.size(0), self.hparams.pred_len, batch_y.size(-1))
            ).type_as(batch_y)
        decoder_input = torch.cat([batch_y[:, : self.hparams.label_len, :], decoder_input], dim=1)
        outputs = self.model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
        if self.hparams.output_attention:
            outputs = outputs[0]
        return outputs

    def shared_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        outputs = self(batch_x, batch_y, batch_x_mark, batch_y_mark)
        f_dim = -1 if self.hparams.variate == "mu" else 0
        batch_y = batch_y[:, -self.model.pred_len :, f_dim:]
        return outputs, batch_y

    def training_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        loss = self.loss(outputs, batch_y)
        self.log("Train_Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        loss = self.loss(outputs, batch_y)
        self.log("Val_Loss", loss)
        if self.hparams.inverse_scaling and self.scaler is not None:
            outputs = self.scaler.inverse_transform(outputs)
            batch_y = self.scaler.inverse_transform(batch_y)
        metrics = self.val_metrics(outputs, batch_y)
        self.log_dict(metrics)
        return {"Val_Loss": loss, "outputs": outputs, "targets": batch_y}

    def test_step(self, batch, batch_idx):
        outputs, batch_y = self.shared_step(batch, batch_idx)
        if self.hparams.inverse_scaling and self.scaler is not None:
            outputs = self.scaler.inverse_transform(outputs)
            batch_y = self.scaler.inverse_transform(batch_y)
        metrics = self.test_metrics(outputs, batch_y)
        self.log_dict(metrics)
        return {"outputs": outputs, "targets": batch_y}

    def on_fit_start(self):
        if self.hparams.inverse_scaling and self.scaler is not None:
            if self.scaler.device == torch.device("cpu"):
                self.scaler.to(self.device)

    def on_test_start(self):
        if self.hparams.inverse_scaling and self.scaler is not None:
            if self.scaler.device == torch.device("cpu"):
                self.scaler.to(self.device)

    def loss(self, outputs, targets, **kwargs):
        if self.hparams.loss == "mse":
            return F.mse_loss(outputs, targets)
        if self.hparams.loss == "cross_entropy":
            return F.cross_entropy(outputs, targets)
        if self.hparams.loss == "poisson":
            return F.poisson_nll_loss(outputs, targets)
        if self.hparams.loss == "l1":
            return F.l1_loss(outputs, targets)
        if self.hparams.loss == "smape":
            return 2 * (outputs - targets).abs() / (outputs.abs() + targets.abs() + 1e-8)

        raise RuntimeError("The loss function {self.hparams.loss} is not implemented.")

    def configure_optimizers(self):
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if self.hparams.lr_scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif self.hparams.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        elif self.hparams.lr_scheduler == "cosine_warm":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
        elif self.hparams.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        elif self.hparams.lr_scheduler == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=0.01, max_lr=0.01, cycle_momentum=False
            )
        elif self.hparams.lr_scheduler == "two_step_exp":

            def two_step_exp(epoch):
                if epoch % 4 == 2:
                    return 0.5
                if epoch % 4 == 0:
                    return 0.2
                return 1.0

            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                optimizer, lr_lambda=[two_step_exp]
            )
        else:
            raise RuntimeError("The scheduler {self.hparams.lr_scheduler} is not implemented.")
        return [optimizer], [scheduler]

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--model_type",
            type=str,
            default="informer",
            choices=["informer", "informer_stack"],
        )
        parser.add_argument(
            "--padding",
            type=int,
            default=0,
            choices=[0, 1],
            help="Type of padding (zero-padding or one-padding)",
        )
        parser.add_argument(
            "--learning_rate",
            "--lr",
            type=float,
            default=0.0001,
            help="Learning rate of the optimizer",
        )
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="exponential",
            choices=lr_scheduler_choices,
        )
        parser.add_argument(
            "--loss",
            type=str,
            default="mse",
            choices=loss_choices,
            help="Name of loss function",
        )
        parser.add_argument(
            "--inverse_scaling",
            "--inverse",
            action="store_true",
            help="Scale back to original values",
        )
        return parser

    @staticmethod
    def get_tuning_params():
        config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([16, 32, 64, 128]),
            "loss": tune.choice(loss_choices),
            "lr_scheduler": tune.choice(lr_scheduler_choices),
        }
        return config
