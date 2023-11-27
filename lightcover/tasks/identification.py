from typing import Tuple

from omegaconf import DictConfig
from hydra.utils import instantiate
from pytorch_lightning import LightningModule

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from lightcover.datas.dataset import collate_csi_data
from lightcover.optims.lr_scheduler import NoamScheduler


class CoverSongIdentificationTask(LightningModule):
    def __init__(self, **kwargs: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.network = instantiate(self.hparams.model.network)
        self.criterion = instantiate(self.hparams.model.criterion)

    def train_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.dataset.train_ds, _recursive_=False)
        loaders = self.hparams.dataset.loaders

        train_dl = DataLoader(
            dataset=dataset,
            collate_fn=collate_csi_data,
            shuffle=True,
            **loaders,
        )

        return train_dl

    def val_dataloader(self) -> DataLoader:
        dataset = instantiate(self.hparams.dataset.val_ds, _recursive_=False)
        loaders = self.hparams.dataset.loaders

        val_dl = DataLoader(
            dataset=dataset,
            collate_fn=collate_csi_data,
            shuffle=False,
            **loaders,
        )

        return val_dl

    def training_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        features, lengths, labels = batch
        logits = self.network(features, lengths)

        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        features, lengths, labels = batch
        logits = self.network(features, lengths)

        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            **self.hparams.model.optimizer,
        )
        scheduler = NoamScheduler(
            optimizer,
            **self.hparams.model.scheduler,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def export(self, filepath: str):
        checkpoint = {
            "state_dict": {
                "network": self.network.state_dict(),
            },
            "hyper_parameters": self.hparams.model,
        }
        torch.save(checkpoint, filepath)
        print(f'Model checkpoint is saved to "{filepath}" ...')
