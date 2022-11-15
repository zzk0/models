import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import AutoModel


class TextClassificationModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.valid_accuracy = torchmetrics.Accuracy()

    def hypermeters(self):
        parameters = ''
        return parameters

    def uniform_log(self, *args):
        if self.cfg.train.accelerator_devices > 1:
            self.log(*args, sync_dist=True)
        else:
            self.log(*args)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x = train_batch['input_ids']
        y = train_batch['labels']
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        self.uniform_log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        self.valid_accuracy.reset()

    def validtaion_and_test_commom_step(self, stage: str, batch, batch_idx):
        x = batch['input_ids']
        y = batch['labels']
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        out = torch.argmax(out, 1)
        self.valid_accuracy.update(out, y)
        self.uniform_log(stage + '_loss', loss)

    def validtaion_and_test_commom_end(self, stage: str, outputs):
        accuracy = self.valid_accuracy.compute().item()
        self.uniform_log(stage + '_acc', accuracy)
        self.valid_accuracy.reset()

    def validation_step(self, val_batch, batch_idx):
        self.validtaion_and_test_commom_step('val', val_batch, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        self.validtaion_and_test_commom_end(validation_step_outputs)

    def test_step(self, test_batch, batch_idx):
        self.validtaion_and_test_commom_step('test', test_batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        self.validtaion_and_test_commom_end('test', test_step_outputs)


class Embedding(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_size = 768
        if self.cfg.embedding.type in ('bert'):
            if self.cfg.embedding.use_local:
                self.embedding = AutoModel.from_pretrained(self.cfg.embedding.path, local_files_only=True)
            else:
                self.embedding = AutoModel.from_pretrained(self.cfg.embedding.name)
        else:
            self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.embedding.dimension, padding_idx=self.cfg.vocab_size-1)
            self.embedding_size = self.cfg.embedding.dimension

    def forward(self, x):
        return self.embedding(x)
