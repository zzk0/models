import torch
from torch import nn
from torch.nn import functional as F

from .model import TextClassificationModel, Embedding


class TorchBertCls(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = Embedding(cfg)
        self.dropout = torch.nn.Dropout(p=self.cfg.model.dropout, inplace=False)
        self.num_features = self.embedding.embedding_size
        self.classifier = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.GELU(),
            nn.Linear(self.num_features, self.cfg.model.num_classes)
        ])

    def forward(self, x):
        out = self.embedding(x)
        last_hidden = out['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        logits = self.classifier(pooled_output)
        return logits


class BertCls(TextClassificationModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = TorchBertCls(self.cfg)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr, 
                                      weight_decay=self.cfg.optimizer.weight_decay)
        return optimizer

    def hypermeters(self):
        parameters = ''
        parameters += 'embedding_name={}, '.format(self.cfg.embedding.name)
        parameters += 'dropout={}, '.format(self.cfg.model.dropout)
        return parameters
