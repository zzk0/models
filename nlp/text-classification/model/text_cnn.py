import torch
from torch import nn
from torch.nn import functional as F

from .model import TextClassificationModel, Embedding


class TorchTextCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = Embedding(cfg)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.cfg.model.num_filters, (k, self.cfg.embedding.dimension)) for k in self.cfg.model.filter_sizes]
        )
        self.dropout = nn.Dropout(self.cfg.model.dropout)
        self.fc = nn.Linear(self.cfg.model.num_filters * len(self.cfg.model.filter_sizes), self.cfg.model.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        if self.cfg.embedding.type in ('bert'):
            out = out[0]
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class TextCNN(TextClassificationModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = TorchTextCNN(self.cfg)

    def hypermeters(self):
        parameters = ''
        if self.cfg.embedding.type in ('bert'):
            parameters += 'embedding_model={}, '.format(self.cfg.embedding.name)
        else:
            parameters += 'embedding_dimension={}, '.format(self.cfg.embedding.dimension)
        parameters += 'num_filters={}, '.format(self.cfg.model.num_filters)
        parameters += 'filter_sizes={}, '.format(self.cfg.model.filter_sizes)
        parameters += 'dropout={}, '.format(self.cfg.model.dropout)
        parameters += 'num_classes={}'.format(self.cfg.model.num_classes)
        return parameters
