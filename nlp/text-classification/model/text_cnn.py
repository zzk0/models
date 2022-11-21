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
        x = F.max_pool1d(x, int(x.size(2))).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        if self.cfg.embedding.type in ('bert'):
            out = out[0]
        out = out.unsqueeze(1)
        
        # out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        
        x0 = F.relu(self.convs[0](out)).squeeze(3)
        x0 = F.max_pool1d(x0, self.cfg.max_seq_len - 1)
        x0 = x0.view((x0.shape[0], x0.shape[1]))

        x1 = F.relu(self.convs[1](out)).squeeze(3)
        x1 = F.max_pool1d(x1, self.cfg.max_seq_len - 2)
        x1 = x1.view((x1.shape[0], x1.shape[1]))

        x2 = F.relu(self.convs[2](out)).squeeze(3)
        x2 = F.max_pool1d(x2, self.cfg.max_seq_len - 3)
        x2 = x2.view((x2.shape[0], x2.shape[1]))

        out = torch.cat([x0, x1, x2], dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class TextCNN(TextClassificationModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = TorchTextCNN(self.cfg)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr, 
                                      weight_decay=self.cfg.optimizer.weight_decay)
        return optimizer

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
        parameters += 'precision={} '.format(self.cfg.train.precision)
        parameters += 'ddp={} '.format(self.cfg.train.accelerator_devices > 1)
        return parameters
