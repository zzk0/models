from torch import nn

from .model import TextClassificationModel, Embedding


class TorchTextRNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = Embedding(cfg)
        self.lstm = nn.LSTM(self.embedding.embedding_size, self.cfg.model.hidden_size, self.cfg.model.num_layers,
                            bidirectional=self.cfg.model.bidirectional, batch_first=True, dropout=self.cfg.model.dropout)
        self.fc = nn.Linear(self.cfg.model.hidden_size * 2, self.cfg.model.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        if self.cfg.embedding.type in ('bert'):
            out = out[0]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out


class TextRNN(TextClassificationModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = TorchTextRNN(self.cfg)

    def hypermeters(self):
        parameters = ''
        if self.cfg.embedding.type in ('bert'):
            parameters += 'embedding_model={}, '.format(self.cfg.embedding.name)
        else:
            parameters += 'embedding_dimension={}, '.format(self.cfg.embedding.dimension)
        parameters += 'hidden_size={}, '.format(self.cfg.model.hidden_size)
        parameters += 'num_layers={}, '.format(self.cfg.model.num_layers)
        parameters += 'bidirectional={}, '.format(self.cfg.model.bidirectional)
        parameters += 'dropout={}, '.format(self.cfg.model.dropout)
        parameters += 'num_classes={}'.format(self.cfg.model.num_classes)
        parameters += 'precision={} '.format(self.cfg.train.precision)
        parameters += 'ddp={} '.format(self.cfg.train.accelerator_devices > 1)
        return parameters
