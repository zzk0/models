import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from transformers import AutoModel


class TextCNN(pl.LightningModule):
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if self.cfg.embedding.type in ('bert'):
			if self.cfg.embedding.use_local:
				self.embedding = AutoModel.from_pretrained(self.cfg.embedding.path, local_files_only=True)
			else:
				self.embedding = AutoModel.from_pretrained(self.cfg.embedding.name)
		else:
			self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.embedding.dimension, padding_idx=self.cfg.vocab_size-1)
		self.convs = nn.ModuleList(
			[nn.Conv2d(1, self.cfg.model.num_filters, (k, self.cfg.embedding.dimension)) for k in self.cfg.model.filter_sizes]
		)
		self.dropout = nn.Dropout(self.cfg.model.dropout)
		self.fc = nn.Linear(self.cfg.model.num_filters * len(self.cfg.model.filter_sizes), self.cfg.model.num_classes)
		self.valid_accuracy = torchmetrics.Accuracy()

		self.best_accuracy = 0.0

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

	def metrics(self):
		indicators = ''
		indicators += 'acc={}'.format(self.best_accuracy)
		return indicators

	def uniform_log(self, *args):
		if self.cfg.train.accelerator_devices > 1:
			self.log(*args, sync_dist=True)
		else:
			self.log(*args)

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

	def validation_step(self, val_batch, batch_idx):
		x = val_batch['input_ids']
		y = val_batch['labels']
		out = self.forward(x)
		loss = F.cross_entropy(out, y)
		out = torch.argmax(out, 1)
		self.valid_accuracy.update(out, y)
		self.uniform_log('val_loss', loss)

	def validation_epoch_end(self, validation_step_outputs):
		self.best_accuracy = max(self.best_accuracy, self.valid_accuracy.compute())
		self.uniform_log('val_acc', self.valid_accuracy.compute())
		self.valid_accuracy.reset()
