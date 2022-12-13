import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score

from .model import Embedding


class TorchSpanEmo(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = Embedding(cfg)
        self.loss_type = self.cfg.model.loss_type
        self.alpha = self.cfg.model.alpha
        self.ffn = nn.Sequential(
            nn.Linear(self.embedding.embedding_size, self.embedding.embedding_size),
            nn.Tanh(),
            nn.Dropout(p=self.cfg.model.dropout),
            nn.Linear(self.embedding.embedding_size, 1)
        )

    def forward(self, batch):
        inputs = batch['input_ids']
        label_idxs = batch['label_idxs'][0]
        num_rows = inputs.size(0)

        out = self.embedding(inputs)
        out = out[0]
        logits = self.ffn(out).squeeze(-1).index_select(dim=1, index=label_idxs)
        y_pred = self.compute_pred(logits)

        if 'targets' not in batch:
            return None, num_rows, logits, y_pred, None

        # Loss Function
        targets = batch['targets']
        if self.loss_type == 'joint':
            cel = F.binary_cross_entropy_with_logits(logits, targets)
            cl = self.corr_loss(logits, targets)
            loss = ((1 - self.alpha) * cel) + (self.alpha * cl)
        elif self.loss_type == 'cross_entropy':
            loss = F.binary_cross_entropy_with_logits(logits, targets)
        elif self.loss_type == 'corr_loss':
            loss = self.corr_loss(logits, targets)
        return loss, num_rows, logits, y_pred, targets

    @staticmethod
    def corr_loss(y_hat, y_true, reduction='mean'):
        loss = torch.zeros(y_true.size(0))
        for idx, (y, y_h) in enumerate(zip(y_true, y_hat.sigmoid())):
            y_z, y_o = (y == 0).nonzero(), y.nonzero()
            if y_o.nelement() != 0:
                output = torch.exp(torch.sub(y_h[y_z], y_h[y_o][:, None]).squeeze(-1)).sum()
                num_comparisons = y_z.size(0) * y_o.size(0)
                loss[idx] = output.div(num_comparisons)
        return loss.mean() if reduction == 'mean' else loss.sum()

    @staticmethod
    def compute_pred(logits, threshold=0.5):
        y_pred = torch.sigmoid(logits) > threshold
        return y_pred.float()


class SpanEmo(pl.LightningModule):
    def __init__(self, cfg):
        """
        torchmetrics not support jaccard index for each instances: https://github.com/Lightning-AI/metrics/issues/452
        use sklearn.metrics instead
        """
        super().__init__()
        self.cfg = cfg
        self.model = TorchSpanEmo(self.cfg)
        self.f1_macro = torchmetrics.classification.MultilabelF1Score(self.cfg.dataset.class_nums, average='macro')
        self.f1_micro = torchmetrics.classification.MultilabelF1Score(self.cfg.dataset.class_nums, average='micro')
        self.valid_accuracy = torchmetrics.Accuracy(task='multilabel', num_labels=11)

    def hypermeters(self):
        parameters = ''
        parameters += 'loss type={}, '.format(self.model.loss_type)
        parameters += 'alpha={}, '.format(self.model.alpha)
        parameters += 'dropout={}, '.format(self.cfg.model.dropout)
        parameters += 'embedding_size={}, '.format(self.model.embedding.embedding_size)
        parameters += 'bert_lr={}, '.format(self.cfg.optimizer.bert_lr)
        parameters += 'ffn_lr={}, '.format(self.cfg.optimizer.ffn_lr)
        parameters += 'precision={}, '.format(self.cfg.train.precision)
        parameters += 'ddp={} '.format(self.cfg.train.accelerator_devices > 1)
        return parameters

    def uniform_log(self, *args):
        if self.cfg.train.accelerator_devices > 1:
            self.log(*args, sync_dist=True)
        else:
            self.log(*args)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = AdamW([
            {'params': self.model.embedding.parameters()}, 
            {'params': self.model.ffn.parameters(), 'lr': self.cfg.optimizer.ffn_lr}
        ], lr=self.cfg.optimizer.bert_lr, correct_bias=True)
        num_train_steps = self.cfg.dataset.size / self.cfg.train.train_batch_size * self.cfg.train.epochs
        num_warmup_steps = int(num_train_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            },
        }

    def training_step(self, train_batch, batch_idx):
        loss, _, _, _, _ = self.forward(train_batch)
        self.uniform_log('train_loss', loss)
        return loss

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=1.0,
            gradient_clip_algorithm='norm'
        )

    def validation_step(self, val_batch, batch_idx):
        return self.common_validation_test_step(val_batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        return self.common_validation_test_epoch_end(validation_step_outputs, 'val')

    def test_step(self, test_batch, batch_idx):
        return self.common_validation_test_step(test_batch, batch_idx, 'test')

    def test_epoch_end(self, test_step_outputs):
        return self.common_validation_test_epoch_end(test_step_outputs, 'test')

    def common_validation_test_step(self, batch, batch_idx, stage: str):
        loss, num_rows, logits, y_pred, targets = self.forward(batch)
        self.f1_macro.update(y_pred, targets)
        self.f1_micro.update(y_pred, targets)
        if 'labels' in batch:
            self.valid_accuracy.update(torch.softmax(logits, dim=-1), batch['labels'])
        self.uniform_log(stage + '_loss', loss)
        return {
            'loss': loss.item() * num_rows,
            'y_pred': y_pred,
            'targets': targets.float()
        }

    def common_validation_test_epoch_end(self, step_outputs, stage: str):
        y_pred_across_devices = []
        targets_across_devices = []
        overall_val_loss = 0
        for output in step_outputs:
            y_pred_across_devices.append(output['y_pred'])
            targets_across_devices.append(output['targets'])
            overall_val_loss += output['loss']
        y_pred = torch.cat(y_pred_across_devices, dim=0).cpu().numpy()
        targets = torch.cat(targets_across_devices, dim=0).cpu().numpy()
        self.uniform_log(stage + '_f1_macro', self.f1_macro.compute().item())
        self.uniform_log(stage + '_f1_micro', self.f1_micro.compute().item())
        self.uniform_log(stage + '_accuracy', self.valid_accuracy.compute().item())
        self.uniform_log(stage + '_jaccard', jaccard_score(targets, y_pred, average='samples'))
        self.uniform_log('overall_val_loss', overall_val_loss)
        self.f1_macro.reset()
        self.f1_micro.reset()
        self.valid_accuracy.reset()
