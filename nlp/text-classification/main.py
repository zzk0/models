import os
import argparse
import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from model import TextCNN, TextRNN, TextRCNN, SpanEmo, BertCls
from dataset import TextClassificationDataModule, ImdbDataset, SemEval18Dataset, ImdbSpanEmoDataset
from database import MLDatabase


def parse_args(): 
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--cfg', type=str, default='./cfg/span_emo_imdb.yml')
    args.add_argument('-s', '--seed', type=int, default=42)
    args.add_argument('-m', '--mode', type=str, default='train')
    return args.parse_args()


def get_model(cfg):
    if cfg.name.startswith('text_cnn'):
        return TextCNN(cfg)
    elif cfg.name.startswith('text_rnn'):
        return TextRNN(cfg)
    elif cfg.name.startswith('text_rcnn'):
        return TextRCNN(cfg)
    elif cfg.name.startswith('span_emo'):
        return SpanEmo(cfg)
    elif cfg.name.startswith('bert_cls'):
        return BertCls(cfg)
    else:
        raise NotImplemented


def write_sqlite(cfg, hypermeters, metrics):
    database = MLDatabase(cfg.sqlite_path)
    database.insert_experiment_result({
        'model': cfg.name,
        'dataset': cfg.dataset.name + cfg.dataset.get('lang', ''),
        'hypermeters': hypermeters,
        'metrics': metrics,
        'accuracy': metrics['accuracy'] if 'accuracy' in metrics else 0.0,
        'f1_micro': metrics['test_f1_micro'] if 'test_f1_micro' in metrics else 0.0,
        'f1_macro': metrics['test_f1_macro'] if 'test_f1_macro' in metrics else 0.0,
        'jaccard': metrics['test_jaccard'] if 'test_jaccard' in metrics else 0.0
    })
    database.close()


def export_onnx(cfg, model: pl.LightningModule):
    input_sample = [i for i in range(3000, 3000 + cfg.max_seq_len - 2)]
    input_sample = [101, *input_sample, 102]
    input_sample = torch.Tensor([input_sample, input_sample]).long()
    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    model.to_onnx(os.path.join(cfg['onnx_path'], cfg.name + ".onnx"), input_sample, export_params=True, opset_version=11,
                  input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)


def test_pytorch_inference(cfg, model):
    input_sample = [i for i in range(3000, 3000 + cfg.max_seq_len - 2)]
    input_sample = [101, *input_sample, 102]
    input_sample = torch.Tensor([input_sample]).long()
    model = model.model
    model.eval()
    with torch.no_grad():
        y = model(input_sample)
        t0 = time.time()
        y = model(input_sample)
        t1 = time.time()
        print('time cost: ', t1 - t0, y)


def test_span_emo(cfg):
    victim = SpanEmo(cfg)
    model_path = '/home/percent1/models/nlp/text-classification/checkpoints/span_emo_english/epoch=3-step=428.ckpt'
    victim = victim.load_from_checkpoint(checkpoint_path=model_path, cfg=cfg)
    dataset = SemEval18Dataset(cfg, 'train')

    def __victim_forward(input_):
        batch = {
            'input_ids': [],
            'length': [],
            'label_idxs': []
        }
        for text in input_:
            input_id, input_length, label_idxs = dataset.process_input(text, None)
            batch['input_ids'].append(input_id)
            batch['length'].append(input_length)
            batch['label_idxs'].append(label_idxs)
        batch['input_ids'] = torch.Tensor(batch['input_ids']).long()
        batch['length'] = torch.Tensor(batch['length']).long()
        batch['label_idxs'] = torch.Tensor(batch['label_idxs']).long()
        _, _, logits, y_pred, _ = victim(batch)
        return logits, y_pred

    input = ["Modern family never fails to cheer me up . Especially Phil .",
             "Modern family never seems to cheer me up . Especially Phil ."]

    print(__victim_forward(input))


def load_cfg(path):
    cfg = OmegaConf.load(path)

    # set default value
    cfg.train.accumulate_grad = cfg.get('train.accumulate_grad', 1)

    return cfg


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    # test_span_emo(cfg)

    pl.seed_everything(args.seed, workers=True)
    if cfg.embedding.type not in ('bert'):
        # build vocab
        imdb_dataset = ImdbDataset(cfg, 'train')
        cfg.vocab = imdb_dataset.vocab
        cfg.vocab_size = len(cfg.vocab)

    if cfg.name.startswith('span_emo') and cfg.dataset.name == 'semeval18':
        semeval18_dataset = SemEval18Dataset(cfg, 'train')
        cfg.dataset.size = len(semeval18_dataset)
    if cfg.name.startswith('span_emo') and cfg.dataset.name == 'imdb_span_emo':
        # span_emo_imdb_dataset = ImdbSpanEmoDataset(cfg, 'train')
        cfg.dataset.size = 25000

    # data
    data_module = TextClassificationDataModule(cfg)

    # model
    model = get_model(cfg)

    # training
    if cfg.name.startswith('span_emo'):
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join("./checkpoints", cfg.name), save_top_k=1, monitor="overall_val_loss")
        trainer = pl.Trainer(
            precision=cfg.train.precision,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.accelerator_devices,
            logger=pl.loggers.TensorBoardLogger(cfg.log_path, cfg.name),
            log_every_n_steps=cfg.log_every_n_steps,
            max_epochs=cfg.train.epochs,
            deterministic=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
                checkpoint_callback
            ]
        )
        trainer.fit(model, data_module)
        model = model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path, cfg=cfg)
        trainer.test(model, data_module)
    else:
        trainer = pl.Trainer(
            precision=cfg.train.precision,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.accelerator_devices,
            logger=pl.loggers.TensorBoardLogger(cfg.log_path, cfg.name),
            log_every_n_steps=cfg.log_every_n_steps,
            max_epochs=cfg.train.epochs,
            deterministic=True,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ]
        )
        trainer.fit(model, data_module)
    write_sqlite(cfg, model.hypermeters() + ', seed=' + str(args.seed), trainer.callback_metrics)
    if 'onnx_path' in cfg:
        test_pytorch_inference(cfg, model)
        export_onnx(cfg, model)


if __name__ == '__main__':
    main()
