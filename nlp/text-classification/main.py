import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from model import TextCNN, TextRNN, TextRCNN, SpanEmo
from dataset import TextClassificationDataModule, ImdbDataset, SemEval18Dataset
from database import MLDatabase


pl.seed_everything(42, workers=True)


def parse_args(): 
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--cfg', type=str, default='./cfg/span_emo_arabic.yml')
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


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    if cfg.embedding.type not in ('bert'):
        # build vocab
        imdb_dataset = ImdbDataset(cfg, 'train')
        cfg.vocab = imdb_dataset.vocab
        cfg.vocab_size = len(cfg.vocab)
    if cfg.name.startswith('span_emo'):
        semeval18_dataset = SemEval18Dataset(cfg, 'train')
        cfg.dataset.size = len(semeval18_dataset)

    # data
    data_module = TextClassificationDataModule(cfg)

    # model
    model = get_model(cfg)

    # training
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
            ModelCheckpoint(dirpath=os.path.join("./checkpoints", cfg.name), save_top_k=3, monitor="val_loss")
        ]
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    write_sqlite(cfg, model.hypermeters(), trainer.callback_metrics)


if __name__ == '__main__':
    main()
