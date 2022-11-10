import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf

from model import TextCNN, TextRNN, TextRCNN
from dataset import TextClassificationDataModule, ImdbDataset
from database import MLDatabase


def parse_args(): 
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--cfg', type=str, default='./cfg/text_rcnn.yml')
    args.add_argument('-m', '--mode', type=str, default='train')
    return args.parse_args()


def get_model(cfg):
    if cfg.name.startswith('text_cnn'):
        return TextCNN(cfg)
    elif cfg.name.startswith('text_rnn'):
        return TextRNN(cfg)
    elif cfg.name.startswith('text_rcnn'):
        return TextRCNN(cfg)
    else:
        raise NotImplemented


def write_sqlite(cfg, hypermeters, metrics, best_accuracy):
    database = MLDatabase(cfg.sqlite_path)
    database.insert_experiment_result({
        'model': cfg.name,
        'dataset': cfg.dataset.name,
        'hypermeters': hypermeters,
        'metrics': metrics,
        'accuracy': best_accuracy
    })
    database.close()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    # build vocab
    imdb_dataset = ImdbDataset(cfg, 'train')
    cfg.vocab = imdb_dataset.vocab
    cfg.vocab_size = len(cfg.vocab)

    # data
    data_module = TextClassificationDataModule(cfg)

    # model
    model = get_model(cfg)

    # training
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.accelerator_devices,
        logger=pl.loggers.TensorBoardLogger(cfg.log_path, cfg.name),
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.train.epochs
    )
    trainer.fit(model, data_module)

    write_sqlite(cfg, model.hypermeters(), model.metrics(), model.best_accuracy)


if __name__ == '__main__':
    main()
