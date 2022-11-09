# text-classification

## Pretrained Models

you can download the pretrained models from hugggingface.co manually or use transformers to download automatically.

```
https://huggingface.co/bert-base-cased/tree/main
```

## dataset

download the dataset and put in the `data` directory.

### IMDB

```
cd data
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvf aclImdb_v1.tar.gz
```

## tensorboard

```
tensorboard --host 0.0.0.0 --port 24321 --logdir ./lightning_logs/ serve
```
