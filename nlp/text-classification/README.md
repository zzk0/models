# text-classification

## TODO

1. Add Job Scheduler to help run jobs
2. Add more datasets and more models

## Pretrained Models

you can download the pretrained models from hugggingface.co manually or use transformers to download automatically.

```
bert: https://huggingface.co/bert-base-cased/tree/main
roberta: https://huggingface.co/roberta-base/tree/main
deberta: https://huggingface.co/microsoft/deberta-v3-base/tree/main
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

## Job Scheduler

Install Dolphin Scheduler

```
DOLPHINSCHEDULER_VERSION=3.1.0
docker run --name dolphinscheduler-standalone-server -p 12345:12345 -p 25333:25333 -d apache/dolphinscheduler-standalone-server:"${DOLPHINSCHEDULER_VERSION}"
```

Default Login Information:

```
http://localhost:12345/dolphinscheduler
username: admin 
password: dolphinscheduler123
```

## Database

We use sqlite3 to store expriment results, you can start the sqlite-web to see the result.

```
sqlite_web ./database/ml.db
```
