# text-classification

## TODO

1. Add Job Scheduler to help run jobs

## Conda 

You can create conda environment to reproduce result.

```
conda create -n lightning python=3.10
conda activate ligthning
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 main.py -c ./cfg/span_emo.yml
```

## Docker

You can install the requirements.txt. If you want to reproduce the result, we recommend use docker.

```
# install docker
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun

# find out the lastest cuda version the driver supported
nvidia-smi

# find the correspond docker image here: https://hub.docker.com/r/pytorch/pytorch
docker pull pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
docker run -it --runtime=nvidia --network=host -v $(pwd):$(pwd) -w $(pwd) --shm-size=8g pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime bash

# inside docker container 
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 main.py -c ./cfg/text_cnn.yml
```

## dataset

download the dataset and put in the `data` directory.

### IMDB

```
cd data
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvf aclImdb_v1.tar.gz
```

the downloaded dataset should be put in the directory called `data`

```
data
├── aclImdb
│   ├── imdbEr.txt
│   ├── imdb.vocab
│   ├── README
│   ├── test
│   └── train
└── aclImdb_v1.tar.gz
```

## Pretrained Models

you can download the pretrained models from hugggingface.co manually or use transformers to download automatically by turning off `use_local` in .yml configuration files.

```
bert: https://huggingface.co/bert-base-cased/tree/main
roberta: https://huggingface.co/roberta-base/tree/main
deberta: https://huggingface.co/microsoft/deberta-v3-base/tree/main
```

the pretrained models should be put in the directory called `pretrained`

```
$ tree pretrained/
pretrained/
├── bert-base-cased
│   ├── config.json
│   ├── gitattributes.txt
│   ├── pytorch_model.bin
│   ├── README.md
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── deberta-v3-base
│   ├── config.json
│   ├── gitattributes.txt
│   ├── pytorch_model.bin
│   ├── README.md
│   ├── spm.model
│   └── tokenizer_config.json
└── roberta-base
    ├── config.json
    ├── dict.txt
    ├── gitattributes.txt
    ├── merges.txt
    ├── pytorch_model.bin
    ├── README.md
    ├── tokenizer.json
    └── vocab.json
```


## tensorboard

```
tensorboard --host 0.0.0.0 --port 24321 --logdir ./logs/ serve
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

