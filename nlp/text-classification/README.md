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

## Deployment

TODO:

1. tensorrt dynamic batch size
2. openvino can support dynamic batch size, but openvino backends not


export model to onnx and simplify it.

```
docker run --runtime=nvidia -it --network=host -v $(pwd):$(pwd) -w $(pwd) nvcr.io/nvidia/tritonserver:22.10-py3 bash
pip install onnxsim
pip install openvino-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openvino-dev[onnx] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnx onnxruntime onnxsim -i https://pypi.tuna.tsinghua.edu.cn/simple
onnxsim ./exported/text_cnn_bert.onnx ./exported/text_cnn_bert_simplified.onnx
```

### openvino

```
mo --input_model ./exported/text_cnn_bert_simplified.onnx --output_dir "./exported/text_cnn_bert_openvino/" --input_shape "(-1, 128)"
mo --input_model ./exported/text_cnn_bert.onnx --output_dir "./exported/text_cnn_bert_openvino2/" --input_shape "(-1, 128)"
```

install: https://docs.openvino.ai/latest/openvino_docs_install_guides_install_dev_tools.html

convert: https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html

dynamic_shape: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

fixed: https://zhuanlan.zhihu.com/p/443195180 ; https://github.com/Tencent/ncnn/issues/1298

```
root@fdc901676b18:/home/percent1/models/nlp/text-classification# python3 inference/openvino/main.py 
(1, 128)
[[-0.45846415  0.39795798]]

root@fdc901676b18:/home/percent1/models/nlp/text-classification# python3 inference/openvino/main.py 
(2, 128)
Traceback (most recent call last):
  File "inference/openvino/main.py", line 16, in <module>
    infer_request.wait()
RuntimeError: Caught exception: Check 'backward_compatible_check || in_out_elements_equal' failed at core/shape_inference/include/shape_nodes.hpp:93:
While validating node 'v1::Reshape /model/embedding/embedding/encoder/layer.0/attention/self/Reshape_1 (/model/embedding/embedding/encoder/layer.0/attention/self/value/Add[0]:f32{?,128,768}, /model/embedding/embedding/encoder/layer.0/attention/self/Constant_1[0]:i32{4}) -> (f32{1,128,12,64})' with friendly_name '/model/embedding/embedding/encoder/layer.0/attention/self/Reshape_1':
Requested output shape {1,128,12,64} is incompatible with input shape {2,128,768}
```

openvino backend implementation seems have problem when model inputs are dynamic shape.

```
    ov::Shape input_shape;
    RETURN_IF_OPENVINO_ASSIGN_ERROR(
        input_shape,
        model_inputs[model_inputs_name_to_index[io_name]].get_shape(),
        ("retrieving original shapes from input " + io_name).c_str());

get_shape was called on a descriptor::Tensor with dynamic shape
```


### tensorrt

trtexec is installed in triton docker images, so don't need to install tensorrt.

```
/usr/src/tensorrt/bin/trtexec --workspace=2048 --onnx=exported/text_cnn_bert_simplified.onnx --saveEngine=exported/text_cnn_bert_simplified.engine --minShapes=input:1x128 --optShapes=input:64x128 --maxShapes=input:128x128
```


### preprocess

BERT models need tokenize the sentence, there are two places we can put the tokenizer.

First, deploy the tokenizer on the server. Advantages: a unified API, client just need to post sentence. Disadvantags: increase server load. Raw data may be larger, so the data transfer would take longer. 

Second, deploy the tokenizer on the client side. Advantags: Make full use of client devices to preprocess. Disadvantags: client need know how to preprocess or server-side developers need provide different language sdk. The inference may be slower if client device has poor performance. The client device may not meet the software conditions needed by the preprocess procedure.
