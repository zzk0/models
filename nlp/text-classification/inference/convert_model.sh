# onnx-simplify
onnxsim ./exported/text_cnn_bert.onnx ./exported/text_cnn_bert_simplified.onnx

# convert to tensorrt
/usr/src/tensorrt/bin/trtexec --maxBatch=128 --workspace=2048 --onnx=./exported/text_cnn_bert_simplified.onnx --saveEngine=./exported/text_cnn_bert_simplified.engine

# copy
cp ./exported/text_cnn_bert_simplified.engine ./inference/triton_models/text_cnn_bert_tensorrt/1/model.plan
