# reference: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
import onnxruntime
import numpy as np
import time

N = 10

input_sample = [i for i in range(3000, 3000 + 128 - 2)]
input_sample = [101, *input_sample, 102]
input_sample = np.array([input_sample for i in range(N)])
print(input_sample.shape)

ort_session = onnxruntime.InferenceSession('./exported/text_cnn_bert_simplified.onnx')
ort_inputs = {'input': input_sample}

t0 = time.time()
ort_outs = ort_session.run(None, ort_inputs)
t1 = time.time()

print('time cost: ', t1 - t0, ort_outs)
