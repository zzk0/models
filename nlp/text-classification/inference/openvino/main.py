import numpy as np
import time
import openvino.runtime as ov

N = 1

input_sample = [i for i in range(3000, 3000 + 128 - 2)]
input_sample = [101, *input_sample, 102]
input_sample = np.array([input_sample for i in range(N)])
print(input_sample.shape)

core = ov.Core()
compiled_model = core.compile_model("/home/percent1/models/nlp/text-classification/exported/text_cnn_bert_openvino/text_cnn_bert_simplified.xml", "AUTO:CPU")
infer_request = compiled_model.create_infer_request()

t0 = time.time()
input_tensor = ov.Tensor(input_sample)
infer_request.set_input_tensor(input_tensor)
infer_request.start_async()
infer_request.wait()
output = infer_request.get_output_tensor()
output_buffer = output.data
t1 = time.time()

print('time cost: ', t1 - t0, output_buffer)


