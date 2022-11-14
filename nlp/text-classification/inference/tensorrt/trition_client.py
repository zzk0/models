import time
import numpy as np
import tritonclient.http as httpclient


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000')
    
    model_config = triton_client.get_model_config('text_cnn_bert_tensorrt')
    print(model_config)

    N = 10
    input_sample = [i for i in range(3000, 3000 + 128 - 2)]
    input_sample = [101, *input_sample, 102]
    input_sample = np.array([input_sample for i in range(N)]).astype(np.int32)
    print(input_sample.shape)

    inputs = []
    inputs.append(httpclient.InferInput('input', input_sample.shape, "INT32"))
    inputs[0].set_data_from_numpy(input_sample, binary_data=False)
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output', binary_data=False))
    t0 = time.time()
    results = triton_client.infer('text_cnn_bert_tensorrt', inputs=inputs, outputs=outputs)
    output_data0 = results.as_numpy('output')
    t1 = time.time()
    print(output_data0.shape)
    print('time cost: ', t1 - t0 ,output_data0)
