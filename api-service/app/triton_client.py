import numpy as np
import tritonclient.http as httpclient

TRITON_URL = "triton:8000"

client = httpclient.InferenceServerClient(url=TRITON_URL)

def infer():

    input_data = np.random.rand(1,10).astype(np.float32)

    inputs = httpclient.InferInput(
        "input",
        input_data.shape,
        "FP32"
    )

    inputs.set_data_from_numpy(input_data)

    outputs = httpclient.InferRequestedOutput("output")

    result = client.infer(
        model_name="toy_model",
        inputs=[inputs],
        outputs=[outputs]
    )

    return result.as_numpy("output")