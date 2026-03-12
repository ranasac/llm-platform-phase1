FROM nvcr.io/nvidia/tritonserver:25.12-py3

COPY model_repository /models

CMD ["tritonserver", "--model-repository=/models"]