import torch
import torch.nn as nn
from pathlib import Path
import onnx
from onnx import version_converter

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = ToyModel()

dummy_input = torch.randn(1,10)

# export in eval mode
model.eval()

# Ensure destination directory exists so ONNX external data can be written
output_path = Path("model_repository/toy_model/1/model.onnx")
output_path.parent.mkdir(parents=True, exist_ok=True)

torch.onnx.export(
    model,
    dummy_input,
    str(output_path),
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("ONNX model exported")
