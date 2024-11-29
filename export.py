import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.conv_module(1, 32)  # 32
        self.conv2 = self.conv_module(32, 64)  # 16
        self.conv3 = self.conv_module(64, 128)  # 8

        self.classifier = self.classification_module(
            8 * 8 * 128, 10 * 2
        )  # Multi-label

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        y = self.classifier(x)

        return y

    @staticmethod
    def conv_module(in_channels, out_channels):
        module = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        return module

    @staticmethod
    def classification_module(in_features, num_class):
        classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_class),
            nn.Softmax(dim=1),
        )

        return classifier


model = ClassificationNet()
model.load_state_dict(torch.load("best.pt"))
model = model.to(device)

dummy_input = torch.randn(1, 1, 64, 64).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "best.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
