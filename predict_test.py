import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torchmetrics.classification import MultilabelAccuracy

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
model.load_state_dict(torch.load("best.pt", weights_only=True))
model = model.to(device)

if __name__ == "__main__":
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/Y_test.npy")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32) / 255
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_loader = DataLoader(test_dataset, batch_size=128, pin_memory=True)
    metric = MultilabelAccuracy(num_labels=10 * 2).to(device)

    with torch.no_grad():

        for data, labels in val_loader:
            data = data.to(device).unsqueeze(1)
            labels = labels.to(device)

            out = model(data)

            first_digit_onehot = F.one_hot(labels[:, 0], num_classes=10)
            second_digit_onehot = F.one_hot(labels[:, 1], num_classes=10)
            one_hot_labels = torch.cat(
                (first_digit_onehot, second_digit_onehot), dim=1
            ).to(torch.float32)

            metric.update(out, one_hot_labels)

        print(f"Accuracy on all test data: {metric.compute().item()}")
