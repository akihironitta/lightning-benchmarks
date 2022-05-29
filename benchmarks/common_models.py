import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10

from benchmarks.common_data import AverageDataset, MNIST
from pytorch_lightning import LightningModule

_PATH_DATASETS = "./data/"


class ParityModuleRNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(10, 20, batch_first=True)
        self.linear_out = nn.Linear(in_features=20, out_features=5)
        self.example_input_array = torch.rand(2, 3, 10)

    def forward(self, x):
        seq, last = self.rnn(x)
        return self.linear_out(seq)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(AverageDataset(), batch_size=30)


class ParityModuleMNIST(LightningModule):
    def __init__(self):
        super().__init__()
        self.c_d1 = nn.Linear(in_features=28 * 28, out_features=128)
        self.c_d1_bn = nn.BatchNorm1d(128)
        self.c_d1_drop = nn.Dropout(0.3)
        self.c_d2 = nn.Linear(in_features=128, out_features=10)
        self.example_input_array = torch.rand(2, 1, 28, 28)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.c_d1(x)
        x = torch.tanh(x)
        x = self.c_d1_bn(x)
        x = self.c_d1_drop(x)
        x = self.c_d2(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        return DataLoader(
            MNIST(root=_PATH_DATASETS, train=True, download=True),
            batch_size=128,
            num_workers=1,
        )


class ParityModuleCIFAR(LightningModule):
    def __init__(self, backbone="resnet101", hidden_dim=1024, learning_rate=1e-3, pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = 10
        self.backbone = getattr(models, backbone)(pretrained=pretrained)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1000, hidden_dim),
            torch.nn.Linear(hidden_dim, self.num_classes),
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        y_hat = self.classifier(y_hat)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            CIFAR10(root=_PATH_DATASETS, train=True, download=True, transform=self.transform),
            batch_size=32,
            num_workers=1,
        )


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)
