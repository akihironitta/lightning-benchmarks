from torchvision.datasets import MNIST, CIFAR10


DATASET_PATH = "./data"
MNIST(root=DATASET_PATH, train=True, download=True)
MNIST(root=DATASET_PATH, train=False, download=True)
CIFAR10(root=DATASET_PATH, train=True, download=True)
CIFAR10(root=DATASET_PATH, train=False, download=True)
