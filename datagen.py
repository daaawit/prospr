import torch

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose, Normalize

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10

def load_mnist(batch_size: int = 64, shuffle: bool = True) -> tuple[MNIST, MNIST]:
    """Load MNIST Dataset from memory or download it if it is not found

    Args:
        batch_size (int): Batch Size for DataLoader
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        tuple[MNIST, MNIST]: (train_data, test_data)
    """

    try:
        train = MNIST(
            root="data",
            train=True,
            download=False,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
            target_transform=Lambda(
                lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            ),
        )

        test = MNIST(
            root="data",
            train=False,
            download=False,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
            # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).
            #                         scatter_(0, torch.tensor(y), value=1))
        )
    except RuntimeError:
        train = MNIST(
            root="data",
            train=True,
            download=True,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
            target_transform=Lambda(
                lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            ),
        )

        test = MNIST(
            root="data",
            train=False,
            download=True,
            transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
            # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).
            #                         scatter_(0, torch.tensor(y), value=1))
        )

    train_data = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_data = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    return train_data, test_data

def load_cifar10(batch_size: int = 128, shuffle: bool = True) -> tuple[CIFAR10, CIFAR10]:
    """Load CIFAR10 Dataset from memory or download it if it is not found

    Args:
        batch_size (int): Batch Size for DataLoader
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        tuple[CIFAR10, CIFAR10]: (train_data, test_data)
    """

    try:
        train = CIFAR10(
            root="data",
            train=True,
            download=False,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),  # https://github.com/kuangliu/pytorch-cifar/issues/19
                ]
            ),
            target_transform=Lambda(
                lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            ),
        )

        test = CIFAR10(
            root="data",
            train=False,
            download=False,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),  # https://github.com/kuangliu/pytorch-cifar/issues/19
                ]
            ),
            # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        )
    except RuntimeError:
        train = CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),  # https://github.com/kuangliu/pytorch-cifar/issues/19
                ]
            ),
            target_transform=Lambda(
                lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            ),
        )

        test = CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                    ),  # https://github.com/kuangliu/pytorch-cifar/issues/19
                ]
            ),
            # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).
            #                         scatter_(0, torch.tensor(y), value=1))
        )

    train_data = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
    test_data = DataLoader(test, batch_size=batch_size, shuffle=shuffle)

    return train_data, test_data