from torch import nn
from torchvision.models import vgg

from models import resnet18, resnet20, resnet50


def model_factory(config: str, dataset: str, disable_auto_patch: bool) -> nn.Module:
    """Load the correct model based on the user inputs.

    Args:
        config (str): The model to use. Choose from ["resnet18", "resnet20", "resnet50", "vgg19", "vgg16"].
        dataset (str): The dataset to use. Choose from ["cifar10", "cifar100", "tiny_imagenet", "imagenet"].
        disable_auto_patch (bool): If True, disables automatic patching of ResNet and VGG models to work 
        with input sizes smaller than ImageNet
    Raises:
        ValueError: If an unknown dataset is passed to dataset.
        TypeError: If an unknown model is passed to config.

    Returns:
        nn.Module: An instance of the correct model.
    """

    dataset = dataset.lower()
    if dataset == "cifar10": # TODO: If we have this, why do we need num_classes for DataLoaders?
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset == "tiny_imagenet":
        num_classes = 200
    elif dataset == "imagenet":
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    if disable_auto_patch or (dataset == "imagenet"):
        patch_for_smaller_input = False
    else:
        patch_for_smaller_input = True

    if config == "resnet18":
        model = resnet18(
            num_classes=num_classes, patch_for_smaller_input=patch_for_smaller_input
        )
    elif config == "resnet20":
        assert dataset == "cifar10", "Model must be resnet20 when using cifar10"
        model = resnet20()
    elif config == "resnet50":
        model = resnet50(
            num_classes=num_classes, patch_for_smaller_input=patch_for_smaller_input
        )
    elif config in ["vgg16", "vgg19"]:

        # Adapted from:
        # https://github.com/alecwangcq/GraSP/blob/master/models/base/vgg.py

        # fmt: off
        vgg_cfg = {
            "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
            "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512],
        }
        # fmt: on

        model = vgg.VGG(
            vgg.make_layers(vgg_cfg[config], batch_norm=True),
            num_classes=num_classes,
        )

        if patch_for_smaller_input:
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.classifier = nn.Linear(512, num_classes)
    else:
        raise TypeError(f"Uknown model {config} passed to config")

    return model
