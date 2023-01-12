if __name__ == "__main__":

    import timm 

    from prospr import prune
    from datagen import load_cifar10

    # SET CONFIG MANUALLY HERE
    model = "resnet18"
    data = "cifar10"
    num_classes = 10
    batch_size = 128
    prune_ratio = 0.9
    num_steps = 4
    inner_lr = 0.03
    inner_momentum = 0.9

    train_data, test_data = load_cifar10(batch_size = 128)

    model = timm.create_model(model, pretrained = False, in_chans = 3, num_classes = num_classes)

    if prune_ratio > 0: 
        masks = prune(model,
                    prune_ratio,
                    train_data,
                    num_steps,
                    inner_lr,
                    inner_momentum)