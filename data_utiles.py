import torchvision.datasets
import multi_dataloader


def read_dataset(use_valid, paths):
    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    train_trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        # 自动数据增强
        torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    test_trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    if use_valid:
        train_set = torchvision.datasets.ImageFolder(paths[0], transform=train_trans)
    else:
        train_set = torchvision.datasets.ImageFolder(paths[1], transform=train_trans)

    test_set = torchvision.datasets.ImageFolder(paths[2], transform=test_trans)

    return train_set, test_set


def get_dataset(batch_size, use_valid, paths):
    """
    :return: [train_iter valid_iter test_iter]
    """
    train_dataset, test_dataset = read_dataset(use_valid, paths)

    train_iter = multi_dataloader.MultiEpochsDataLoader(train_dataset,
                                                        batch_size,
                                                        shuffle=True,
                                                        drop_last=True,
                                                        num_workers=8,
                                                        pin_memory=True)
    # test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)
    test_iter = multi_dataloader.MultiEpochsDataLoader(test_dataset,
                                                       batch_size,
                                                       shuffle=False,
                                                       drop_last=False,
                                                       num_workers=8,
                                                       pin_memory=True)

    return train_iter, test_iter, train_dataset
