import collections
import json
import math
import os
import shutil

import torch.utils.data

import torchvision.datasets


def read_params(filename):
    with open(filename, 'r') as f:
        data = f.read()
    params = json.loads(data)
    return Hps(**params)


def read_csv_labels(filename):
    with open(os.path.join('dataset', filename), 'r') as f:
        lines = f.readlines()[1:]
    lines = [line.strip().split(',') for line in lines]
    return dict((idx, label) for idx, label in lines)


def img_normalize(data):
    means = []
    stds = []
    for X, _ in data:
        means.append(X.mean(dim=(0, 2, 3)))
        stds.append(X.std(dim=(0, 2, 3)))

    mean = torch.stack(means, dim=0).mean(dim=0)
    std = torch.stack(stds, dim=0).mean(dim=0)
    return mean, std


def read_dataset(use_valid):
    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    train_trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        # 自动数据增强
        torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    test_trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    if use_valid:
        train_set = torchvision.datasets.ImageFolder(os.path.join('dataset', 'data', 'valid'), transform=train_trans)
    else:
        train_set = torchvision.datasets.ImageFolder(os.path.join('S:', 'dataset', 'cifar-10', 'train'), transform=train_trans)

    test_set = torchvision.datasets.ImageFolder(os.path.join('S:', 'dataset', 'cifar-10', 'test'), transform=test_trans)

    return train_set, test_set


def copyfile(file, dire):
    """ 拷贝文件file到dir
    :param file: path/to/file.type
    :param dire: path/to/directory
    """
    os.makedirs(dire, exist_ok=True)
    shutil.copy(file, dire)


def data_init(valid_ratio):
    """ 将train数据集划分一部分给valid集
    :param valid_ratio: 划分比例
    """
    old_path = os.path.join('dataset', 'train')
    new_path = os.path.join('dataset', 'data')
    if os.path.exists(new_path):
        return
    os.mkdir(new_path)
    os.mkdir(os.path.join(new_path, 'train'))
    os.mkdir(os.path.join(new_path, 'valid'))
    # os.mkdir(os.path.join(path, 'test'))

    labels = read_csv_labels('trainLabels.csv')
    # 出现次数最少的标签
    n = collections.Counter(labels.values()).most_common()[-1][1]  # 给字典的value出现次数排序, [-1]即出现次数最少的(元素, 次数)
    # valid集每个标签的样本量 = 最少的标签数 * valid_ratio
    n_per_label = max(1, math.floor(n * valid_ratio))
    labels_count = {}
    for file in os.listdir(old_path):
        label = labels[file.split('.')[0]]
        fdir1 = os.path.join(old_path, file)
        if label not in labels_count or labels_count[label] < n_per_label:
            fdir2 = os.path.join(new_path, 'valid', label)
            labels_count[label] = 1 if label not in labels_count else labels_count[label] + 1
        else:
            fdir2 = os.path.join(new_path, 'train', label)
        copyfile(fdir1, fdir2)

    pass


def get_dataset(batch_size, valid_ratio, use_valid):
    """
    :return: [train_iter valid_iter test_iter]
    """
    data_init(valid_ratio)
    train_dataset, test_dataset = read_dataset(use_valid)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    # test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)

    return train_iter, test_iter, train_dataset


class Hps:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = Hps(**v)
            self[k] = v

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)
