import collections
import math
import os
import shutil


def read_csv_labels(filename):
    with open(os.path.join('dataset', filename), 'r') as f:
        lines = f.readlines()[1:]
    lines = [line.strip().split(',') for line in lines]
    return dict((idx, label) for idx, label in lines)


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
