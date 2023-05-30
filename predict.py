import argparse
import pandas as pd

import data_utiles
from torch import nn
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    args = parser.parse_args()
    net = torch.load(args.model)
    net = nn.DataParallel(net, [0]).to(0)
    train_iter, test_iter, train_dataset = data_utiles.get_dataset(128, 0.8, 0)
    predict_test(test_iter, net, train_dataset)


def predict_test(test_iter, net, train_dataset):
    res = []
    net.eval()
    for i, (img, _) in enumerate(test_iter):
        predict = net(img.to(0))
        res.extend(predict.argmax(dim=1).type(torch.int32).cpu().numpy())
        if i == 100:
            break
    s_ids = list(range(1, 101*128 + 1))
    s_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': s_ids, 'label': res})
    df['label'] = df['label'].apply(lambda x: train_dataset.classes[x])
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
