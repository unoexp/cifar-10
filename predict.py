import argparse
import pandas as pd

import config
import data_utiles
import torch
from models import resnet34


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    args = parser.parse_args()

    net = resnet34(3, 10)
    state_dict = torch.load(args.model)
    net.load_state_dict(state_dict['model'])
    hps = config.read_params('config.json')
    paths = [hps.data.valid_set, hps.data.train_set, hps.data.test_set]
    batch_size = hps.train.batch_size

    net = net.to('cuda:0')
    train_iter, test_iter, train_dataset = data_utiles.get_dataset(batch_size, 0, paths)
    predict_test(test_iter, net, train_dataset)


def predict_test(test_iter, net, train_dataset):
    res = []
    net.eval()
    for i, (img, _) in enumerate(test_iter):
        predict = net(img.to(0))
        res.extend(predict.argmax(dim=1).type(torch.int32).cpu().numpy())
    s_ids = list(range(1, len(res) + 1))
    s_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': s_ids, 'label': res})
    df['label'] = df['label'].apply(lambda x: train_dataset.classes[x])
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
