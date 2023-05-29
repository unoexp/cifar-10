import pickle

import torch
import data_utiles
from torch import nn
import pandas as pd
from models import resnet18


def main():
    assert torch.cuda.is_available() is True
    hps = data_utiles.read_params('config.json')
    run(hps)


def run(hps):
    use_valid = hps.data.use_valid
    valid_ratio = hps.data.valid_ratio

    batch_size = hps.train.batch_size
    lr, wd = hps.train.lr_rate, hps.train.weight_decay
    lr_period, lr_decay = hps.train.lr_period, hps.train.lr_decay
    max_epochs = hps.train.max_epochs

    num_classes = hps.model.num_classes

    train_iter, test_iter, train_dataset = data_utiles.get_dataset(batch_size, valid_ratio, use_valid)

    net = resnet18(num_classes, 3)

    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    net = nn.DataParallel(net, [0]).to(0)
    for epoch in range(max_epochs):
        ls, acc = train_epoch(net, train_iter, trainer, loss)
        print(f'epoch:{epoch}, loss:{ls}, acc:{acc}')
        scheduler.step()
        if epoch % 10 == 0:
            with open('checkpoint' + str(epoch) + '.pkl', 'wb+') as f:
                pickle.dump(net.state_dict(), f)
    net.eval()
    res = []
    for i, (img, _) in enumerate(test_iter):
        predict = net(img.to(0))
        res.extend(predict.argmax(dim=1).type(torch.int32).cpu().numpy())
        if i == 100:
            break
    s_ids = list(range(1, 101 * 128 + 1))
    s_ids.sort(key=lambda x: str(x))
    print(len(s_ids), len(res))
    df = pd.DataFrame({'id': s_ids, 'label': res})
    df['label'] = df['label'].apply(lambda x: train_dataset.classes[x])
    df.to_csv('submission.csv', index=False)

def train_epoch(net, train_iter, trainer, loss):
    net.train()
    ls = 0
    cnt0, cnt1 = 0, 0
    for i, (img, label) in enumerate(train_iter):
        img = img.to(0)
        label = label.to(0)

        trainer.zero_grad()
        predict = net(img)
        l = loss(predict, label)
        l.sum().backward()
        trainer.step()
        loss_sum = l.sum()
        ls += loss_sum
        a, b = accuracy(predict, label)
        cnt0, cnt1 = cnt0 + a, cnt1 + b
    return ls / cnt1, cnt0 / cnt1


def accuracy(t, y):
    t = t.argmax(dim=1)
    cmp = y.type(t.dtype) == t
    return cmp.sum(), len(y)


if __name__ == "__main__":
    main()
