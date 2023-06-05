import argparse

import torch
import data_utiles
from torch import nn
from models import resnet18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default=None)
    args = parser.parse_args()
    assert torch.cuda.is_available() is True
    hps = data_utiles.read_params('config.json')
    if args.model is None:
        run(hps)
    else:
        run(hps, args.model)


def run(hps, model=None):
    use_valid = hps.data.use_valid
    valid_ratio = hps.data.valid_ratio

    batch_size = hps.train.batch_size
    lr, wd = hps.train.lr_rate, hps.train.weight_decay
    lr_period, lr_decay = hps.train.lr_period, hps.train.lr_decay
    max_epochs = hps.train.max_epochs

    num_classes = hps.model.num_classes
    save_interval = hps.model.save_interval

    train_iter, test_iter, train_dataset = data_utiles.get_dataset(batch_size, valid_ratio, use_valid)
    net = resnet18(num_classes, 3)
    if model is not None:
        state_dict = torch.load(model)
        net.load_state_dict(state_dict['model'])

    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)

    net = nn.DataParallel(net, [0]).to(0)
    for epoch in range(max_epochs):
        ls, acc = train_epoch(net, train_iter, trainer, loss)
        print(f'epoch:{epoch}, loss:{ls}, acc:{acc}')
        scheduler.step()
        if epoch % save_interval == 0:
            torch.save({'model': net.state_dict()}, 'check_point_' + str(epoch))


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
