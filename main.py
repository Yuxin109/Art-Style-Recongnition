import argparse,os
import timm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from model import BaseModel
from DataLoader import dataload
from Trainer import Trainer

def main(args):

    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)

    model = BaseModel(out_feature=4)
    if torch.cuda.is_available():
        model = model.cuda()

    train_dataset, test_dataset = dataload(args.root)

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers
    )

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=trainloader,
        testloader=testloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        patience=args.patience,
        save_path=args.save_path
    )

    for epoch in range(args.epoch):
        early_stop, early_stop_count, best_acc = trainer.train_epoch(epoch)

        if early_stop:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--num-class', type=int, default=4)
    arg('--size', type=int, default=256)
    arg('--root', type=str, default="C://Users/user/Downloads/artdataset/artdataset")
    arg('--batch-size', type=int, default=16)
    arg('--num-workers', type=int, default=0)
    arg('--shuffle', type=bool, default=True)
    arg('--optimizer', type=str, default='adam')
    arg('--lr', type=float, default=0.0001)
    arg('--epoch', type=int, default=100)
    arg('--load-model-path', type=str, default=None)
    arg('--patience', type=int, default=5)
    arg('--exp-name', type=str, default="CV_final_project")
    arg('--save-path', type=str, default="ckpt_transforms")
    arg('--gpus', type=str, default='0')
    arg('--seed', type=int, default=123)

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    main(args)

