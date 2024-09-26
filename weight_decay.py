'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from dataloader import get_test_dataloader, get_training_dataloader
from tqdm import tqdm


import matplotlib.pyplot as plt
import numpy as np
import random
def get_acc(outputs, label):
    total = outputs.shape[0]
    probs, pred_y = outputs.data.max(dim=1) # 得到概率
    correct = (pred_y == label).sum().data
    return torch.div(correct, total)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        # torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss



def plot_history(epochs, Acc, Loss, lr, weight_decay_value):
    plt.rcParams['figure.figsize'] = (12.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    epoch_list = range(1, epochs + 1)
    plt.plot(epoch_list, Loss['train_loss'])
    plt.plot(epoch_list, Loss['test_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss Value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'history_Loss_weight_decay_{weight_decay_value}.png')
    plt.show()

    plt.plot(epoch_list, Acc['train_acc'])
    plt.plot(epoch_list, Acc['test_acc'])
    plt.xlabel('epoch')
    plt.ylabel('Acc Value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'history_Acc_weight_decay_{weight_decay_value}.png')
    plt.show()

    plt.plot(epoch_list, lr)
    plt.xlabel('epoch')
    plt.ylabel('Train LR')
    plt.savefig(f'history_Lr_weight_decay_{weight_decay_value}.png')
    plt.show()


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=True, help=' use GPU?')
    parser.add_argument('--batch-size', default=512, type=int, help="Batch Size for Training")
    parser.add_argument('--num-workers', default=2, type=int, help='num-workers')
    parser.add_argument('--net', type=str, choices=['LeNet5', 'AlexNet', 'VGG16', 'VGG19', 'ResNet18', 'ResNet34',
                                                    'DenseNet', 'MobileNetv1', 'MobileNetv2'], default='VGG16',
                        help='net type')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--patience', '-p', type=int, default=7, help='patience for Early stop')
    parser.add_argument('--optim', '-o', type=str, choices=['sgd', 'adam', 'adamw'], default='adamw',
                        help='choose optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay value')

    args = parser.parse_args()

    print(args)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Train Data
    trainloader = get_training_dataloader(batch_size=args.batch_size, num_workers=args.num_workers)
    testloader = get_test_dataloader(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    # Model
    print('==> Building model..')
    if args.net == 'VGG16':
        from nets.VGG import VGG

        net = VGG('VGG16')
    elif args.net == 'VGG19':
        from nets.VGG import VGG

        net = VGG('VGG19')
    elif args.net == 'ResNet18':
        from nets.ResNet import ResNet18

        net = ResNet18()
    elif args.net == 'ResNet34':
        from nets.ResNet import ResNet34

        net = ResNet34()
    elif args.net == 'LeNet5':
        from nets.LeNet5 import LeNet5

        net = LeNet5()
    elif args.net == 'AlexNet':
        from nets.AlexNet import AlexNet

        net = AlexNet()
    elif args.net == 'DenseNet':
        from nets.DenseNet import densenet_cifar

        net = densenet_cifar()
    elif args.net == 'MobileNetv1':
        from nets.MobileNetv1 import MobileNet

        net = MobileNet()
    elif args.net == 'MobileNetv2':
        from nets.MobileNetv2 import MobileNetV2

        net = MobileNetV2()

    if args.cuda:
        device = 'cuda'
        net = torch.nn.DataParallel(net)
        # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}_ckpt.pth'.format(args.net))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        args.lr = checkpoint['lr']

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94, verbose=True, patience=1,
                                                           min_lr=0.000001)  # 动态更新学习率

    epochs = args.epochs


    def train(net, trainloader, testloader, epochs, optimizer, criterion, scheduler, path='./model.pth', writer=None, verbose=False, weight_decay_value=5e-4):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best_acc = 0
        train_acc_list, test_acc_list = [], []
        train_loss_list, test_loss_list = [], []
        lr_list = []
        for i in range(epochs):
            train_loss = 0
            train_acc = 0
            test_loss = 0
            test_acc = 0
            if torch.cuda.is_available():
                net = net.to(device)
            net.train()
            train_step = len(trainloader)
            with tqdm(total=train_step, desc=f'Train Epoch {i + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
                for step, data in enumerate(trainloader, start=0):
                    im, label = data
                    im = im.to(device)
                    label = label.to(device)

                    optimizer.zero_grad()
                    # 释放内存
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    # formard
                    outputs = net(im)
                    loss = criterion(outputs, label)
                    # backward
                    loss.backward()
                    # 更新参数
                    optimizer.step()
                    # 累计损失
                    train_loss += loss.item()
                    train_acc += get_acc(outputs, label).item()
                    pbar.set_postfix(**{'Train Acc': train_acc / (step + 1),
                                        'Train Loss': train_loss / (step + 1)})
                    pbar.update(1)
                pbar.close()
            train_loss = train_loss / len(trainloader)
            train_acc = train_acc * 100 / len(trainloader)
            if verbose:
                train_acc_list.append(train_acc)
                train_loss_list.append(train_loss)
            # 记录学习率
            lr = optimizer.param_groups[0]['lr']
            if verbose:
                lr_list.append(lr)
            # 更新学习率
            scheduler.step(train_loss)
            if testloader is not None:
                net.eval()
                test_step = len(testloader)
                with torch.no_grad():
                    with tqdm(total=test_step, desc=f'Test Epoch {i + 1}/{epochs}', postfix=dict,
                              mininterval=0.3) as pbar:
                        for step, data in enumerate(testloader, start=0):
                            im, label = data
                            im = im.to(device)
                            label = label.to(device)
                            # 释放内存
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                            outputs = net(im)
                            loss = criterion(outputs, label)
                            test_loss += loss.item()
                            test_acc += get_acc(outputs, label).item()
                            pbar.set_postfix(**{'Test Acc': test_acc / (step + 1),
                                                'Test Loss': test_loss / (step + 1)})
                            pbar.update(1)
                        pbar.close()
                    test_loss = test_loss / len(testloader)
                    test_acc = test_acc * 100 / len(testloader)
                if verbose:
                    test_loss_list.append(test_loss)
                    test_acc_list.append(test_acc)
                print(
                    'Epoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                        i + 1, epochs, train_loss, train_acc, test_loss, test_acc, lr))
                trainAcc2_txt = "train_acc.txt"
                output=('Epoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}% Test Loss:{:>.6f}  Test Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                        i + 1, epochs, train_loss, train_acc, test_loss, test_acc, lr))
                with open(trainAcc2_txt, "a+") as f:
                    f.write(output + '\n')
                    f.close

            else:
                print('Epoch [{:>3d}/{:>3d}]  Train Loss:{:>.6f}  Train Acc:{:>3.2f}%  Learning Rate:{:>.6f}'.format(
                    i + 1, epochs, train_loss, train_acc, lr))

            # ====================== 使用 tensorboard ==================
            if writer is not None:
                writer.add_scalars('Loss', {'train': train_loss,
                                            'test': test_loss}, i + 1)
                writer.add_scalars('Acc', {'train': train_acc,
                                           'test': test_acc}, i + 1)
                writer.add_scalar('Learning Rate', lr, i + 1)
            # =========================================================
            # 如果取得更好的准确率，就保存模型
            if test_acc > best_acc:
                torch.save(net, path)
                best_acc = test_acc
        Acc = {}
        Loss = {}
        Acc['train_acc'] = train_acc_list
        Acc['test_acc'] = test_acc_list
        Loss['train_loss'] = train_loss_list
        Loss['test_loss'] = test_loss_list
        Lr = lr_list
        return Acc, Loss, Lr



    save_path = './model/weight_decay.pth'
    Acc, Loss, Lr = train(net, trainloader, testloader, epochs, optimizer, criterion, scheduler, save_path, verbose=True, weight_decay_value=args.weight_decay)
    plot_history(epochs, Acc, Loss, Lr, args.weight_decay)

    torch.cuda.empty_cache()