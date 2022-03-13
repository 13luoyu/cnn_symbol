import os

from PIL import Image

import download_file
import numpy as np
import pickle
import torch
from torch import nn
from d2l import torch as d2l


# 每个batch文件包含一个字典，每个字典包含有：
#
# Data 一个10000 * 3072
# 的numpy数组，数据类型是无符号整形uint8。这个数组的每一行存储了32 * 32
# 大小的彩色图像（32 * 32 * 3通道 = 3072）。
# 另外，图像是以行的顺序存储的，也就是说前32个数就是这幅图的像素矩阵的第一行。

# labels  一个范围在0 - 9
# 的含有10000个数的列表（一维的数组）。第i个数就是第i个图像的类标。

def load_batch(fpath):
    """从下载好的文件中读取训练集和测试集的图片和标签，返回四个数组"""
    # 训练集
    images = []
    labels = []
    for i in range(5):
        filename = os.path.join(fpath, "data_batch_" + str(i + 1))
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            image = datadict['data']
            label = datadict['labels']
            image = image.reshape(10000, 3, 32, 32).astype('float')
            label = np.array(label)
            images.append(image)
            labels.append(label)
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    # 测试集
    filename = os.path.join(fpath, "test_batch")
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        image = datadict['data']
        label = datadict['labels']
        image = image.reshape(10000, 3, 32, 32).astype('float')
        label = np.array(label)
    return images, labels, image, label


class CIFAR10:
    """保存了cifar10数据集的训练集、验证集、测试集"""

    def __init__(self):
        cifar_dir = download_file.download_and_extract("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
        cifar_dir = os.path.join('data', 'cifar-10-batches-py')
        train_data, train_label, test_data, test_label = \
            load_batch(cifar_dir)
        # 将训练集合再分，分为训练集和验证集，其数量为45000:5000
        VALIDATION_SIZE = 5000
        self.test_data = test_data
        self.test_labels = test_label
        self.valid_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.valid_labels = train_label[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_label[VALIDATION_SIZE:]


# c = CIFAR10()


class CIFAR10_Model:
    def __init__(self, restore=None, use_softmax=False):
        activation = nn.ReLU()
        if restore:
            net = torch.load(restore)
        else:
            net = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64,
                          kernel_size=(3, 3), padding=(1, 1)),
                activation,
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=(3, 3), padding=(1, 1)),
                activation,
                nn.MaxPool2d(kernel_size=(2, 2)),

                nn.Conv2d(in_channels=64, out_channels=128,
                          kernel_size=(3, 3), padding=(1, 1)),
                activation,
                nn.Conv2d(in_channels=128, out_channels=128,
                          kernel_size=(3, 3), padding=(1, 1)),
                activation,
                nn.MaxPool2d(kernel_size=(2, 2)),

                nn.Flatten(),
                nn.Linear(8192, 256),  # 128*8*8
                activation,
                nn.Linear(256, 256),
                activation,
                nn.Linear(256, 10)
            )
            if use_softmax:
                net.add_module('softmax', nn.Softmax(dim=-1))

        self.net = net
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10


# m = CIFAR10_Model()



def train(save_filename=None):
    c = CIFAR10()
    m = CIFAR10_Model()
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(c.train_data, dtype=torch.float32), torch.tensor(c.train_labels, dtype=torch.long)
    )
    train_iter = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True
    )
    valid_dataset = torch.utils.data.TensorDataset(
        torch.tensor(c.valid_data, dtype=torch.float32), torch.tensor(c.valid_labels, dtype=torch.long)
    )
    valid_iter = torch.utils.data.DataLoader(
        valid_dataset, batch_size, shuffle=True, drop_last=True
    )

    num_epoches, lr = 8, 0.01
    d2l.train_ch13(m.net, train_iter, valid_iter,
                   nn.CrossEntropyLoss(), torch.optim.SGD(m.net.parameters(), lr=lr),
                   num_epoches)
    d2l.plt.show()

    if save_filename is None:
        save_filename = 'model/CIFAR10_Model.pth'
    torch.save(m.net, save_filename)

# train()

def read_mnist_and_save(i):
    c = CIFAR10()
    data = c.train_data[i]
    Image.fromarray(data.transpose(1, 2, 0).astype('uint8')).save('images/c' + str(c.train_labels[i]) + '.jpg')