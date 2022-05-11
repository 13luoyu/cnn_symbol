import numpy as np
import os

from PIL import Image

import download_file
import gzip
import torch
from torch import nn
from d2l import torch as d2l

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images * 28 * 28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1).transpose((0,3,1,2))
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    # return (np.arange(10) == labels[:, None]).astype(np.float32)
    return labels


class MNIST:
    def __init__(self):
        train_images = download_file.download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
        t10k_images = download_file.download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
        train_labels = download_file.download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
        t10k_labels = download_file.download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
        train_data = extract_data(train_images, 60000)
        train_labels = extract_labels(train_labels, 60000)
        self.test_data = extract_data(t10k_images, 10000)
        self.test_labels = extract_labels(t10k_labels, 10000)
        VALIDATION_SIZE = 5000
        self.valid_data = train_data[:VALIDATION_SIZE,:,:,:]
        self.valid_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:,:,:,:]
        self.train_labels = train_labels[VALIDATION_SIZE:]

# c = MNIST()
# print(c.train_labels.shape)
# print(c.train_data)


class MNIST_Model:
    def __init__(self, restore=None, use_softmax=False):
        """
        :param restore: 如果不为None，则使用训练好的模型，否则使用未训练的模型
        :param use_softmax: 如果为True，在网络最后添加softmax层，否则不添加
        """
        activation = nn.ReLU()
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        if restore:
            net = torch.load(restore)
        else:
            net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
                activation,
                nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
                activation,
                nn.MaxPool2d(kernel_size=(2, 2)),

                nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
                activation,
                nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
                activation,  # 64*14*14  todo
                nn.MaxPool2d(kernel_size=(2, 2)),  # 64*7*7

                nn.Flatten(),
                nn.Linear(3136, 200),
                activation,
                nn.Linear(200, 200),
                nn.BatchNorm1d(200),
                activation,
                nn.Linear(200, 10)
            )
            if use_softmax:
                net.add_module('softmax', nn.Softmax(dim=-1))

        self.net = net

# m = MNIST_Model()


def train(save_filename=None):
    c = MNIST()
    m = MNIST_Model()
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
        save_filename = 'model/MNIST_Model.pth'
    torch.save(m.net, save_filename)

# train()  # loss 0.001, train acc 0.995, test acc 0.994

def read_mnist_and_save(i):
    c = MNIST()
    data = c.train_data[i]
    Image.fromarray(((data + 0.5) * 255).astype('uint8').reshape(28, 28), mode='L').save('images/' + str(c.train_labels[i]) + '.jpg')
