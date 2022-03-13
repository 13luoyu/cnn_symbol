import os
import numpy as np
from torchvision import datasets

import download_file
from PIL import Image
import torch
from torch import nn
import torchvision
from d2l import torch as d2l

def load_images(path):
    labels = []
    image_names = []
    with open(os.path.join(path, 'wnids.txt')) as wnid:
        for line in wnid:
            labels.append(line.strip('\n'))
    for label in labels:
        txt_path = os.path.join(path, 'train', label, label+'_boxes.txt')
        image_name = []
        with open(txt_path) as txt:
            for line in txt:
                image_name.append(line.strip('\n').split('\t')[0])
        image_names.append(image_name)
    x_train = []
    y_train = []
    for i in range(len(labels)):
        for j in range(len(image_names[i])):
            img = Image.open(os.path.join(path, 'train', labels[i], 'images', image_names[i][j]))
            # print(np.array(img).shape)  # (64,64,3)
            x_train.append(np.array(img).tolist())
            y_train.append(labels[i])

    x_valid = []
    y_valid = []
    with open(os.path.join(path, 'val', 'val_annotations.txt')) as txt:
        for line in txt:
            image_name = line.strip('\n').split('\t')[0]
            label = line.strip('\n').split('\t')[1]
            img = Image.open(os.path.join(path, 'val', 'images', image_name))
            x_valid.append(np.array(img).tolist())
            y_valid.append(label)

    x_test = []
    y_test = []
    # for image in os.listdir(os.path.join(path, 'test', 'images')):
    #     img = Image.open(os.path.join(path, 'test', 'images', image))
    #     x_test.append(np.array(img))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


class tinyImagenet():
    def __init__(self):
        path = download_file.download_and_extract('http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        # print(path)  # data\tiny-imagenet-200
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_images(path)
        self.train_data = x_train
        self.train_labels = y_train
        self.valid_data = x_valid
        self.valid_labels = y_valid
        self.test_data = x_test
        self.test_labels = y_test


# c = tinyImagenet()  # 非常耗时


class TinyImagenet_Model():
    def __init__(self, restore=None): # 使用resnet18
        self.net = torchvision.models.resnet18(pretrained=True)
        self.net.fc = nn.Sequential(
            nn.Linear(512, 200)
        )
        if restore:
            self.net.load_state_dict(torch.load(restore))

# m = TinyImagenet_Model()

def train():
     batch_size = 32
     # train_dataset = torch.utils.data.TensorDataset(
     #     torch.tensor(c.train_data, dtype=torch.float32),
     #     torch.tensor(c.train_labels, dtype=torch.long)
     # )
     # train_iter = torch.utils.data.DataLoader(
     #     train_dataset, batch_size, shuffle=True, drop_last=True
     # )
     # valid_dataset = torch.utils.data.TensorDataset(
     #     torch.tensor(c.valid_data, dtype=torch.float32),
     #     torch.tensor(c.valid_labels, dtype=torch.long)
     # )
     # valid_iter = torch.utils.data.DataLoader(
     #     valid_dataset, batch_size, shuffle=True, drop_last=True
     # )

     trainset = datasets.ImageFolder(root=os.path.join('data/tiny-imagenet-200', 'train'), transform=torchvision.transforms.ToTensor())
     testset = datasets.ImageFolder(root=os.path.join('data/tiny-imagenet-200', 'val'), transform=torchvision.transforms.ToTensor())
     train_iter = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
     valid_iter = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

     num_epoches, lr = 8, 0.01
     d2l.train_ch13(m.net, train_iter, valid_iter,
                    nn.CrossEntropyLoss(), torch.optim.SGD(m.net.parameters(), lr=lr),
                    num_epoches)
     d2l.plt.show()
     torch.save(m.net.state_dict(), 'TinyImagenet_Model.pth')

# train()














