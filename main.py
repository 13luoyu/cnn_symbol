import torch
from torch import nn
import numpy as np
import activations
from PIL import Image
import copy
import time

class Model():
    def __init__(self, net=None):
        if net is None:
            net = nn.Sequential(
                nn.Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(10, 0.001),
                nn.AvgPool2d((3, 3), padding=(1, 1)),
                nn.Flatten(),
                nn.Linear(40, 5)
            )
        self.net = net

def verify_robustness(model, x_record, true_label, x_maxpool_loc, input, disturb_radius, dataset):
    '''
    反向传播，验证指定的模型是否鲁棒。
    :param model:  训练好的模型
    :param x_record:  每一层的x表达式，x[0]为第0层输入，x[1]为第0层输出，第1层输入
    :param true_label:  正确的分类结果
    :param x_maxpool_loc:  记录最大池层中，最大值取的输入的位置
    :param input:  输入x的真实值
    :param disturb_radius:  扰动范围
    :param dataset:  数据集
    :return:  status, image（若status=1，表示未找到对抗性示例，image为空，status=-1，image为对抗性示例）
    '''
    x = x_record[-1]
    if dataset == 'MNIST':
        input_l = np.maximum((input - disturb_radius), -0.5).reshape(input.shape[0] * input.shape[1] * input.shape[2])
        input_h = np.minimum((input + disturb_radius), 0.5).reshape(input.shape[0] * input.shape[1] * input.shape[2])
    elif dataset == 'CIFAR10':
        input_l = np.maximum((input - disturb_radius), 0).reshape(input.shape[0] * input.shape[1] * input.shape[2])
        input_h = np.minimum((input + disturb_radius), 1).reshape(input.shape[0] * input.shape[1] * input.shape[2])

    # 计算表达式x[true_label]-x[i]的结果
    y = []
    for i in range(x.shape[0]):
        # 计算x[true_label]与其他相减的表达式
        # y_local = x[true_label]
        y_local = dict()
        for xi, wi in x[true_label].items():
            y_local[xi] = wi
        for xi, wi in x[i].items():
            y_local.setdefault(xi, 0)
            y_local[xi] -= wi
        y.append(y_local)

    # 计算y表达式的最小值，以及可能产生错误的y的序号(可能产生错误指的是x[true_label]-x[i]的最小值<=0)
    eqorbelow_zero_index = []  # y的最小值<=0的index
    for i, y_local in enumerate(y):
        val = 0.0
        for xi, wi in y_local.items():  # 求表达式最小值，常数直接相加，wi>0取xi最小值，wi<0取xi最大值
            if xi == '1':
                val += wi
                continue
            index = int(xi[1:])
            if wi < 0:
                val += wi * input_h[index]
            else:
                val += wi * input_l[index]
        if val <= 0 and i != true_label:  # 表达式可能<=0，即出现分类错误
            eqorbelow_zero_index.append(i)


    if len(eqorbelow_zero_index) == 0:  # 没有可能的分类错误
        return 1, None


    # 反向传播，依据本层输出推出本层输入
    # 记录输出表达式，输出关联符号的应取的是最大or最小值，权重，一直向前，直到非线性层或开始
    for cnt, index in enumerate(eqorbelow_zero_index):  # 对每个可能的分类错误
        express = y[index]
        maxpool_num = len(x_maxpool_loc) - 1
        weight_dic = dict()  # 反向传播中记录的权重之积，用以当冲突时进行判定
        want_min, want_max = [], []  # 一些xi希望取大，一些希望取小，用以计算最终的输出
        input_cp = np.zeros(input.shape, dtype=np.float32)  # 输入的真实值
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                for k in range(input.shape[2]):
                    input_cp[i,j,k] = input[i,j,k]

        # 看结果要求哪些xi取大，哪些取小
        for xi, wi in express.items():
            if xi == '1':
                continue
            weight_dic.setdefault(xi, 0)
            weight_dic[xi] += np.abs(wi)
            if wi > 0:
                want_min.append(xi)
            else:
                want_max.append(xi)

        # 开始反向传播
        for i in range(len(model.net)-1, -1, -1):
            layer = model.net[i]
            print('Verify Layer {}, type: {}'.format(i, layer.__class__.__name__))

            # 如果是线性层，不做处理，原因在于线性层只对w作修改，而w的权重正负在非线性层和结尾层会考虑。

            # 激活函数层，依据输出范围计算输入范围，计算输入每个符号希望取大还是取小
            # 该层输入应该是{xi:wi,xj:wj,...}，输出应该是{xi:1}
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Sigmoid) or isinstance(layer, nn.Tanh):
                in_express = x_record[i]  # x_record[i]是第i层的输入
                new_want_min = []
                new_want_max = []
                want_min_weight_dic = dict()
                want_max_weight_dic = dict()
                if len(in_express.shape) == 3:  # 卷积层输出

                    print('activation, shape=3, want_min=' + str(len(want_min)))
                    for xii in want_min:  # 希望输出xii取小，因为激活函数单调性，所以输入整个表达式取小
                        index = int(xii[1:])
                        j = int(index/(in_express.shape[1]*in_express.shape[2]))
                        k = int((index-j*(in_express.shape[1]*in_express.shape[2]))/in_express.shape[2])
                        l = int((index-j*(in_express.shape[1]*in_express.shape[2]))%in_express.shape[2])
                        x = in_express[j,k,l]  # 要这个表达式最小
                        for xi, wi in x.items():
                            if xi == '1':
                                continue
                            if wi < 0:
                                if xi not in new_want_max:
                                    new_want_max.append(xi)
                                want_max_weight_dic.setdefault(xi, 0)
                                want_max_weight_dic[xi] += weight_dic[xii]*np.abs(wi)
                            else:
                                if xi not in new_want_min:
                                    new_want_min.append(xi)
                                want_min_weight_dic.setdefault(xi, 0)
                                want_min_weight_dic[xi] += weight_dic[xii]*wi

                    print('activation, shape=3, want_max=' + str(len(want_max)))
                    for xii in want_max:
                        index = int(xii[1:])
                        j = int(index / (in_express.shape[1] * in_express.shape[2]))
                        k = int((index - j * (in_express.shape[1] * in_express.shape[2])) / in_express.shape[2])
                        l = int((index - j * (in_express.shape[1] * in_express.shape[2])) % in_express.shape[2])
                        x = in_express[j, k, l]  # 要这个表达式最小
                        for xi, wi in x.items():
                            if xi == '1':
                                continue
                            if wi < 0:
                                if xi not in new_want_min:
                                    new_want_min.append(xi)
                                want_min_weight_dic.setdefault(xi, 0)
                                want_min_weight_dic[xi] += weight_dic[xii]*np.abs(wi)
                            else:
                                if xi not in new_want_max:
                                    new_want_max.append(xi)
                                want_max_weight_dic.setdefault(xi, 0)
                                want_max_weight_dic[xi] += weight_dic[xii]*wi
                elif len(in_express.shape) == 1:

                    print('activation, shape=1, want_min='+str(len(want_min)))
                    for xii in want_min:
                        index = int(xii[1:])
                        x = in_express[index]  # 要这个表达式最小
                        for xi, wi in x.items():
                            if xi == '1':
                                continue
                            if wi < 0:
                                if xi not in new_want_max:
                                    new_want_max.append(xi)
                                want_max_weight_dic.setdefault(xi, 0)
                                want_max_weight_dic[xi] += weight_dic[xii] * np.abs(wi)
                            else:
                                if xi not in new_want_min:
                                    new_want_min.append(xi)
                                want_min_weight_dic.setdefault(xi, 0)
                                want_min_weight_dic[xi] += weight_dic[xii] * wi

                    print('activation, shape=1, want_max='+str(len(want_max)))
                    for xii in want_max:
                        index = int(xii[1:])
                        x = in_express[index]  # 要这个表达式最小
                        for xi, wi in x.items():
                            if xi == '1':
                                continue
                            if wi < 0:
                                if xi not in new_want_min:
                                    new_want_min.append(xi)
                                want_min_weight_dic.setdefault(xi, 0)
                                want_min_weight_dic[xi] += weight_dic[xii] * np.abs(wi)
                            else:
                                if xi not in new_want_max:
                                    new_want_max.append(xi)
                                want_max_weight_dic.setdefault(xi, 0)
                                want_max_weight_dic[xi] += weight_dic[xii] * wi

                # 更新想要的取大和取小的xi，更新它们的权重
                want_max.clear()
                want_min.clear()
                weight_dic.clear()
                print('activation, shape=1, new_want_min=' + str(len(new_want_min)))
                print('activation, shape=1, new_want_max=' + str(len(new_want_max)))
                for xi in new_want_max:
                    if xi in new_want_min:
                        max_min = want_max_weight_dic[xi] - want_min_weight_dic[xi]
                        if max_min > 0:  # 取大
                            want_max.append(xi)
                        else:
                            want_min.append(xi)
                        weight_dic[xi] = np.abs(max_min)
                    else:
                        want_max.append(xi)
                        weight_dic[xi] = want_max_weight_dic[xi]
                for xi in new_want_min:
                    if xi in new_want_max:
                        continue
                    else:
                        want_min.append(xi)
                        weight_dic[xi] = want_min_weight_dic[xi]

            # 最大池层，选出n个中最大的一个作为结果，因为输入有上下界，输出选择上界中最大的和下界中最大的，不一定是同一个点
            # 现在倒推，如果取小则取输入下界最大的点，取大则取输入上界最大的点。
            elif isinstance(layer, nn.MaxPool2d):
                in_express = x_record[i]
                new_want_min = []
                new_want_max = []
                want_min_weight_dic = dict()
                want_max_weight_dic = dict()

                print('max-pool, want_min=' + str(len(want_min)))
                for xii in want_min:  # (通道数，高，宽)
                    index = int(xii[1:])
                    min_loc = x_maxpool_loc[maxpool_num][index][0]  # 输入下界最大的点所在位置

                    index = min_loc  # 输入取min的表达式在in_express的位置
                    j = int(index / (in_express.shape[1] * in_express.shape[2]))
                    k = int((index - j * (in_express.shape[1] * in_express.shape[2])) / in_express.shape[2])
                    l = int((index - j * (in_express.shape[1] * in_express.shape[2])) % in_express.shape[2])
                    x = in_express[j, k, l]  # 要这个表达式最小
                    for xi, wi in x.items():
                        if xi == '1':
                            continue
                        if wi < 0:
                            if xi not in new_want_max:
                                new_want_max.append(xi)
                            want_max_weight_dic.setdefault(xi, 0)
                            want_max_weight_dic[xi] += weight_dic[xii]*np.abs(wi)
                        else:
                            if xi not in new_want_min:
                                new_want_min.append(xi)
                            want_min_weight_dic.setdefault(xi, 0)
                            want_min_weight_dic[xi] += weight_dic[xii]*wi

                print('max-pool, want_max=' + str(len(want_max)))
                for xii in want_max:
                    index = int(xii[1:])  # 输出x的位置
                    max_loc = x_maxpool_loc[maxpool_num][index][1]  # 在kernel的第几个
                    index = max_loc  # 输入取max的表达式位置
                    j = int(index / (in_express.shape[1] * in_express.shape[2]))
                    k = int((index - j * (in_express.shape[1] * in_express.shape[2])) / in_express.shape[2])
                    l = int((index - j * (in_express.shape[1] * in_express.shape[2])) % in_express.shape[2])
                    x = in_express[j, k, l]  # 要这个表达式最小
                    for xi, wi in x.items():
                        if xi == '1':
                            continue
                        if wi < 0:
                            if xi not in new_want_min:
                                new_want_min.append(xi)
                            want_min_weight_dic.setdefault(xi, 0)
                            want_min_weight_dic[xi] += weight_dic[xii]*np.abs(wi)
                        else:
                            if xi not in new_want_max:
                                new_want_max.append(xi)
                            want_max_weight_dic.setdefault(xi, 0)
                            want_max_weight_dic[xi] += weight_dic[xii]*wi
                want_max.clear()
                want_min.clear()
                weight_dic.clear()
                print('max-pool, new_want_min=' + str(len(new_want_min)))
                print('max-pool, new_want_max=' + str(len(new_want_max)))
                for xi in new_want_max:
                    if xi in new_want_min:
                        max_min = want_max_weight_dic[xi] - want_min_weight_dic[xi]
                        if max_min > 0:  # 取大
                            want_max.append(xi)
                        else:
                            want_min.append(xi)
                        weight_dic[xi] = np.abs(max_min)
                    else:
                        want_max.append(xi)
                        weight_dic[xi] = want_max_weight_dic[xi]
                for xi in new_want_min:
                    if xi in new_want_max:
                        continue
                    else:
                        want_min.append(xi)
                        weight_dic[xi] = want_min_weight_dic[xi]
                maxpool_num -= 1


            # 全部层遍历结束，得到每个输入符号取大取小，由此计算干扰输入，并带入网络，看预测结果是否正确
            if i == 0:

                print('i=0, want_max=' + str(len(want_max)))
                for x in want_max:
                    index = int(x[1:])
                    j = int(index / (input.shape[1] * input.shape[2]))
                    k = int((index - j * (input.shape[1] * input.shape[2])) / input.shape[2])
                    l = int((index - j * (input.shape[1] * input.shape[2])) % input.shape[2])
                    if x in want_min:
                        max_min = want_max[x] - want_min[x]
                        if max_min > 0:
                            input_cp[j,k,l] += disturb_radius
                        else:
                            input_cp[j,k,l] -= disturb_radius
                    else:
                        input_cp[j,k,l] += disturb_radius

                print('i=0, want_min=' + str(len(want_min)))
                for x in want_min:
                    index = int(x[1:])
                    j = int(index / (input.shape[1] * input.shape[2]))
                    k = int((index - j * (input.shape[1] * input.shape[2])) / input.shape[2])
                    l = int((index - j * (input.shape[1] * input.shape[2])) % input.shape[2])
                    if x in want_max:
                        continue
                    else:
                        input_cp[j,k,l] -= disturb_radius
                label = torch.argmax(model.net(torch.tensor(input_cp.reshape(1,input.shape[0],input.shape[1],input.shape[2]))))
                if label != true_label:
                    return -1, input_cp
                else:
                    print("predict label is correct: {}, continue...".format(label))
                    break
    return 1, None





def run(dataset, input, disturb_radius, net, true_label):
    '''
    主程序，输入一个图片、一个网络模型，一个扰动半径，判断网络是否鲁棒
    :param true_label: 真实标签
    :param input: 输入图像路径
    :param disturb_radius: 扰动半径
    :param net: 网络模型文件
    :param dataset: 数据集
    :return: 网络是否鲁棒，1表示鲁棒，-1表示不鲁棒，0表示无法确定，-2表示分类错误
    '''
    start_time = time.time()
    if dataset == 'MNIST':
        input = np.array(Image.open(input), dtype=np.float32).reshape(1,28,28)  # (28,28) => (1,28,28)
        input = input / 255.0 - 0.5
    elif dataset == 'CIFAR10':
        input = np.array(Image.open(input), dtype=np.float32).transpose(2,0,1)  # (h,w,c) => (c,h,w)
    else:
        raise Exception("Unknown dataset: " + dataset)
    net = torch.load(net).to('cpu')  # 读入网络
    model = Model(net)
    model.net.eval()  # 验证而非训练模式
    y = model.net(torch.tensor(input.reshape(1,input.shape[0],input.shape[1],input.shape[2])))
    label = torch.argmax(y)
    if label != true_label:  # 预测不正确，直接退出
        return -2, None

    x = []
    for i in range(input.shape[0] * input.shape[1] * input.shape[2]):
        x.append({'x' + str(i): 1})
    x = np.array(x).reshape(input.shape[0], input.shape[1], input.shape[2])  # 通道，高，宽，每个块有1个元素，该元素为字典(表达式：权重)
    # x是一个数组，表示输入，其中每个数组元素应该是一个表达式，比如a*x0+b*x1+...+n*xn
    # 为了方便表达，将其表示为一个数组，数组元素为字典，包含一个变量x和一个权重a，在计算时ax之间使用+连接
    if dataset == 'MNIST':
        x_true_l = np.maximum((input - disturb_radius), -0.5).reshape(input.shape[0]*input.shape[1]*input.shape[2])
        x_true_h = np.minimum((input + disturb_radius), 0.5).reshape(input.shape[0]*input.shape[1]*input.shape[2])
    elif dataset == 'CIFAR10':
        x_true_l = np.maximum((input - disturb_radius), 0).reshape(input.shape[0] * input.shape[1] * input.shape[2])
        x_true_h = np.minimum((input + disturb_radius), 1).reshape(input.shape[0] * input.shape[1] * input.shape[2])
    x_record = []  # 记录每层x表达式
    x_maxpool_loc = []  # 记录在maxpool层选的是哪个表达式

    for i, layer in enumerate(list(model.net)):
        if isinstance(layer, nn.Conv2d):
            print('Layer {}, type: {}'.format(i, layer.__class__.__name__))
            w = layer.weight.data.numpy()  # kernel,(输出通道，输入通道，核高，核宽)
            b = layer.bias.data.numpy()  # 偏移
            padding = layer.padding
            if padding == 0:
                padding = (0, 0)
            stride = layer.stride

            new_h = int((x.shape[1] + padding[0] * 2 - w.shape[2]) / stride[0]) + 1  # 输出高
            new_w = int((x.shape[2] + padding[1] * 2 - w.shape[3]) / stride[1]) + 1  # 输出宽
            y = np.zeros((w.shape[0], new_h, new_w), dtype=dict)
            x_pad = np.zeros((w.shape[1], x.shape[1] + padding[0] * 2, x.shape[2] + padding[1] * 2),
                             dtype=dict)  # 经过填充后的x
            for k in range(w.shape[1]):
                x_pad[k, padding[0]:padding[0] + x.shape[1], padding[1]:padding[1] + x.shape[2]] = x[k]

            for j in range(w.shape[0]):  # 输出通道数
                for k in range(w.shape[1]):  # 输入通道数, x.shape[0]
                    for l in range(new_h):  # 输出高
                        h_start = stride[0] * l
                        h_end = h_start + w.shape[2]
                        for m in range(new_w):  # 输出宽
                            w_start = stride[1] * m
                            w_end = w_start + w.shape[3]
                            local_y = dict()
                            for hh in range(h_start, h_end):
                                for ww in range(w_start, w_end):
                                    if x_pad[k, hh, ww] == 0:
                                        continue
                                    for xi, wi in x_pad[k, hh, ww].items():
                                        local_y.setdefault(xi, 0)
                                        local_y[xi] += wi * w[j, k, hh - h_start, ww - w_start]
                            if y[j, l, m] == 0:
                                y[j, l, m] = local_y
                            else:
                                for xi, wi in local_y.items():
                                    y[j, l, m].setdefault(xi, 0)
                                    y[j, l, m][xi] += wi
                            if k == 0:  # 重要！这里保证偏置bias仅仅在每个输出通道上加一次
                                y[j, l, m].setdefault('1', 0)
                                y[j, l, m]['1'] += b[j]
            x_record.append(x)
            x = y

        elif isinstance(layer, nn.Linear):
            print('Layer {}, type: {}'.format(i, layer.__class__.__name__))
            w = layer.weight.data.numpy()  # 输出通道数*输入通道数
            b = layer.bias.data.numpy()  # 输出通道数
            y = []
            for j in range(w.shape[0]):
                result = dict()  # 存储所有输出通道表达式
                for k in range(w.shape[1]):  # x.shape[0]
                    for xi, wi in x[k].items():
                        result.setdefault(xi, 0)
                        result[xi] += wi * w[j][k]
                result.setdefault('1', 0)
                result['1'] += b[j]
                y.append(result)
            x_record.append(x)
            x = np.array(y)

        elif isinstance(layer, nn.Flatten):
            print('Layer {}, type: {}'.format(i, layer.__class__.__name__))
            c, h, w = x.shape
            y = np.zeros(c*h*w, dtype=dict)
            for i in range(c):
                for j in range(h):
                    for k in range(w):
                        rs = dict()
                        for xi, wi in x[i,j,k].items():
                            rs[xi] = wi
                        y[i*h*w+j*w+k] = rs
            x_record.append(x)
            x = y


        elif isinstance(layer, nn.ReLU):
            print('Layer {}, type: {}'.format(i, layer.__class__.__name__))
            if len(x.shape) == 3:
                y = np.zeros(x.shape, dtype=dict)
                new_x_true_l = np.zeros(x.shape[0] * x.shape[1] * x.shape[2], dtype=np.float32)
                new_x_true_h = np.zeros(x.shape[0] * x.shape[1] * x.shape[2], dtype=np.float32)
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        for k in range(x.shape[2]):
                            sum_l = 0.0
                            sum_h = 0.0
                            for xi, wi in x[i, j, k].items():
                                if xi == '1':
                                    sum_l += wi
                                    sum_h += wi
                                else:
                                    idx = int(xi[1:])
                                    if wi > 0:  # w>0
                                        sum_l += wi * x_true_l[idx]
                                        sum_h += wi * x_true_h[idx]
                                    else:
                                        sum_h += wi * x_true_l[idx]
                                        sum_l += wi * x_true_h[idx]
                            assert sum_l <= sum_h
                            # 重置x字典，设置仅剩1个x和w，同时更新x_true范围
                            y[i, j, k] = dict()
                            y[i, j, k]['x' + str(i * x.shape[2] * x.shape[1] + j * x.shape[2] + k)] = 1
                            new_x_true_l[i * x.shape[2] * x.shape[1] + j * x.shape[2] + k] = max(sum_l, 0.0)
                            new_x_true_h[i * x.shape[2] * x.shape[1] + j * x.shape[2] + k] = max(sum_h, 0.0)
                x_true_l = new_x_true_l
                x_true_h = new_x_true_h
            elif len(x.shape) == 1:
                y = np.zeros(x.shape[0], dtype=dict)
                new_x_true_l = np.zeros(x.shape[0], dtype=np.float32)
                new_x_true_h = np.zeros(x.shape[0], dtype=np.float32)
                for i in range(x.shape[0]):
                    sum_l, sum_h = 0.0, 0.0
                    for xi, wi in x[i].items():
                        if xi == '1':
                            sum_l += wi
                            sum_h += wi
                        else:
                            idx = int(xi[1:])
                            if wi > 0:
                                sum_l += wi * x_true_l[idx]
                                sum_h += wi * x_true_h[idx]
                            else:
                                sum_l += wi * x_true_h[idx]
                                sum_h += wi * x_true_l[idx]
                    assert sum_l <= sum_h
                    y[i] = dict()
                    y[i]['x' + str(i)] = 1
                    new_x_true_l[i] = max(sum_l, 0.0)
                    new_x_true_h[i] = max(sum_h, 0.0)
                x_true_l = new_x_true_l
                x_true_h = new_x_true_h
            x_record.append(x)
            x = y

        elif isinstance(layer, nn.Sigmoid):
            print('Layer {}, type: {}'.format(i, layer.__class__.__name__))
            if len(x.shape) == 3:
                y = np.zeros(x.shape, dtype=dict)
                new_x_true_l = np.zeros(x.shape[0] * x.shape[1] * x.shape[2], dtype=np.float32)
                new_x_true_h = np.zeros(x.shape[0] * x.shape[1] * x.shape[2], dtype=np.float32)
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        for k in range(x.shape[2]):
                            sum_l = 0.0
                            sum_h = 0.0
                            for xi, wi in x[i, j, k].items():
                                if xi == '1':
                                    sum_l += wi
                                    sum_h += wi
                                else:
                                    idx = int(xi[1:])
                                    if wi > 0:  # w>0
                                        sum_l += wi * x_true_l[idx]
                                        sum_h += wi * x_true_h[idx]
                                    else:
                                        sum_h += wi * x_true_l[idx]
                                        sum_l += wi * x_true_h[idx]
                            assert sum_l <= sum_h
                            # 重置x字典，设置仅剩1个x和w，同时更新x_true范围
                            y[i, j, k] = dict()
                            y[i, j, k]['x' + str(i * x.shape[2] * x.shape[1] + j * x.shape[2] + k)] = 1
                            new_x_true_l[i * x.shape[2] * x.shape[1] + j * x.shape[2] + k] = activations.sigmoid(sum_l)
                            new_x_true_h[i * x.shape[2] * x.shape[1] + j * x.shape[2] + k] = activations.sigmoid(sum_h)
                x_true_l = new_x_true_l
                x_true_h = new_x_true_h
            elif len(x.shape) == 1:
                y = np.zeros(x.shape[0], dtype=dict)
                new_x_true_l = np.zeros(x.shape[0], dtype=np.float32)
                new_x_true_h = np.zeros(x.shape[0], dtype=np.float32)
                for i in range(x.shape[0]):
                    sum_l, sum_h = 0.0, 0.0
                    for xi, wi in x[i].items():
                        if xi == '1':
                            sum_l += wi
                            sum_h += wi
                        else:
                            idx = int(xi[1:])
                            if wi > 0:
                                sum_l += wi * x_true_l[idx]
                                sum_h += wi * x_true_h[idx]
                            else:
                                sum_l += wi * x_true_h[idx]
                                sum_h += wi * x_true_l[idx]
                    assert sum_l <= sum_h
                    y[i] = dict()
                    y[i]['x' + str(i)] = 1
                    new_x_true_l[i] = activations.sigmoid(sum_l)
                    new_x_true_h[i] = activations.sigmoid(sum_h)
                x_true_l = new_x_true_l
                x_true_h = new_x_true_h
            x_record.append(x)
            x = y

        elif isinstance(layer, nn.Tanh):
            print('Layer {}, type: {}'.format(i, layer.__class__.__name__))
            if len(x.shape) == 3:
                y = np.zeros(x.shape, dtype=dict)
                new_x_true_l = np.zeros(x.shape[0] * x.shape[1] * x.shape[2], dtype=np.float32)
                new_x_true_h = np.zeros(x.shape[0] * x.shape[1] * x.shape[2], dtype=np.float32)
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        for k in range(x.shape[2]):
                            sum_l = 0.0
                            sum_h = 0.0
                            for xi, wi in x[i, j, k].items():
                                if xi == '1':
                                    sum_l += wi
                                    sum_h += wi
                                else:
                                    idx = int(xi[1:])
                                    if wi > 0:  # w>0
                                        sum_l += wi * x_true_l[idx]
                                        sum_h += wi * x_true_h[idx]
                                    else:
                                        sum_h += wi * x_true_l[idx]
                                        sum_l += wi * x_true_h[idx]
                            assert sum_l <= sum_h
                            # 重置x字典，设置仅剩1个x和w，同时更新x_true范围
                            y[i, j, k] = dict()
                            y[i, j, k]['x' + str(i * x.shape[2] * x.shape[1] + j * x.shape[2] + k)] = 1
                            new_x_true_l[i * x.shape[2] * x.shape[1] + j * x.shape[2] + k] = activations.tanh(sum_l)
                            new_x_true_h[i * x.shape[2] * x.shape[1] + j * x.shape[2] + k] = activations.tanh(sum_h)
                x_true_l = new_x_true_l
                x_true_h = new_x_true_h
            elif len(x.shape) == 1:
                y = np.zeros(x.shape[0], dtype=dict)
                new_x_true_l = np.zeros(x.shape[0], dtype=np.float32)
                new_x_true_h = np.zeros(x.shape[0], dtype=np.float32)
                for i in range(x.shape[0]):
                    sum_l, sum_h = 0.0, 0.0
                    for xi, wi in x[i].items():
                        if xi == '1':
                            sum_l += wi
                            sum_h += wi
                        else:
                            idx = int(xi[1:])
                            if wi > 0:
                                sum_l += wi * x_true_l[idx]
                                sum_h += wi * x_true_h[idx]
                            else:
                                sum_l += wi * x_true_h[idx]
                                sum_h += wi * x_true_l[idx]
                    assert sum_l <= sum_h
                    y[i] = dict()
                    y[i]['x' + str(i)] = 1
                    new_x_true_l[i] = activations.tanh(sum_l)
                    new_x_true_h[i] = activations.tanh(sum_h)
                x_true_l = new_x_true_l
                x_true_h = new_x_true_h
            x_record.append(x)
            x = y

        elif isinstance(layer, nn.BatchNorm2d):
            print('Layer {}, type: {}'.format(i, layer.__class__.__name__))
            mean = layer.running_mean.numpy()
            var = layer.running_var.numpy()
            gamma = layer.weight.data.numpy()
            beta = layer.bias.data.numpy()
            eps = layer.eps  # Avoids zero division

            y = np.zeros(x.shape, dtype=dict)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        y[i,j,k] = dict()
                        for xi, wi in x[i, j, k].items():
                            # ((x0*a + x1*b + c)-mean)/var*gamma+beta = x0*a*gamma/var + x1*b*gamma/var + c*gamma/var - mean*gamma/var + beta
                            y[i, j, k][xi] = wi * gamma[i] / np.sqrt(var[i] + eps)
                        y[i, j, k]['1'] = x[i, j, k]['1'] - mean[i] * gamma[i] / np.sqrt(var[i] + eps) + beta[i]
            x_record.append(x)
            x = y

        elif isinstance(layer, nn.MaxPool2d):
            print('Layer {}, type: {}'.format(i, layer.__class__.__name__))
            kernel_size = layer.kernel_size
            padding = layer.padding
            if padding == 0:
                padding = (0, 0)
            stride = layer.stride
            new_h = int((x.shape[1] + padding[0] * 2 - kernel_size[0]) / stride[0]) + 1  # 输出高
            new_w = int((x.shape[2] + padding[1] * 2 - kernel_size[1]) / stride[1]) + 1  # 输出宽
            y = np.zeros((x.shape[0], new_h, new_w), dtype=dict)
            x_pad = np.zeros((x.shape[0], x.shape[1] + padding[0] * 2, x.shape[2] + padding[1] * 2), dtype=dict)
            x_pad[:, padding[0]:padding[0] + x.shape[1], padding[1]:padding[1] + x.shape[2]] = x

            new_x_true_h = np.zeros(y.shape[0] * y.shape[1] * y.shape[2], dtype=np.float32)
            new_x_true_l = np.zeros(y.shape[0] * y.shape[1] * y.shape[2], dtype=np.float32)
            loc = np.zeros(y.shape[0] * y.shape[1] * y.shape[2], dtype=object)
            # 最大池层计算
            for i in range(y.shape[0]):  # x_pad, y的通道数
                for j in range(y.shape[1]):  # 输出高
                    h_start = j * stride[0]
                    h_end = h_start + kernel_size[0]
                    for k in range(y.shape[2]):  # 输出宽
                        w_start = k * stride[1]
                        w_end = w_start + kernel_size[1]
                        max_sum_l, max_sum_h = -np.inf, -np.inf
                        index_l = -1
                        index_h = -1
                        for l in range(h_start, h_end):  # y对应的x_pad的范围
                            for m in range(w_start, w_end):
                                sum_l, sum_h = 0.0, 0.0
                                if x_pad[i, l, m] == 0:
                                    # !!!!!!!!!!!max_pool填充部分为0，计算时不考虑
                                    # max_sum_l = max(0, max_sum_l)
                                    # max_sum_h = max(0, max_sum_h)
                                    continue
                                for xi, wi in x_pad[i, l, m].items():
                                    if xi == '1':
                                        sum_l += wi
                                        sum_h += wi
                                    else:
                                        idx = int(xi[1:])
                                        if wi > 0:
                                            sum_l += wi * x_true_l[idx]
                                            sum_h += wi * x_true_h[idx]
                                        else:
                                            sum_h += wi * x_true_l[idx]
                                            sum_l += wi * x_true_h[idx]
                                if sum_l > max_sum_l:
                                    max_sum_l = sum_l
                                    index_l = i*x.shape[1]*x.shape[2]+(l-padding[0])*x.shape[2]+(m-padding[1])
                                if sum_h > max_sum_h:
                                    max_sum_h = sum_h
                                    index_h = i*x.shape[1]*x.shape[2]+(l-padding[0])*x.shape[2]+(m-padding[1])
                        idx = i * y.shape[1] * y.shape[2] + j * y.shape[2] + k
                        y[i, j, k] = dict()
                        y[i, j, k]['x' + str(idx)] = 1
                        new_x_true_h[idx] = max_sum_h
                        new_x_true_l[idx] = max_sum_l
                        loc[idx] = [index_l, index_h]  # 这里记录输出表达式为x[i]的点，其输入上下界所取的点所在的一维位置为（index_l,index_h）

            x_true_h = new_x_true_h
            x_true_l = new_x_true_l
            x_record.append(x)
            x = y
            x_maxpool_loc.append(loc)


        elif isinstance(layer, nn.AvgPool2d):
            print('Layer {}, type: {}'.format(i, layer.__class__.__name__))
            kernel_size = layer.kernel_size
            padding = layer.padding
            if padding == 0:
                padding = (0, 0)
            stride = layer.stride
            new_h = int((x.shape[1] + padding[0] * 2 - kernel_size[0]) / stride[0]) + 1  # 输出高
            new_w = int((x.shape[2] + padding[1] * 2 - kernel_size[1]) / stride[1]) + 1  # 输出宽
            y = np.zeros((x.shape[0], new_h, new_w), dtype=dict)
            x_pad = np.zeros((x.shape[0], x.shape[1] + padding[0] * 2, x.shape[2] + padding[1] * 2), dtype=dict)
            x_pad[:, padding[0]:padding[0] + x.shape[1], padding[1]:padding[1] + x.shape[2]] = x
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    h_start = j * stride[0]
                    h_end = h_start + kernel_size[0]
                    for k in range(y.shape[2]):
                        w_start = k * stride[1]
                        w_end = w_start + kernel_size[1]
                        for l in range(h_start, h_end):
                            for m in range(w_start, w_end):
                                if x_pad[i, l, m] == 0:
                                    continue
                                for xi, wi in x_pad[i, l, m].items():
                                    if y[i, j, k] == 0:
                                        y[i, j, k] = dict()
                                    y[i, j, k].setdefault(xi, 0)
                                    y[i, j, k][xi] += wi
            div = float(kernel_size[0] * kernel_size[1])
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    for k in range(y.shape[2]):
                        for xi, wi in y[i, j, k].items():
                            y[i, j, k][xi] = wi / div
            x_record.append(x)
            x = y

        else:
            print('Ignore layer {}, type: {}'.format(i, layer.__class__.__name__))

    x_record.append(x)
    rs = verify_robustness(model, x_record, true_label, x_maxpool_loc, input, disturb_radius, dataset)

    end_time = time.time()
    print("Program running time: {}".format(end_time-start_time))
    print(rs)

# print(x)



