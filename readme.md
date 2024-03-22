# CNN-SYMBOL——基于符号区间的神经网络鲁棒性验证工具

1. 项目分别在MNIST数据集和CIFAR-10数据集上训练了一个简化的VGG模型，模型以及数据集、训练过程定义在
setup_mnist.py和setup_cifar10.py中，训练好的模型包存在model目录下
   
2. main.py文件为验证程序的主文件，其中主要实现了两个函数：run()函数实现了符号区间算法，它指定一个
   神经网络、训练神经网络的数据集、一张输入图像、图像对应的正确标签以及一个扰动值，程序判断对于这个扰动
   神经网络能否得到正确的结果。verify_robustness()函数实现了反向符号区间算法，它判断对于给定的错误
   输出，能否真的找到一个输入，使得经过神经网络后得到该输出。如果存在，说明找到了一个对抗性示例，网络
   不鲁棒，否则网络在该扰动下鲁棒。
   
3. test.py文件给出了调用main.py文件的方法，它们对应的图像存储在images目录下。

