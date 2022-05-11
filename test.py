import main

def test():
     main.run('MNIST', 'images/3.jpg', 0.1, 'model/MNIST_Model.pth', 3)
    # main.run("CIFAR10", 'images/c6.jpg', 0.1, 'model/CIFAR10_Model.pth', 6)
    # main.run('MNIST', 'images/4.jpg', 0.2, 'model/MNIST_Model.pth', 4)
    # main.run('MNIST', 'images/6.jpg', 0.5, 'model/MNIST_Model.pth', 6)
    # main.run('CIFAR10', 'images/c0.jpg', 0.2, 'model/CIFAR10_Model.pth', 0)
    # main.run('CIFAR10', 'images/c9.jpg', 0.5, 'model/CIFAR10_Model.pth', 9)



test()
