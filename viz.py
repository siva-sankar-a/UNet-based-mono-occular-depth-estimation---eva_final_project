import numpy as np
import matplotlib.pyplot as plt
from .settings import cifar10_classes

import torchvision

def viz_cifar10_grid(train_loader):
    def imshow(img):
        img = (img - img.min()) / (img.max() - img.min())     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()  
    # Modification comment

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % cifar10_classes[labels[j]] for j in range(4)))

def viz_cifar10_grid_(train_loader):
    def show(img):
        npimg = img.numpy()
        plt.figure(figsize = (15, 15))
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()  
    # Modification comment

    # show images
    show(torchvision.utils.make_grid(images))


def viz_cifar_single_image(train_loader):

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    img = images[0]
    img = (img - img.min()) / (img.max() - img.min())     # unnormalize
    plt.imshow(img.numpy().swapaxes(0, 2).swapaxes(0, 1))
    plt.show()