import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

def show_images(data_loader):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    img = torchvision.utils.make_grid(images) / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
