"""
This model can reach test set loss: 0.2552 and accuracy: 0.9221 with naive early
stopping (stops if new validation accuracy is lower than earlier) random erasing
and gaussian noise.
"""

import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import transforms, datasets
from debug import *

NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'

N_EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.0001


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def load_dataset(dataset_name, shuffle=False, extra_transforms=[]):
    common_transforms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5, inplace=True)
    ]

    dataset = datasets.ImageFolder(
        DATA_DIR % dataset_name,
        transform=transforms.Compose(common_transforms + extra_transforms)
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

    return data_loader

class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.seq3 = nn.Sequential(
            nn.Linear(in_features=64 * 6 * 6, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=NUM_CLASSES)
        )
        self.fc1 = nn.Linear(32 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)
        x = x.view(x.size(0), -1)
        x = self.seq3(x)
        x = F.log_softmax(x, dim=1)
        return x


train_loader = load_dataset(
    'train',
    shuffle=True,
    extra_transforms=[
        #AddGaussianNoise(0.1, 0.1),
    ]
)

# show_images(train_loader)

dev_loader = load_dataset('dev')
test_loader = load_dataset('test')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()


def test(data_loader):
    cumulative_loss = 0
    correct_count = 0

    with torch.no_grad():
        for batch_index, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            prediction = model(data)
            _, predicted = torch.max(prediction.data, 1)

            correct_count += (predicted == target).sum()
            cumulative_loss += loss_function(prediction, target)

    accuracy = correct_count / len(data_loader.dataset)
    loss = cumulative_loss / len(data_loader)

    return accuracy, loss


def train():
    current_validation_accuracy = 0

    for epoch_index in range(N_EPOCHS):
        cumulative_loss = 0
        train_correct = 0
        total = len(train_loader.dataset)

        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            prediction = model(data)
            output = loss_function(prediction, target)

            cumulative_loss += output.data.item()
            _, predicted = torch.max(prediction, 1)
            train_correct += (predicted == target).sum().item()

            output.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('Epoch %d | batch %d %% done' %
                  (epoch_index + 1,
                   100 * (batch_index + 1) / len(train_loader),),
                  end="\r",
                  flush=True)

        new_validation_accuracy, new_validation_loss = test(dev_loader)

        print('Epoch %d | training acc %.4f, loss %.4f | validation acc %.4f, loss %.4f' %
              (epoch_index + 1,
               train_correct / total,
               cumulative_loss / len(train_loader),
               new_validation_accuracy,
               new_validation_loss))

        #if current_validation_accuracy > new_validation_accuracy:
        #    print('Old validation accuracy was greater, terminating.')
        #    break

        current_validation_accuracy = new_validation_accuracy


if __name__ == "__main__":
    train()

    accuracy, loss = test(test_loader)

    print('Test set loss: %.4f | accuracy: %.4f %%' % (loss, accuracy))
