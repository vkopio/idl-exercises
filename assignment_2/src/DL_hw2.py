import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import transforms, datasets

NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'

N_EPOCHS = 10
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LEARNING_RATE = 0.01


def load_dataset(dataset_name, shuffle=False, extra_transforms=[]):
    common_transforms = [
        transforms.ToTensor()
    ]

    dataset = datasets.ImageFolder(
        DATA_DIR % dataset_name,
        transform=transforms.Compose(common_transforms + extra_transforms)
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=shuffle
    )

    return data_loader


class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


train_loader = load_dataset('train', shuffle=True, extra_transforms=[])
dev_loader = load_dataset('dev')
test_loader = load_dataset('test')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
loss_function = nn.CrossEntropyLoss()


def train():
    for epoch in range(N_EPOCHS):
        train_loss = 0
        train_correct = 0
        total = 0

        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            prediction = model(data)
            output = loss_function(prediction, target)

            train_loss += output.data.item()
            _, predicted = torch.max(prediction, 1)
            train_correct += (predicted == target).sum().item()
            total += target.size(0)

            output.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('Training: Epoch %d/%d | Batch %d/%d | Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
                  (epoch + 1,
                   N_EPOCHS,
                   batch_num,
                   len(train_loader),
                   train_loss / (batch_num + 1),
                   100. * train_correct / total,
                   train_correct,
                   total),
                  end="\r",
                  flush=True)

        # Please implement early stopping here.


def test():
    test_loss = 0
    test_correct = 0
    total = 0

    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            prediction = model(data)
            _, predicted = torch.max(prediction.data, 1)

            total += target.size(0)
            test_correct += (predicted == target).sum()
            test_loss += loss_function(prediction, target)

            print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' %
                  (batch_num + 1,
                   len(test_loader),
                   test_loss / (batch_num + 1),
                   100. * test_correct / total,
                   test_correct,
                   total))


if __name__ == "__main__":
    train()
    test()
